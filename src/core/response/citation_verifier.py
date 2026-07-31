"""Citation Verifier — check each claim in a generated answer against source texts.

Splits the answer into individual claims (sentences), then for each claim
checks whether it has supporting evidence in the retrieved source chunks.
Uses keyword overlap for fast first-pass verification — no extra API calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class VerifiedClaim:
    """A single claim with its verification result.

    Attributes:
        claim: The claim sentence.
        verified: True if supported by at least one source chunk.
        evidence: The best-matching source text snippet, or empty.
        source_num: Which source [N] supports it, or 0.
    """

    claim: str
    verified: bool = False
    evidence: str = ""
    source_num: int = 0


@dataclass
class VerificationReport:
    """Full citation verification report for one answer.

    Attributes:
        claims: Per-claim verification results.
        total_claims: Total number of claims checked.
        verified_count: Number of claims with supporting evidence.
    """

    claims: List[VerifiedClaim] = field(default_factory=list)

    @property
    def total_claims(self) -> int:
        return len(self.claims)

    @property
    def verified_count(self) -> int:
        return sum(1 for c in self.claims if c.verified)

    @property
    def verification_rate(self) -> float:
        """Fraction of claims verified (0.0 – 1.0)."""
        if not self.claims:
            return 1.0
        return self.verified_count / len(self.claims)

    def to_text(self) -> str:
        """Human-readable report."""
        lines = [
            f"引用核查报告: {self.verified_count}/{self.total_claims} 条声明可验证",
            f"验证率: {self.verification_rate:.1%}",
            "",
        ]
        for i, c in enumerate(self.claims, 1):
            icon = "✅" if c.verified else "⚠️"
            lines.append(f"{icon} 声明 {i}: {c.claim[:80]}...")
            if c.verified:
                lines.append(f"   证据来源 [{c.source_num}]: {c.evidence[:100]}...")
            else:
                lines.append(f"   ❌ 未找到原文依据")
        return "\n".join(lines)


class CitationVerifier:
    """Verify that each claim in an answer is supported by source chunks.

    Uses keyword overlap (Jaccard-like) — fast, no LLM calls.

    Example::

        verifier = CitationVerifier()
        report = verifier.verify(
            answer="FinFET是三维晶体管[1]。它的栅极宽度约14nm[2]。",
            sources=[{"num": "1", "text": "FinFET是三维..."}],
            chunks=[chunk1, chunk2],
        )
        print(report.to_text())
    """

    def verify(
        self,
        answer: str,
        chunks: List[Any],
        min_overlap: float = 0.10,
    ) -> VerificationReport:
        """Verify claims in answer against source chunks.

        Args:
            answer: The LLM-generated answer text.
            chunks: Original RetrievedResult objects (same as used to generate).
            min_overlap: Minimum keyword overlap ratio to count as verified.

        Returns:
            VerificationReport with per-claim results.
        """
        # Split into claims (sentences, stripped of citation markers)
        claims = self._split_claims(answer)
        if not claims:
            return VerificationReport()

        # Extract source texts with their numbers
        source_texts: List[tuple[int, str]] = []
        for i, chunk in enumerate(chunks, 1):
            text = getattr(chunk, "text", "") if hasattr(chunk, "text") else str(chunk)
            source_texts.append((i, text))

        # Verify each claim
        results: List[VerifiedClaim] = []
        for claim in claims:
            verified, evidence, src_num = self._verify_claim(claim, source_texts, min_overlap)
            results.append(VerifiedClaim(
                claim=claim,
                verified=verified,
                evidence=evidence,
                source_num=src_num,
            ))

        return VerificationReport(claims=results)

    # ── internal ──────────────────────────────────────────────────

    def _split_claims(self, text: str) -> List[str]:
        """Split answer text into individual claims (sentences)."""
        # Remove citation markers [1], [2] etc.
        clean = re.sub(r"\[\d+\]", "", text)
        # Split on Chinese/English sentence boundaries
        raw = re.split(r"[。！？\.!?\n]+", clean)
        return [s.strip() for s in raw if len(s.strip()) > 5]

    def _verify_claim(
        self,
        claim: str,
        sources: List[tuple[int, str]],
        min_overlap: float,
    ) -> tuple[bool, str, int]:
        """Check if a single claim is supported by any source.

        Returns:
            (verified, best_evidence_text, source_number)
        """
        claim_keywords = self._tokenize(claim)
        if not claim_keywords:
            return False, "", 0

        best_score = 0.0
        best_evidence = ""
        best_src_num = 0

        for src_num, src_text in sources:
            src_keywords = self._tokenize(src_text)
            if not src_keywords:
                continue
            # Jaccard-like overlap
            overlap = len(claim_keywords & src_keywords)
            score = overlap / len(claim_keywords)
            if score > best_score:
                best_score = score
                best_evidence = src_text
                best_src_num = src_num

        if best_score >= min_overlap:
            return True, best_evidence, best_src_num
        return False, "", 0

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Extract meaningful tokens from Chinese/English text."""
        # Chinese: character bigrams. English: lowercase words.
        tokens: set[str] = set()
        # Add English words (len >= 2)
        for w in re.findall(r"[a-zA-Z]{2,}", text):
            tokens.add(w.lower())
        # Add Chinese bigrams
        chinese_chars = re.findall(r"[一-鿿]", text)
        for i in range(len(chinese_chars) - 1):
            tokens.add(chinese_chars[i] + chinese_chars[i + 1])
        return tokens
