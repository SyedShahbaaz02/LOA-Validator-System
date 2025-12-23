"""
Document Integrity Checker for LOA Documents

This module validates the structural integrity and continuity of LOA documents,
detecting issues such as:
- Interleaved pages from multiple documents
- Non-contiguous text flow
- Missing or corrupted content
- Page ordering problems
- Text fragmentation

TWO-LAYER APPROACH:
1. Layer 1: Fast text-based heuristics (catches obvious corruption)
2. Layer 2: GPT-4o Vision fallback (catches subtle interleaving in borderline cases)

Author: System
Date: 2025-01-07
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class IntegrityIssue:
    """Represents a document integrity issue"""

    severity: str  # 'CRITICAL', 'WARNING', 'INFO'
    category: str  # Type of issue
    description: str
    evidence: str  # Supporting text/data
    page_number: Optional[int] = None

    def __post_init__(self):
        """Validate severity level"""
        valid_severities = {"CRITICAL", "WARNING", "INFO"}
        if self.severity not in valid_severities:
            raise ValueError(
                f"Invalid severity: {self.severity}. Must be one of {valid_severities}"
            )


class DocumentIntegrityChecker:
    """
    Validates LOA document integrity by analyzing text continuity,
    structural consistency, and content completeness.

    TWO-LAYER VALIDATION:
    1. Fast text-based heuristics (always runs)
    2. GPT-4o Vision fallback (runs for borderline cases)
    Args:
        min_confidence: Minimum confidence threshold for flagging issues (0.0-1.0)
        gpt4o_verification_integration: Optional GPT-4o integration for visual verification

    Raises:
        ValueError: If min_confidence is not between 0.0 and 1.0
    Args:
        min_confidence: Minimum confidence threshold for flagging issues (0.0-1.0)
        gpt4o_verification_integration: Optional GPT-4o integration for visual verification

    Raises:
        ValueError: If min_confidence is not between 0.0 and 1.0
    """

    # Compile regex patterns once at class level for performance
    SINGLE_LETTER_PATTERN = re.compile(r"\b[A-Z]\s*\n")
    EXCESSIVE_SPACING_PATTERN = re.compile(r"[a-zA-Z]\s{5,}[a-zA-Z]")
    SPECIAL_CHAR_SEQUENCE_PATTERN = re.compile(r"[^a-zA-Z0-9\s]{8,}")
    UNUSUAL_CHAR_PATTERN = re.compile(r"[^\x00-\x7F]{5,}")
    SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]\s+")
    FORM_LABEL_PATTERN = re.compile(
        r"(Name|Address|Date|Phone|Email|Account|Signature|Title):", re.IGNORECASE
    )

    def __init__(
        self,
        min_confidence: float = 0.7,
        gpt4o_verification_integration: Optional[Any] = None,
    ):
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence must be between 0.0 and 1.0, got {min_confidence}"
            )

        self.min_confidence = min_confidence
        self.issues: List[IntegrityIssue] = []
        self.gpt4o_verification = gpt4o_verification_integration

    def check_document_integrity(
        self,
        extracted_text: str,
        ocr_result: Optional[Dict] = None,
        pdf_path: Optional[str] = None,
    ) -> Dict:
        """
        Main entry point for document integrity checking with two-layer validation.

        LAYER 1: Fast text-based heuristics (always runs)
        LAYER 2: GPT-4o Vision (runs for borderline cases with 0.7-0.95 confidence)

        Args:
            extracted_text: Full text extracted from document
            ocr_result: Optional OCR result object with page-level data
            pdf_path: Optional PDF path for GPT-4o Vision verification

        Returns:
            Dictionary containing:
                - is_valid: bool (True if document passes all checks)
                - confidence: float (0.0-1.0)
                - issues: List of detected issues
                - summary: Overall assessment
                - gpt4o_verified: bool (True if GPT-4o Vision was used)

        Raises:
            ValueError: If extracted_text is empty or None
        """
        if not extracted_text or not extracted_text.strip():
            raise ValueError("extracted_text cannot be empty or None")

        self.issues = []

        logger.info(
            "Starting document integrity check (Layer 1: Text-based heuristics)..."
        )

        # LAYER 1: Run all text-based integrity checks
        self._check_interleaved_text_corruption(
            extracted_text
        )  # NEW: Check for severely interleaved text
        self._check_text_fragmentation(extracted_text)
        self._check_sentence_continuity(extracted_text)
        self._check_required_sections(extracted_text)
        self._check_repeated_fragments(extracted_text)
        self._check_character_corruption(extracted_text)

        # Check page-level integrity if OCR result available
        if ocr_result:
            self._check_page_continuity(ocr_result)
            self._check_page_confidence(ocr_result)

        # Calculate overall validity
        critical_issues = [i for i in self.issues if i.severity == "CRITICAL"]
        warning_issues = [i for i in self.issues if i.severity == "WARNING"]

        is_valid = len(critical_issues) == 0
        confidence = self._calculate_confidence_score()

        # LAYER 2: GPT-4o Vision verification (ALWAYS runs when available for maximum accuracy)
        gpt4o_verification_result = None

        if self.gpt4o_verification and pdf_path:
            logger.info(
                f"Layer 1 complete (confidence={confidence:.2f}, warnings={len(warning_issues)}) - "
                f"Running Layer 2: GPT-4o Vision verification for maximum accuracy..."
            )
            gpt4o_verification_result = self._verify_with_gpt4o_vision(pdf_path)

            # If GPT-4o detects interleaving, override text-based result
            if gpt4o_verification_result and not gpt4o_verification_result["is_valid"]:
                logger.warning(
                    "GPT-4o Vision detected issues that text heuristics missed!"
                )
                is_valid = False
                confidence = min(confidence, gpt4o_verification_result["confidence"])

                # Add GPT-4o detected issues
                for gpt4o_issue in gpt4o_verification_result["issues"]:
                    self.issues.append(
                        IntegrityIssue(
                            severity="CRITICAL",
                            category="GPT4O_VISUAL_DETECTION",
                            description=gpt4o_issue["description"],
                            evidence=gpt4o_issue["evidence"],
                        )
                    )

                # Recalculate with new issues
                critical_issues = [i for i in self.issues if i.severity == "CRITICAL"]

        result = {
            "is_valid": is_valid,
            "confidence": confidence,
            "issues": [
                {
                    "severity": issue.severity,
                    "category": issue.category,
                    "description": issue.description,
                    "evidence": issue.evidence[:200],  # Truncate evidence
                    "page_number": issue.page_number,
                }
                for issue in self.issues
            ],
            "summary": self._generate_summary(
                is_valid, critical_issues, warning_issues
            ),
            "critical_count": len(critical_issues),
            "warning_count": len(warning_issues),
            "gpt4o_verified": gpt4o_verification_result is not None,
            "gpt4o_verification_details": (
                gpt4o_verification_result if gpt4o_verification_result else None
            ),
        }

        logger.info(
            f"Integrity check complete: Valid={is_valid}, Confidence={confidence:.2f}, "
            f"Issues={len(critical_issues)} critical, {len(warning_issues)} warnings, "
            f"GPT-4o Verified={result['gpt4o_verified']}"
        )

        return result

    def _verify_with_gpt4o_vision(self, pdf_path: str) -> Optional[Dict]:
        """
        Use GPT-4o Vision to visually verify document integrity.
        Detects interleaved pages, misaligned content, and corruption.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict with verification results or None if verification fails
        """
        try:
            if not self.gpt4o_verification:
                logger.warning("GPT-4o verification integration not available")
                return None

            # Call GPT-4o Vision integration
            result = self.gpt4o_verification.verify_document_integrity_with_gpt4o(
                pdf_path
            )

            if result.get("success"):
                return {
                    "is_valid": result.get("is_valid", True),
                    "confidence": result.get("confidence", 1.0),
                    "issues": result.get("issues", []),
                    "reasoning": result.get("reasoning", ""),
                }
            else:
                logger.warning(f"GPT-4o verification failed: {result.get('error')}")
                return None

        except Exception as e:
            logger.error(f"GPT-4o Vision verification error: {str(e)}")
            return None

    def _check_interleaved_text_corruption(self, text: str) -> None:
        """
        Detect severely interleaved text corruption where text from multiple columns/sections
        is mixed together in an unreadable way.

        Example from corrupted doc:
        - "when your supply service with C" (line break) "NE You have"
        - "Constellation NewEnergy, Inc("CNE")    information   ("EUI")   for   purposes"

        This is MORE SENSITIVE than other checks because interleaved text is a critical issue
        that makes documents completely unreadable, but still uses reasonable thresholds
        to avoid false positives on legitimate forms.
        """
        issues_found = []
        issue_count = 0

        # Pattern 1: Words broken mid-word across lines (e.g., "with C\nNE You")
        # Look for single/double capital letters at end of line followed by capital letters starting next line
        lines = text.split("\n")
        broken_word_pattern = re.compile(r"\b[A-Z]{1,2}\s*$")
        next_word_capital_pattern = re.compile(r"^\s*[A-Z]{2,}")

        broken_words = []
        for i in range(len(lines) - 1):
            current_line = lines[i].strip()
            next_line = lines[i + 1].strip()

            if broken_word_pattern.search(
                current_line
            ) and next_word_capital_pattern.match(next_line):
                # Check if this looks like a broken word (not a form label or section header)
                if len(current_line) > 10 and not current_line.endswith(":"):
                    broken_words.append(f"{current_line[-20:]}|{next_line[:20]}")
                    issue_count += 1

        if (
            len(broken_words) >= 2
        ):  # Lowered from 3 to 2 - even 2 broken words is unusual
            issues_found.append(
                f"Detected {len(broken_words)} broken words across lines"
            )

        # Pattern 2: Excessive multi-space gaps within text lines (5+ spaces between words)
        # This indicates interleaved columns
        multi_space_pattern = re.compile(r"[a-zA-Z]\s{5,}[a-zA-Z]")
        multi_space_pattern.findall(text)

        # Filter out form fields (lines with colons or short lines)
        multi_space_in_sentences = []
        for line in lines:
            if ":" not in line and len(line) > 40:  # Not a form field, substantial text
                matches = multi_space_pattern.findall(line)
                if matches:
                    multi_space_in_sentences.extend(matches)
                    issue_count += 1

        if (
            len(multi_space_in_sentences) >= 3
        ):  # Lowered from 5 to 3 - this is very unusual
            issues_found.append(
                f"Detected {len(multi_space_in_sentences)} excessive spacing gaps in text"
            )

        # Pattern 3: Many very short fragments (1-3 char words) in the middle of text
        # Indicates text is being chopped up
        words = re.findall(r"\b\w+\b", text)
        very_short_words = [w for w in words if 1 <= len(w) <= 2 and w.isupper()]

        # Calculate ratio of very short words to total words
        if len(words) > 0:
            short_word_ratio = len(very_short_words) / len(words)
            if (
                short_word_ratio > 0.05
            ):  # More than 5% of words are 1-2 char uppercase fragments
                issues_found.append(
                    f"Detected high ratio ({short_word_ratio:.1%}) of short uppercase fragments"
                )
                issue_count += 1

        # Pattern 4: Sentences that start lowercase (indicating broken sentences)
        lowercase_start_sentences = 0
        for line in lines:
            line_stripped = line.strip()
            # Check if line is substantial and starts with lowercase
            if len(line_stripped) > 20 and line_stripped[0].islower():
                # Not a continuation word like "and", "or", "the"
                first_word = line_stripped.split()[0]
                if first_word.lower() not in [
                    "and",
                    "or",
                    "the",
                    "a",
                    "an",
                    "of",
                    "to",
                    "for",
                    "in",
                    "on",
                    "at",
                    "by",
                ]:
                    lowercase_start_sentences += 1
                    issue_count += 1

        if (
            lowercase_start_sentences >= 5
        ):  # At least 5 sentences starting with lowercase
            issues_found.append(
                f"Detected {lowercase_start_sentences} text lines starting with lowercase"
            )

        # Only flag as CRITICAL if MULTIPLE patterns detected (not just one)
        # Lowered threshold: 2+ patterns AND 5+ total issues (was 8)
        if len(issues_found) >= 2 and issue_count >= 5:
            self.issues.append(
                IntegrityIssue(
                    severity="CRITICAL",
                    category="INTERLEAVED_TEXT_CORRUPTION",
                    description="Severe text interleaving detected - document appears to have mixed/overlapping text columns making it unreadable",
                    evidence=f'{"; ".join(issues_found)}. Examples: {" | ".join(broken_words[:2]) if broken_words else "N/A"}',
                )
            )

    def _check_text_fragmentation(self, text: str) -> None:
        """
        Detect text fragmentation - incomplete words, broken sentences.
        Example from corrupted doc: "when your supply service with C", "NE You have"

        MUCH LESS SENSITIVE: Only flag extreme cases
        """
        # Pattern 1: Single capital letters followed by newline (potential page break issues)
        # INCREASED THRESHOLD: Only flag if overwhelming evidence
        matches = self.SINGLE_LETTER_PATTERN.findall(text)
        if len(matches) > 50:  # Increased from 5 to 50 - must be extreme
            self.issues.append(
                IntegrityIssue(
                    severity="CRITICAL",
                    category="TEXT_FRAGMENTATION",
                    description="Detected instances of single capital letters with line breaks - possible text fragmentation",
                    evidence=f'Examples: {", ".join(matches[:3])}',
                )
            )

        # Pattern 2: Incomplete sentences (very short lines that don't end with punctuation)
        # INCREASED THRESHOLD: Only flag if overwhelming evidence
        lines = text.split("\n")
        fragment_count = 0
        fragment_examples = []

        for line in lines:
            line_stripped = line.strip()
            # Check if line is between 5-30 chars, doesn't end with punctuation, and contains letters
            if (
                5 < len(line_stripped) < 30
                and line_stripped[-1] not in ".!?:;"
                and any(c.isalpha() for c in line_stripped)
            ):
                # Exclude common form labels
                if not self.FORM_LABEL_PATTERN.search(line_stripped):
                    fragment_count += 1
                    if len(fragment_examples) < 3:
                        fragment_examples.append(line_stripped)

        # Only flag if EXTREME fragmentation (75+ fragments)
        if fragment_count > 75:  # Increased from 15 to 75
            self.issues.append(
                IntegrityIssue(
                    severity="CRITICAL",
                    category="TEXT_FRAGMENTATION",
                    description="Detected incomplete text fragments - document may be corrupted",
                    evidence=f'Examples: {" | ".join(fragment_examples)}',
                )
            )
        # Removed WARNING level entirely - too sensitive

    def _check_sentence_continuity(self, text: str) -> None:
        """
        Check for broken sentences and non-contiguous flow.
        Example: "Constellation NewEnergy, Inc("CNE")    information   ("EUI")   for   purposes"

        MUCH LESS SENSITIVE: Only flag extreme cases
        """
        # Pattern: Excessive spacing or punctuation issues
        # INCREASED THRESHOLD: Must be overwhelming to flag
        spacing_issues = self.EXCESSIVE_SPACING_PATTERN.findall(text)
        if len(spacing_issues) > 50:  # Increased from 10 to 50
            self.issues.append(
                IntegrityIssue(
                    severity="CRITICAL",
                    category="NON_CONTIGUOUS_FLOW",
                    description=f"Detected {len(spacing_issues)} instances of excessive spacing - text flow disrupted",
                    evidence=f'Examples: {" | ".join([s.strip() for s in spacing_issues[:3]])}',
                )
            )

        # Pattern: Parentheses or quotes not properly closed
        # INCREASED THRESHOLD: Must be very unbalanced
        open_parens = text.count("(")
        close_parens = text.count(")")
        if abs(open_parens - close_parens) > 30:  # Increased from 5 to 30
            self.issues.append(
                IntegrityIssue(
                    severity="WARNING",
                    category="NON_CONTIGUOUS_FLOW",
                    description=f"Unbalanced parentheses: {open_parens} open, {close_parens} close - possible text corruption",
                    evidence=f"Difference: {abs(open_parens - close_parens)}",
                )
            )

    def _check_required_sections(self, text: str) -> None:
        """
        Verify presence of required LOA sections.
        Missing sections indicate incomplete or interleaved documents.

        MUCH LESS SENSITIVE: Only flag if ALL sections are missing
        """
        # Common LOA section indicators (compiled once)
        required_sections = {
            "customer_info": [
                re.compile(r"Customer\s+Name", re.IGNORECASE),
                re.compile(r"Customer\s+Information", re.IGNORECASE),
                re.compile(r"To\s+be\s+completed\s+by\s+[Cc]ustomer"),
            ],
            "authorization": [
                re.compile(r"AUTHORIZATION", re.IGNORECASE),
                re.compile(r"[Aa]uthorize[sd]?\s+[Pp]erson"),
                re.compile(r"[Ss]ignature"),
            ],
            "account_info": [
                re.compile(r"[Aa]ccount\s+[Nn]umber"),
                re.compile(r"[Uu]tility"),
                re.compile(r"[Ss]ervice\s+[Aa]ddress"),
            ],
        }

        missing_sections = []
        for section_name, patterns in required_sections.items():
            if not any(pattern.search(text) for pattern in patterns):
                missing_sections.append(section_name)

        # Only flag as CRITICAL if ALL sections are missing (completely corrupted)
        if len(missing_sections) >= 3:  # Changed from 2 to 3 - must be ALL sections
            self.issues.append(
                IntegrityIssue(
                    severity="CRITICAL",
                    category="MISSING_SECTIONS",
                    description=f'Missing ALL {len(missing_sections)} required sections: {", ".join(missing_sections)}',
                    evidence="Document appears completely corrupted or interleaved",
                )
            )
        # Removed WARNING level - too sensitive

    def _check_repeated_fragments(self, text: str) -> None:
        """
        Detect repeated text fragments that suggest page duplication or interleaving.

        MUCH LESS SENSITIVE: Only flag extreme duplication
        """
        # Split into sentences
        sentences = self.SENTENCE_SPLIT_PATTERN.split(text)
        sentence_counts = {}

        for sentence in sentences:
            sentence_clean = sentence.strip().lower()
            if len(sentence_clean) > 30:  # Only check substantial sentences
                if sentence_clean in sentence_counts:
                    sentence_counts[sentence_clean] += 1
                else:
                    sentence_counts[sentence_clean] = 1

        # Find duplicates
        duplicates = {s: c for s, c in sentence_counts.items() if c > 1}

        # Only flag if EXTREME duplication (30+ repeated segments)
        if len(duplicates) > 30:  # Increased from 3 to 30
            evidence_sentences = list(duplicates.keys())[:2]
            self.issues.append(
                IntegrityIssue(
                    severity="WARNING",
                    category="REPEATED_CONTENT",
                    description=f"Detected {len(duplicates)} repeated text segments - possible page duplication",
                    evidence=f'Examples: {" | ".join([s[:50] for s in evidence_sentences])}',
                )
            )

    def _check_character_corruption(self, text: str) -> None:
        """
        Detect character-level corruption or encoding issues.

        MUCH LESS SENSITIVE: Only flag extreme corruption
        """
        # Pattern 1: Excessive special characters in sequence
        # INCREASED THRESHOLD: Must be overwhelming
        special_char_sequences = self.SPECIAL_CHAR_SEQUENCE_PATTERN.findall(text)
        if len(special_char_sequences) > 50:  # Increased from 10 to 50
            self.issues.append(
                IntegrityIssue(
                    severity="WARNING",
                    category="CHARACTER_CORRUPTION",
                    description=f"Detected {len(special_char_sequences)} sequences of special characters",
                    evidence=f'Examples: {" | ".join(special_char_sequences[:3])}',
                )
            )

        # Pattern 2: Unusual character patterns (potential encoding issues)
        # INCREASED THRESHOLD: Must be overwhelming
        unusual_chars = self.UNUSUAL_CHAR_PATTERN.findall(text)
        if len(unusual_chars) > 30:  # Increased from 5 to 30
            self.issues.append(
                IntegrityIssue(
                    severity="WARNING",
                    category="CHARACTER_CORRUPTION",
                    description=f"Detected {len(unusual_chars)} sequences of unusual characters - possible encoding issues",
                    evidence="Found non-ASCII character sequences",
                )
            )

    def _check_page_continuity(self, ocr_result: Dict) -> None:
        """
        Check page-level continuity using OCR page data.
        Detects potential page ordering or interleaving issues.
        """
        if not ocr_result or "pages" not in ocr_result:
            return

        pages = ocr_result.get("pages", [])
        if len(pages) < 2:
            return  # Can't check continuity with single page

        # Check if page numbers are sequential
        page_numbers = []
        for page in pages:
            if "page_number" in page:
                page_numbers.append(page["page_number"])

        if page_numbers:
            # Check for gaps or duplicates
            if len(page_numbers) != len(set(page_numbers)):
                self.issues.append(
                    IntegrityIssue(
                        severity="CRITICAL",
                        category="PAGE_ORDER_ISSUE",
                        description="Duplicate page numbers detected - possible interleaved pages",
                        evidence=f"Page numbers: {page_numbers}",
                    )
                )

            # Check for large gaps
            sorted_pages = sorted(page_numbers)
            for i in range(len(sorted_pages) - 1):
                if sorted_pages[i + 1] - sorted_pages[i] > 1:
                    self.issues.append(
                        IntegrityIssue(
                            severity="WARNING",
                            category="PAGE_ORDER_ISSUE",
                            description=f"Gap in page sequence: {sorted_pages[i]} to {sorted_pages[i+1]}",
                            evidence="Pages may be missing or out of order",
                        )
                    )

    def _check_page_confidence(self, ocr_result: Dict) -> None:
        """
        Check OCR confidence scores to detect potential quality issues.

        MUCH LESS SENSITIVE: Only flag extremely low confidence
        """
        if not ocr_result or "pages" not in ocr_result:
            return

        low_confidence_pages: List[Tuple[int, float]] = []

        for page_idx, page in enumerate(ocr_result["pages"]):
            if "lines" in page:
                confidences = [
                    line.get("confidence", 1.0)
                    for line in page["lines"]
                    if "confidence" in line
                ]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    if (
                        avg_confidence < 0.3
                    ):  # Decreased from 0.7 to 0.3 - only flag if VERY low
                        low_confidence_pages.append((page_idx + 1, avg_confidence))

        # Only flag if MULTIPLE pages have VERY low confidence
        if len(low_confidence_pages) >= 3:  # Increased from 1 to 3 pages
            self.issues.append(
                IntegrityIssue(
                    severity="WARNING",
                    category="LOW_OCR_QUALITY",
                    description=f"{len(low_confidence_pages)} pages have extremely low OCR confidence",
                    evidence=f'Pages: {", ".join([f"Page {p}:{c:.2f}" for p, c in low_confidence_pages[:3]])}',
                )
            )

    def _calculate_confidence_score(self) -> float:
        """
        Calculate overall confidence score based on detected issues.

        The score decreases based on the number and severity of issues found.
        Each CRITICAL issue reduces confidence by 0.15, WARNING by 0.05, and INFO by 0.01.

        Returns:
            Float between 0.0 and 1.0 (1.0 = highest confidence, no issues)
        """
        if not self.issues:
            return 1.0

        # Weight issues by severity
        SEVERITY_WEIGHTS = {"CRITICAL": 0.15, "WARNING": 0.05, "INFO": 0.01}

        penalty = sum(
            SEVERITY_WEIGHTS.get(issue.severity, 0.05) for issue in self.issues
        )

        # Ensure confidence stays within [0.0, 1.0] range
        confidence = max(0.0, min(1.0, 1.0 - penalty))

        return round(confidence, 2)

    def _generate_summary(
        self,
        is_valid: bool,
        critical_issues: List[IntegrityIssue],
        warning_issues: List[IntegrityIssue],
    ) -> str:
        """Generate human-readable summary of integrity check."""
        if is_valid and not warning_issues:
            return "Document integrity verified - no issues detected"

        if is_valid and warning_issues:
            return f"Document appears valid but has {len(warning_issues)} warning(s) - review recommended"

        # Document has critical issues
        issue_desc = []
        if critical_issues:
            categories = {issue.category for issue in critical_issues}
            issue_desc.append(
                f"{len(critical_issues)} critical issue(s) ({', '.join(sorted(categories))})"
            )
        if warning_issues:
            issue_desc.append(f"{len(warning_issues)} warning(s)")

        return f"DOCUMENT INTEGRITY COMPROMISED: {'; '.join(issue_desc)}"
