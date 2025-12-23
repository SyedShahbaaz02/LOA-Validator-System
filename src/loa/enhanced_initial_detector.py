"""
Enhanced Initial Box Detector
This module adds the ability to detect handwritten initials in document images
"""

import re


class EnhancedInitialDetector:
    """Enhanced Initial Box Detector Class.

    Provides enhanced detection of handwritten initials using drawing path analysis.
    """

    def __init__(self):
        pass

    def detect_handwritten_initials(self, extracted_text):
        """
        Detect handwritten initials in document text using enhanced methods

        Args:
            extracted_text: The OCR-extracted text from the document

        Returns:
            Dict containing detection results
        """
        detection_results = {
            "initial_boxes": [],
            "potential_initials": [],
            "drawing_paths": [],
            "detection_method": "enhanced_pattern_matching",
        }

        # Enhanced pattern matching for initial boxes
        initial_box_patterns = [
            r"Initial\s+[Bb]ox[^:]*:\s*\n?\s*([A-Za-z]{1,3})\s",
            r"^([A-Za-z]{1,3})\s+(?:Account|Interval|Historical|Energy|Usage|Data|Release)",
            r"[\[\]☐☑✓✗]\s*([A-Za-z]{1,3})\s",
            r"^([A-Za-z]{1,3})\s+(?:Account/SDI|Number Release|Interval Historical)",
            r"^([A-Z]{1,3})\s+Account/SDI\s+Number\s+Release",
            r"^([A-Z][+\-]?)\s+Interval\s+Historical",
            r"\n([A-Za-z0-9]{1,3})\nAccount/SDI",
            r"\n([A-Za-z0-9]{1,3})\nResidential",
        ]

        # Check for empty initial boxes
        underscore_patterns = [
            r"Initial[^:]*:\s*_{1,}",
            r"_{3,}\s*(?:Initial|Initials)",
        ]

        # Find empty boxes first
        for pattern in underscore_patterns:
            matches = re.finditer(pattern, extracted_text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                detection_results["initial_boxes"].append(
                    {
                        "type": "empty_box",
                        "text": "",
                        "context": match.group(0),
                        "is_filled": False,
                        "detection_method": "underscore_pattern",
                    }
                )

        # Find potential initial boxes with content
        for pattern in initial_box_patterns:
            matches = re.finditer(pattern, extracted_text, re.MULTILINE)
            for match in matches:
                initial_text = match.group(1) if match.lastindex else match.group(0)

                false_positives = [
                    "An",
                    "By",
                    "In",
                    "On",
                    "At",
                    "To",
                    "Of",
                    "Or",
                    "If",
                    "As",
                    "Is",
                    "It",
                    "We",
                    "ID",
                ]
                if initial_text.strip() not in false_positives:
                    start = max(0, match.start() - 50)
                    end = min(len(extracted_text), match.end() + 100)
                    context = extracted_text[start:end].replace("\n", " ")

                    initial_box = {
                        "type": "filled_box",
                        "text": initial_text.strip(),
                        "context": context,
                        "is_filled": True,
                        "is_likely_initial": len(initial_text.strip()) <= 3,
                        "detection_method": f"pattern_{initial_box_patterns.index(pattern) + 1}",
                    }

                    detection_results["initial_boxes"].append(initial_box)

                    # Also add to potential_initials for backward compatibility
                    detection_results["potential_initials"].append(
                        {
                            "text": initial_text.strip(),
                            "is_likely_initial": len(initial_text.strip()) <= 3,
                            "context": context,
                        }
                    )

        # Look for drawing paths or handwritten content indicators
        handwritten_indicators = [
            "signature",
            "initials",
            "initial here",
            "signed by",
            "authorized signature",
            "handwritten",
            "written by hand",
        ]

        for indicator in handwritten_indicators:
            if indicator in extracted_text.lower():
                detection_results["drawing_paths"].append(
                    {
                        "type": "handwritten_indicator",
                        "text": indicator,
                        "detection_method": "text_pattern_match",
                    }
                )

        return detection_results
