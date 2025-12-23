"""
Enhanced Selection Mark and Initial Box Validation Module
This module adds additional validation logic to detect:
1. X marks in initial boxes (which are not valid for Ohio documents)
2. Unselected required checkboxes
"""

import re  # Import for regex pattern matching


class EnhancedSelectionValidator:
    """
    Enhanced Selection Mark and Initial Box Validator Class
    Wraps the enhanced validation functions into a class interface

    Args:
        region: The region for validation rules ("Great Lakes" or "New England")
    """

    def __init__(self, region="Great Lakes"):
        self.region = region
        # Map region to default state
        if region == "New England":
            self.default_state = "MA"
        else:
            self.default_state = "OH"

    def validate_selection_marks(self, selection_marks, extracted_text):
        """
        Validate selection marks and detect issues like X marks or empty boxes

        Args:
            selection_marks: List of selection marks from OCR
            extracted_text: Full document text

        Returns:
            Dict containing validation results
        """
        # Use the state determined by region
        state = self.default_state

        validation_results = {
            "x_marks_found": 0,
            "valid_initials_found": 0,
            "empty_boxes_found": 0,
            "issues": [],
            "analysis": "",
        }

        # Count different types of marks
        selected_marks = [
            mark for mark in selection_marks if mark.get("state") == "selected"
        ]
        unselected_marks = [
            mark for mark in selection_marks if mark.get("state") == "unselected"
        ]

        validation_results["empty_boxes_found"] = len(unselected_marks)

        # Check for X marks in text
        x_mark_patterns = [
            r"[Xx]\s+(?:Account|Interval|Historical)",
            r":selected:\s*[Xx]",
            r"Initial\s+Box[^:]*:\s*[Xx]",
        ]

        x_marks_in_text = 0
        for pattern in x_mark_patterns:
            matches = re.findall(pattern, extracted_text, re.IGNORECASE)
            x_marks_in_text += len(matches)

        validation_results["x_marks_found"] = x_marks_in_text

        # Look for valid initials (letters, not X)
        initial_patterns = [
            r"Initial\s+Box[^:]*:\s*([A-Za-z]{1,3})\s",
            r"^([A-Za-z]{1,3})\s+(?:Account|Interval|Historical)",
        ]

        valid_initials = 0
        for pattern in initial_patterns:
            matches = re.findall(pattern, extracted_text, re.MULTILINE)
            for match in matches:
                if match.upper() != "X" and match not in [
                    "An",
                    "By",
                    "In",
                    "On",
                    "At",
                    "To",
                ]:
                    valid_initials += 1

        validation_results["valid_initials_found"] = valid_initials

        # Generate analysis
        analysis_parts = []
        analysis_parts.append(f"Total selection marks: {len(selection_marks)}")
        analysis_parts.append(f"Selected marks: {len(selected_marks)}")
        analysis_parts.append(f"Unselected marks: {len(unselected_marks)}")
        analysis_parts.append(f"X marks detected: {x_marks_in_text}")
        analysis_parts.append(f"Valid initials detected: {valid_initials}")

        # Add issues
        if x_marks_in_text > 0:
            # Only flag X marks as invalid for Ohio documents
            if state == "OH":
                validation_results["issues"].append(
                    f"Found {x_marks_in_text} X marks (invalid for {state} documents)"
                )
            else:
                # For New England states, X marks may be acceptable
                validation_results["issues"].append(
                    f"Found {x_marks_in_text} X marks in initial boxes"
                )

        if len(unselected_marks) > 0:
            validation_results["issues"].append(
                f"Found {len(unselected_marks)} empty/unselected boxes"
            )

        if valid_initials == 0 and len(selection_marks) > 0:
            validation_results["issues"].append("No valid letter initials detected")

        validation_results["analysis"] = "; ".join(analysis_parts)

        return validation_results
