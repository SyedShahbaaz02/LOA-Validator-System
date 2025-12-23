"""
GPT-4o OCR Integration Module for Production LOA Validation
This module provides GPT-4o vision capabilities as a fallback OCR solution when
Azure Document Intelligence fails to detect selection marks (checkboxes) in LOA documents.

Production version - uses Openai4oService instead of hardcoded credentials.
"""

import base64
import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, Optional

import fitz  # PyMuPDF

from intelligentflow.business_logic.openai_4o_service import Openai4oService

from ..utils.retry import aggressive_retry


class GPT4oOCRIntegration:
    """Production GPT-4o OCR integration using dependency injection.

    Args:
        openai_4o_service: The OpenAI service for GPT-4o processing
    """

    def __init__(self, openai_4o_service: Openai4oService):
        if openai_4o_service is None:
            raise ValueError("openai_4o_service is required and cannot be None")

        self.openai_4o_service = openai_4o_service
        self.logger = logging.getLogger(__name__)

    def extract_pdf_image(self, pdf_path: str, page_num: int = 0) -> Optional[bytes]:
        """Extract an image from a PDF page for OCR processing.

        Args:
            pdf_path: Path to the PDF file
            page_num: Page number to extract (0-based index)

        Returns:
            bytes: Image data as bytes, or None if extraction fails
        """
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)

            # Check if the page exists
            if page_num >= len(doc):
                self.logger.error(
                    f"PDF only has {len(doc)} pages, requested page {page_num+1}"
                )
                return None

            # Get the page
            page = doc[page_num]

            # Render the page to an image (higher resolution for better OCR)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

            # Convert to PNG bytes
            image_bytes = pix.tobytes("png")

            doc.close()
            return image_bytes

        except Exception as e:
            self.logger.error(f"Error extracting image from PDF: {str(e)}")
            return None

    def encode_image_to_base64(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 for API transmission."""
        return base64.b64encode(image_bytes).decode("utf-8")

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def process_pdf_with_gpt4o_fallback(
        self, pdf_path: str, extraction_log: Optional[Dict] = None
    ) -> Dict:
        """Process a PDF using GPT-4o's vision capabilities to detect checkboxes when Azure Layout fails.

        CRITICAL: This function uses aggressive retry logic to guarantee success.
        It will retry up to 50 times with exponential backoff rather than failing.

        Args:
            pdf_path: Path to the PDF file
            extraction_log: The current extraction log from Azure Layout

        Returns:
            dict: Results including success status and updated extraction log
        """
        if extraction_log is None:
            extraction_log = {
                "pdf_path": pdf_path,
                "extraction_timestamp": datetime.now().isoformat(),
                "extraction_method": "GPT-4o Vision OCR",
                "selection_marks": [],
                "initial_boxes": [],
            }

        # Extract the first page as an image
        self.logger.info(f"Extracting image from: {os.path.basename(pdf_path)}")
        image_data = self.extract_pdf_image(pdf_path, page_num=0)

        if not image_data:
            return {
                "success": False,
                "error": "Failed to extract image from PDF",
                "extraction_log": extraction_log,
            }

        # Encode the image for API transmission
        base64_image = self.encode_image_to_base64(image_data)

        # Create a detailed checkbox extraction prompt for GPT-4o
        checkbox_extraction_prompt = """Analyze this document image and extract ALL checkbox information:

        1. Identify all checkboxes (square or round boxes that can be checked/selected)
        2. For each checkbox:
           - Determine if it's selected/checked or unselected/unchecked
           - Extract the text associated with the checkbox
           - Assess confidence level (0-100)
        3. Also identify any initial boxes (places for customer initials) and their content

        CRITICAL RULES FOR INITIAL BOXES:
        - Report EXACTLY what you see in each initial box
        - If you see a single "X" mark in an initial box with nothing else, report it as "X" with is_filled: true
        - If the box is empty or has just underscores/lines, report is_filled: false with empty text
        - Be accurate - distinguish between actual letter initials and X marks or checkmarks

        Return the information in this JSON format:
        {
          "checkboxes": [
            {
              "text": "Text associated with the checkbox",
              "is_selected": true/false,
              "confidence": 0-100,
              "location": "brief description of where on page"
            }
          ],
          "initial_boxes": [
            {
              "text": "exactly what is written (X if X mark, empty if blank)",
              "is_filled": true/false,
              "location": "brief description of where on page",
              "content_type": "x_mark|empty|unclear"
            }
          ]
        }

        Focus especially on checkboxes related to:
        - Account/SDI Number Release
        - Interval Historical Energy Usage Data Release
        - Historical Usage Data Release
        - Any other authorization or consent checkboxes"""

        # Create the user prompt for vision analysis
        user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
        {checkbox_extraction_prompt}

        [ATTACHED_IMAGE]
        data:image/png;base64,{base64_image}"""

        # System prompt for GPT-4o vision
        system_prompt = "You are an expert document analyzer specializing in form field detection. Your task is to accurately identify checkboxes and their states (selected/unselected) in legal documents, especially Letter of Authorization (LOA) forms."

        try:
            self.logger.info("Calling GPT-4o API for checkbox detection...")
            start_time = datetime.now()

            # Use the production OpenAI service
            gpt_response_data = self.openai_4o_service.process_with_prompts(
                system_prompt, user_prompt, max_token=4000, raw_response=True
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Extract the actual response content from the service response
            if gpt_response_data and gpt_response_data[0]["ai_result"]:
                analysis_text = gpt_response_data[0]["ai_result"][0]["result"]
            else:
                return {
                    "success": False,
                    "error": "No response from OpenAI service",
                    "extraction_log": extraction_log,
                }

            # Extract JSON from the response
            try:
                # First try to extract JSON from code blocks
                json_match = re.search(
                    r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL
                )
                if json_match:
                    analysis_json = json.loads(json_match.group(1))
                else:
                    # Otherwise try to parse the entire response as JSON
                    analysis_json = json.loads(analysis_text)

                # Process the checkboxes
                self.logger.info(
                    "Successfully extracted checkbox information with GPT-4o"
                )
                checkboxes_found = analysis_json.get("checkboxes", [])
                initial_boxes_found = analysis_json.get("initial_boxes", [])

                self.logger.info(
                    f"Found {len(checkboxes_found)} checkboxes and {len(initial_boxes_found)} initial boxes"
                )

                # Add selection marks from GPT-4o to the extraction log
                for i, checkbox in enumerate(checkboxes_found):
                    mark_info = {
                        "page": 1,  # Assume first page for now
                        "mark_index": len(extraction_log["selection_marks"]) + i,
                        "state": (
                            "selected"
                            if checkbox.get("is_selected", False)
                            else "unselected"
                        ),
                        "confidence": checkbox.get("confidence", 90)
                        / 100.0,  # Convert to 0-1 scale
                        "content": checkbox.get("text", ""),
                        "detection_source": "gpt4o_fallback",
                        "gpt4o_verified": True,
                    }
                    extraction_log["selection_marks"].append(mark_info)

                # Add initial boxes from GPT-4o to the extraction log
                for i, initial_box in enumerate(initial_boxes_found):
                    box_info = {
                        "type": (
                            "filled_box"
                            if initial_box.get("is_filled", False)
                            else "empty_box"
                        ),
                        "text": initial_box.get("text", ""),
                        "context": f"Location: {initial_box.get('location', 'Unknown')}",
                        "is_filled": initial_box.get("is_filled", False),
                        "is_likely_initial": True,
                        "detection_method": "gpt4o_fallback",
                    }
                    extraction_log["initial_boxes"].append(box_info)

                # Update GPT-4o details in the extraction log
                if "gpt4o_details" not in extraction_log:
                    extraction_log["gpt4o_details"] = {}

                extraction_log["gpt4o_details"]["ocr_fallback_used"] = True
                extraction_log["gpt4o_details"]["selection_marks_count"] = len(
                    checkboxes_found
                )
                extraction_log["gpt4o_details"]["selected_marks_count"] = len(
                    [c for c in checkboxes_found if c.get("is_selected", False)]
                )
                extraction_log["gpt4o_details"]["unselected_marks_count"] = len(
                    [c for c in checkboxes_found if not c.get("is_selected", False)]
                )
                extraction_log["gpt4o_details"]["initial_boxes_count"] = len(
                    initial_boxes_found
                )

                return {
                    "success": True,
                    "processing_time": processing_time,
                    "checkboxes_found": len(checkboxes_found),
                    "initial_boxes_found": len(initial_boxes_found),
                    "extraction_log": extraction_log,
                }

            except json.JSONDecodeError as e:
                error_msg = f"Error parsing GPT-4o JSON response: {str(e)}"
                self.logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "raw_response": analysis_text,
                    "extraction_log": extraction_log,
                }

        except Exception as e:
            error_msg = f"GPT-4o processing error: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "extraction_log": extraction_log,
            }
