"""
Enhanced LOA Validation System - Multi-Region Universal Utility Recognition
Supports Multiple Regions:
- Great Lakes Region: OH, MI, IL
- New England Region: ME, MA, NH, RI, CT
Uses Azure Document Intelligence Layout Model for Form Field Detection
Accepts any utility name mentioned in the document with improved handwritten initial recognition
Includes separate validation category for Interval Data Granularity
Illinois-Specific Features:
- ComEd utility automatically sets state to IL
- Third-party brokers allowed in IL (unlike OH)
- Ohio-specific requirements not applied to IL LOAs
- IL-specific interval data validation for EUI access statements
New England Features:
- Service option detection (One Time Request vs Annual Subscription)
- Region-specific utility patterns and time limits
- Summary usage only validation for specific utilities
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, List

from dateutil.relativedelta import relativedelta

from intelligentflow.business_logic.loa.document_integrity_checker import (
    DocumentIntegrityChecker,
)
from intelligentflow.business_logic.loa.enhanced_initial_detector import (
    EnhancedInitialDetector,
)
from intelligentflow.business_logic.loa.enhanced_selection_validation import (
    EnhancedSelectionValidator,
)
from intelligentflow.business_logic.loa.gpt4o_ocr_integration import GPT4oOCRIntegration
from intelligentflow.business_logic.loa.gpt4o_verification_integration import (
    GPT4oVerificationIntegration,
)
from intelligentflow.business_logic.openai_4o_service import Openai4oService
from intelligentflow.utils.field_extraction_utils import (
    extract_account_numbers,
    extract_field,
)


class EnhancedLOAValidator:
    """Enhanced LOA validator with multi-region support, advanced layout analysis, improved initial recognition, and universal utility name validation.

    Args:
        openai_4o_service: The OpenAI service for GPT-4o processing
        region: The region to validate LOAs for (e.g., "Great Lakes", "New England").
               Default is "Great Lakes".
        udc: The utility distribution company (UDC) code for precise state detection.
            Optional parameter that takes precedence over OCR-based utility detection.
        x_mark_confidence_threshold: Minimum confidence threshold (0.0-1.0) for flagging X marks.
            Only X marks detected with confidence >= this threshold will be flagged as validation issues.
            Default is 0.95 (95%). Higher values = more strict = fewer false positives.
        interval_needed: Whether interval data granularity should be required for validation.
            If False, the LOA will not be rejected for missing interval data specifications.
            Default is True to maintain backward compatibility.
        account_name: The account name from Salesforce to compare against LOA customer name.
            Optional parameter used for account name validation.
    """

    # Supported regions and their states
    SUPPORTED_REGIONS = {
        "Great Lakes": ["OH", "MI", "IL"],
        "New England": ["ME", "MA", "NH", "RI", "CT"],
    }

    # Error messages for validation
    ERROR_MESSAGES = {
        "customer_section_missing": "Customer section 'To be completed by Customer' not found in document",
        "customer_name_missing": "Customer Name (as it appears on the bill) field is missing",
        "customer_name_empty": "Customer Name (as it appears on the bill) field is empty or contains placeholder text",
        "broker_title": "NECO LOAs must be signed by the customer, not by a {role} (Title field: '{title_text}')",
        "broker_company_mismatch": "NECO LOAs must be signed by the customer. Company Name '{company_name}' appears to be a broker/third-party company, different from Customer Name '{customer_name}'",
        "supplier_section_missing": "Supplier/Third Party information section is missing",
        "supplier_name_missing": "Supplier/Third Party Name is missing or empty",
        "supplier_contact_missing": "Supplier/Third Party Contact is missing or empty",
        "supplier_email_missing": "Supplier/Third Party Email is missing",
        "supplier_signature_missing": "Supplier/Third Party signature is missing",
        "supplier_signature_date_missing": "Supplier/Third Party signature date is missing",
        "no_accounts_found": "No account numbers found in the document",
        "neco_subscription_none": "No NECO subscription option selected - exactly ONE option must be selected",
        "neco_subscription_multiple": "Multiple NECO subscription options selected - exactly ONE option must be selected",
        # COMED-specific error messages
        "comed_customer_name_missing": "ILLINOIS: Customer Name is missing",
        "comed_customer_address_missing": "ILLINOIS: Customer Address is missing",
        "comed_authorized_person_missing": "ILLINOIS: Authorized Person name is missing",
        "comed_authorized_person_title_missing": "ILLINOIS: Authorized Person Title is missing",
        "comed_signature_missing": "ILLINOIS: Customer signature is missing",
        "comed_signature_date_missing": "ILLINOIS: Signature date is missing",
        "comed_account_numbers_missing": "ILLINOIS: Account numbers are missing (or no indication of attached accounts)",
        "comed_interval_authorization_missing": "ILLINOIS: Missing required interval authorization (required mention of our permission to access interval data)",
        "comed_supplier_info_missing": "ILLINOIS: Supplier (Constellation) information is incomplete or missing",
        "comed_illinois_authorization_missing": "ILLINOIS: Missing required interval authorization (required mention of our permission to access interval data)",
        "comed_agent_checkbox_not_marked": "ILLINOIS: Agent authorization checkbox is not marked (Authorized Person must check the box indicating they are an agent for the Customer)",
        "comed_utility_not_mentioned": "ILLINOIS: Document specifies a different utility company than expected",
        # First Energy-specific error messages
        "firstenergy_customer_name_missing": "First Energy: Customer Name field is empty or not provided",
        "firstenergy_customer_address_missing": "First Energy: Customer Address field is empty or not provided",
        "firstenergy_customer_phone_missing": "First Energy: Phone Number field is empty or not provided",
        "firstenergy_authorized_person_title_missing": "First Energy: Authorized Person/Title field is empty or not provided",
        "firstenergy_account_numbers_missing": "First Energy: Account/SDI Number field is empty - no account numbers provided",
        "firstenergy_account_numbers_invalid_length": "First Energy: Account/SDI numbers must be 20 digits long - found numbers with incorrect length",
        "firstenergy_cres_name_missing": "First Energy: CRES Provider Name is missing or empty",
        "firstenergy_cres_address_missing": "First Energy: CRES Provider Address is missing or empty",
        "firstenergy_cres_phone_missing": "First Energy: CRES Provider Phone Number is missing or empty",
        "firstenergy_cres_email_missing": "First Energy: CRES Provider Email is missing or empty",
        "firstenergy_ohio_signature_missing": "First Energy: Signature under Ohio authorization statement is missing or empty",
        "firstenergy_ohio_date_missing": "First Energy: Date under Ohio authorization statement is missing or empty",
        "firstenergy_wrong_form": "Wrong form - this is not an accepted First Energy LOA format",
        "firstenergy_wrong_utility_in_ohio_phrase": "Ohio authorization statement must reference CEI, OE, TE, or The Illuminating Company - found different utility",
        # CINERGY/DUKE ENERGY-specific error messages (Ohio)
        "cinergy_account_format_invalid": "CINERGY/DUKE ENERGY: Account number must be exactly 22 digits, start with '910', and have 'Z' as the 13th character",
        "cinergy_account_missing": "CINERGY/DUKE ENERGY: Account number is not in the expected format (must be 22 characters: 21 digits + letter 'Z' at position 13, starting with '910'. Example: 910117129533Z109008636) or is missing",
        "cinergy_signature_expired": "CINERGY/DUKE ENERGY: Signature date is expired (must be within 1 year for Ohio)",
        "cinergy_signature_date_missing": "CINERGY/DUKE ENERGY: Signature date is missing",
        # AEP-specific error messages (Ohio)
        "aep_customer_name_missing": "AEP: Customer Name field is empty or not provided",
        "aep_customer_address_missing": "AEP: Customer Address field is empty or not provided",
        "aep_customer_phone_missing": "AEP: Phone Number field is empty or not provided",
        "aep_authorized_person_title_missing": "AEP: Authorized Person/Title field is empty or not provided",
        "aep_account_numbers_missing": "AEP: Account/SDI Number field is empty - no account numbers provided",
        "aep_account_numbers_invalid_length": "AEP: Account/SDI numbers must be 17 digits (standalone) or 11/17 format (with slash) - found numbers with incorrect length or format",
        "aep_cres_name_missing": "AEP: CRES Provider Name is missing or empty",
        "aep_cres_address_missing": "AEP: CRES Provider Address is missing or empty",
        "aep_cres_phone_missing": "AEP: CRES Provider Phone Number is missing or empty",
        "aep_cres_email_missing": "AEP: CRES Provider Email is missing or empty",
        "aep_ohio_signature_missing": "AEP: Signature under Ohio authorization statement is missing or empty",
        "aep_ohio_date_missing": "AEP: Date under Ohio authorization statement is missing or empty",
        "aep_wrong_form": "Wrong form - this is not an accepted AEP LOA format",
        "aep_wrong_utility_in_ohio_phrase": "Ohio authorization statement must reference AEP, CSPC, OPC, Columbus Southern Power, or Ohio Power Company - found different utility",
    }

    def __init__(
        self,
        openai_4o_service: Openai4oService,
        region: str = "Great Lakes",
        udc: str = None,
        x_mark_confidence_threshold: float = 0.95,
        interval_needed: bool = True,
        account_name: str = None,
        service_location_ldc: str = None,
    ):
        if openai_4o_service is None:
            raise ValueError("openai_4o_service is required and cannot be None")

        self.openai_4o_service = openai_4o_service
        self.logger = logging.getLogger(__name__)

        # Set the region (default to Great Lakes if not specified or invalid)
        self.region = region if region in self.SUPPORTED_REGIONS else "Great Lakes"

        # Store the provided UDC for use in validation
        self.provided_udc = udc

        # Store the account name for comparison
        self.account_name = account_name

        # Store the service location LDC (account numbers) for comparison
        self.service_location_ldc = service_location_ldc

        # Store whether interval data granularity is required
        self.interval_needed = bool(interval_needed)

        # Store X mark confidence threshold (convert to 0-1 scale if needed)
        if x_mark_confidence_threshold > 1.0:
            self.x_mark_confidence_threshold = (
                x_mark_confidence_threshold / 100.0
            )  # Convert from 0-100 to 0-1 scale
        else:
            self.x_mark_confidence_threshold = x_mark_confidence_threshold

        # Ensure threshold is within valid range
        self.x_mark_confidence_threshold = max(
            0.0, min(1.0, self.x_mark_confidence_threshold)
        )

        # Set region-specific prompt file names
        if self.region == "New England":
            # For New England region, check if UDC is BECO specifically
            if self.provided_udc and "BECO" in self.provided_udc.upper():
                # Use BECO-specific prompt for BECO documents
                self.system_prompt_file = "system_prompt_beco.md"
                self.user_prompt_file = "user_prompt_new_england.md"
            else:
                # Use standard New England prompts for other utilities
                self.system_prompt_file = "system_prompt_new_england.md"
                self.user_prompt_file = "user_prompt_new_england.md"
        else:  # Great Lakes region (default)
            self.system_prompt_file = "loa_validation_system_prompt_great_lakes.md"
            self.user_prompt_file = "loa_validation_user_prompt_great_lakes.md"

        # Initialize enhanced components with region awareness
        self.enhanced_selection_validator = EnhancedSelectionValidator(
            region=self.region
        )
        self.enhanced_initial_detector = EnhancedInitialDetector()

        # Initialize GPT-4o processors with direct class instantiation
        self.gpt4o_ocr_integration = GPT4oOCRIntegration(openai_4o_service)
        self.gpt4o_verification_integration = GPT4oVerificationIntegration(
            openai_4o_service
        )

    def _load_system_prompt(self, detected_state: str) -> str:
        """Load system prompt from markdown file and format with current values."""

        try:
            # Get the directory of the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Use region-specific prompt file
            prompt_file_path = os.path.join(
                current_dir, "prompts", self.system_prompt_file
            )

            # Read the markdown file
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                prompt_template = f.read()

            # Format the template with current values
            current_date = datetime.now().strftime("%m/%d/%Y")
            current_year = datetime.now().year

            # Replace placeholders with current values
            formatted_prompt = prompt_template.replace(
                "{datetime.now().strftime('%m/%d/%Y')}", current_date
            )
            formatted_prompt = formatted_prompt.replace(
                "{datetime.now().year}", str(current_year)
            )
            formatted_prompt = formatted_prompt.replace(
                "{detected_state}", detected_state
            )

            return formatted_prompt

        except FileNotFoundError:
            # Try fallback to legacy prompt file (Great Lakes fallback)
            try:
                fallback_path = os.path.join(
                    current_dir,
                    "prompts",
                    "loa_validation_system_prompt_great_lakes.md",
                )
                with open(fallback_path, "r", encoding="utf-8") as f:
                    prompt_template = f.read()

                formatted_prompt = prompt_template.format(
                    detected_state=detected_state,
                    current_date=datetime.now().strftime("%m/%d/%Y"),
                    current_year=datetime.now().year,
                )

                return formatted_prompt
            except Exception:
                # Final fallback to basic prompt
                return (
                    f"You are an expert LOA validator for Constellation Energy's {self.region} Region.\n\n"
                    f"CRITICAL STATE DETECTION:\n"
                    f"- DETECTED STATE: {detected_state}\n\n"
                    f"Your task is to validate LOAs against ALL regulatory requirements.\n\n"
                    f"ERROR: Could not load system prompt from region-specific file. Using fallback prompt.\n"
                    f"Please ensure the prompt file exists at: intelligentflow/business_logic/loa/prompts/{self.system_prompt_file}"
                )

        except Exception as e:
            # Fallback for any other errors
            return (
                f"You are an expert LOA validator for Constellation Energy's {self.region} Region.\n\n"
                f"CRITICAL STATE DETECTION:\n"
                f"- DETECTED STATE: {detected_state}\n\n"
                f"ERROR: Failed to load system prompt: {str(e)}\n"
                f"Using fallback prompt."
            )

    def _load_user_prompt(self, **kwargs) -> str:
        """Load user prompt from markdown file and format with provided values."""

        try:
            # Get the directory of the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Use region-specific prompt file
            prompt_file_path = os.path.join(
                current_dir, "prompts", self.user_prompt_file
            )

            # Read the markdown file
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                prompt_template = f.read()

            # Replace the dynamic values in the template
            formatted_prompt = prompt_template
            for key, value in kwargs.items():
                placeholder = "{" + key + "}"
                formatted_prompt = formatted_prompt.replace(placeholder, str(value))

            return formatted_prompt

        except FileNotFoundError:
            # Try fallback to legacy prompt file (Great Lakes fallback)
            try:
                fallback_path = os.path.join(
                    current_dir, "prompts", "loa_validation_user_prompt.md"
                )
                with open(fallback_path, "r", encoding="utf-8") as f:
                    prompt_template = f.read()

                formatted_prompt = prompt_template.format(**kwargs)
                return formatted_prompt
            except Exception:
                # Final fallback to basic prompt
                return (
                    f"Analyze this LOA document using both text content and advanced layout analysis results.\n\n"
                    f"DOCUMENT ID: {kwargs.get('document_id', 'UNKNOWN')}\n\n"
                    f"ERROR: Could not load user prompt from region-specific file. Using fallback prompt.\n"
                    f"Please ensure the prompt file exists at: intelligentflow/business_logic/loa/prompts/{self.user_prompt_file}"
                )

        except Exception as e:
            # Fallback for any other errors
            return (
                f"Analyze this LOA document using both text content and advanced layout analysis results.\n\n"
                f"DOCUMENT ID: {kwargs.get('document_id', 'UNKNOWN')}\n\n"
                f"ERROR: Failed to load user prompt: {str(e)}\n"
                f"Using fallback prompt."
            )

    def extract_layout_from_ocr_result(
        self, ocr_result, document_id: str = None
    ) -> Dict:
        """Extract advanced layout analysis from OCR result object (from queue handler).

        This method processes the Document Intelligence result object that comes from the queue handler
        and extracts all the advanced layout information needed for validation.
        """

        extraction_log = {
            "document_id": document_id,
            "extraction_timestamp": datetime.now().isoformat(),
            "extraction_success": False,
            "extraction_method": "Azure Document Intelligence - Layout Model (from OCR result)",
            "error_details": None,
            "extracted_text": "",
            "text_length": 0,
            "page_count": 0,
            "confidence_scores": [],
            "processing_time_seconds": 0,
            "form_fields": [],
            "selection_marks": [],
            "key_value_pairs": [],
            "tables": [],
            "paragraphs": [],
            "potential_initials": [],  # Field to store detected potential initials
            "detected_utilities": [],  # Field to store detected utility names
            "initial_boxes": [],  # New field to store initial box detections
            "figures": [],  # New field to store figure data from OCR
            "service_options": {  # New field for New England service options
                "detected": False,
                "one_time_selected": False,
                "annual_subscription_selected": False,
                "selection_count": 0,
            },
        }

        start_time = datetime.now()

        try:
            # Extract text content from OCR result
            extracted_text = ""
            if ocr_result and "content" in ocr_result:
                extracted_text = ocr_result["content"]

            # CRITICAL: Add special handling for BHE documents
            is_bhe = self.provided_udc and (
                "BHE" in self.provided_udc.upper() or self.provided_udc.upper() == "BHE"
            )
            if is_bhe and self.region == "New England":
                # Add an explicit note to the extracted text that will be seen by the GPT model
                bhe_note = "\n\n[SYSTEM NOTE: THIS IS A BHE (BANGOR HYDRO ELECTRIC) DOCUMENT. SERVICE OPTION SELECTION IS NOT REQUIRED FOR BHE DOCUMENTS. ANY SERVICE OPTION OR LACK OF SERVICE OPTION SELECTION SHOULD BE IGNORED FOR VALIDATION PURPOSES.]\n\n"
                extracted_text = bhe_note + extracted_text

            extraction_log["extracted_text"] = extracted_text
            extraction_log["text_length"] = len(extracted_text)
            extraction_log["page_count"] = (
                len(ocr_result["pages"])
                if ocr_result and "pages" in ocr_result and ocr_result["pages"]
                else 0
            )
            extraction_log["extraction_success"] = len(extracted_text) > 0

            # Extract advanced layout elements from OCR result
            if ocr_result and "pages" in ocr_result and ocr_result["pages"]:
                for page_idx, page in enumerate(ocr_result["pages"]):
                    # Extract selection marks (checkboxes, radio buttons)
                    if "selection_marks" in page and page["selection_marks"]:
                        for mark_idx, mark in enumerate(page["selection_marks"]):
                            mark_info = {
                                "page": page_idx + 1,
                                "mark_index": mark_idx,
                                "state": (
                                    mark["state"] if "state" in mark else "unknown"
                                ),
                                "confidence": (
                                    mark["confidence"] if "confidence" in mark else None
                                ),
                                "bounding_box": (
                                    [point for point in mark["polygon"]]
                                    if "polygon" in mark
                                    else None
                                ),
                                "content": (
                                    mark["content"] if "content" in mark else None
                                ),
                            }
                            extraction_log["selection_marks"].append(mark_info)

                    # Extract lines and confidence scores
                    if "lines" in page and page["lines"]:
                        for line in page["lines"]:
                            if "confidence" in line and line["confidence"]:
                                extraction_log["confidence_scores"].append(
                                    line["confidence"]
                                )

            # Extract key-value pairs (form fields) from OCR result
            if (
                ocr_result
                and "key_value_pairs" in ocr_result
                and ocr_result["key_value_pairs"]
            ):
                for kv_idx, kv_pair in enumerate(ocr_result["key_value_pairs"]):
                    kv_info = {
                        "pair_index": kv_idx,
                        "key": (
                            kv_pair["key"]["content"]
                            if "key" in kv_pair and "content" in kv_pair["key"]
                            else None
                        ),
                        "value": (
                            kv_pair["value"]["content"]
                            if "value" in kv_pair and "content" in kv_pair["value"]
                            else None
                        ),
                        "confidence": (
                            kv_pair["confidence"] if "confidence" in kv_pair else None
                        ),
                    }
                    extraction_log["key_value_pairs"].append(kv_info)

            # Extract paragraphs for better text structure understanding
            if ocr_result and "paragraphs" in ocr_result and ocr_result["paragraphs"]:
                for para_idx, paragraph in enumerate(ocr_result["paragraphs"]):
                    para_info = {
                        "paragraph_index": para_idx,
                        "content": (
                            paragraph["content"] if "content" in paragraph else None
                        ),
                        "role": paragraph["role"] if "role" in paragraph else None,
                        "bounding_box": (
                            [
                                point
                                for point in paragraph["bounding_regions"][0]["polygon"]
                            ]
                            if "bounding_regions" in paragraph
                            and paragraph["bounding_regions"]
                            else None
                        ),
                    }
                    extraction_log["paragraphs"].append(para_info)

            # Extract tables if present
            if ocr_result and "tables" in ocr_result and ocr_result["tables"]:
                for table_idx, table in enumerate(ocr_result["tables"]):
                    table_info = {
                        "table_index": table_idx,
                        "row_count": table["row_count"] if "row_count" in table else 0,
                        "column_count": (
                            table["column_count"] if "column_count" in table else 0
                        ),
                        "cells": [],
                    }

                    if "cells" in table and table["cells"]:
                        for cell in table["cells"]:
                            cell_info = {
                                "content": (
                                    cell["content"] if "content" in cell else None
                                ),
                                "row_index": (
                                    cell["row_index"] if "row_index" in cell else None
                                ),
                                "column_index": (
                                    cell["column_index"]
                                    if "column_index" in cell
                                    else None
                                ),
                                "kind": cell["kind"] if "kind" in cell else None,
                            }
                            table_info["cells"].append(cell_info)

                    extraction_log["tables"].append(table_info)

            # Try to extract figures from the raw JSON response if available
            if hasattr(ocr_result, "_response") and hasattr(
                ocr_result._response, "json"
            ):
                try:
                    raw_json = ocr_result._response.json()
                    if "figures" in raw_json:
                        for fig_idx, figure in enumerate(raw_json["figures"]):
                            figure_info = {
                                "figure_id": figure.get("id", f"figure_{fig_idx}"),
                                "bounding_regions": [],
                            }

                            if "boundingRegions" in figure:
                                for region in figure["boundingRegions"]:
                                    region_info = {
                                        "page_number": region.get("pageNumber", 1),
                                        "polygon": region.get("polygon", None),
                                    }
                                    figure_info["bounding_regions"].append(region_info)

                            extraction_log["figures"].append(figure_info)
                except Exception:
                    # Log error but continue processing
                    pass

            # Detect initial boxes and their content (enhanced version)
            self.detect_initial_boxes(extracted_text, extraction_log)

            # Detect potential handwritten initials in the text
            self.detect_potential_initials(extracted_text, extraction_log)

            # Detect New England specific service options if applicable
            if self.region == "New England":
                self.detect_service_options(extracted_text, extraction_log)

            # Detect MECO/NANT-specific subscription options (3 options)
            if self.provided_udc and (
                "MECO" in self.provided_udc.upper()
                or "NANT" in self.provided_udc.upper()
            ):
                self.detect_meco_subscription_options(extracted_text, extraction_log)

            # Detect NECO-specific subscription options (2 options)
            if self.provided_udc and "NECO" in self.provided_udc.upper():
                self.detect_neco_subscription_options(extracted_text, extraction_log)

            # Detect NHEC-specific request type options (2 options)
            if self.provided_udc and "NHEC" in self.provided_udc.upper():
                self.detect_nhec_request_type_options(extracted_text, extraction_log)

            # Detect CMP/FGE-specific billing options (2 options)
            if self.provided_udc and (
                "CMP" in self.provided_udc.upper() or "FGE" in self.provided_udc.upper()
            ):
                self.detect_cmp_billing_options(extracted_text, extraction_log)

            # UDC detection removed - now using provided UDC parameter only

        except Exception as e:
            extraction_log["error_details"] = str(e)
            extraction_log["extracted_text"] = f"OCR_ERROR: {str(e)}"

        # Calculate processing time
        end_time = datetime.now()
        extraction_log["processing_time_seconds"] = (
            end_time - start_time
        ).total_seconds()

        return extraction_log

    def detect_initial_boxes(self, text: str, extraction_log: Dict) -> None:
        """Detect initial boxes and their content using enhanced detection methods."""

        # Use the enhanced initial detector for improved accuracy
        if hasattr(self, "enhanced_initial_detector"):
            # Get enhanced detection results
            enhanced_results = (
                self.enhanced_initial_detector.detect_handwritten_initials(text)
            )

            # Store the enhanced results in the extraction log
            extraction_log["initial_boxes"] = enhanced_results.get("initial_boxes", [])
            extraction_log["potential_initials"] = enhanced_results.get(
                "potential_initials", []
            )

            # Add drawing path analysis results if available
            if enhanced_results.get("drawing_paths"):
                extraction_log["drawing_path_analysis"] = enhanced_results[
                    "drawing_paths"
                ]
        else:
            # Fallback to basic detection if enhanced detector is not available
            self._basic_initial_box_detection(text, extraction_log)

    def _basic_initial_box_detection(self, text: str, extraction_log: Dict) -> None:
        """Basic initial box detection as fallback method."""

        # Pattern for detecting initial boxes and their content
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

        underscore_patterns = [
            r"Initial[^:]*:\s*_{1,}",
            r"_{3,}\s*(?:Initial|Initials)",
        ]

        # Initialize result structures
        extraction_log["initial_boxes"] = []
        extraction_log["potential_initials"] = []

        # Check for empty initial boxes
        for pattern in underscore_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                extraction_log["initial_boxes"].append(
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
            matches = re.finditer(pattern, text, re.MULTILINE)
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
                    end = min(len(text), match.end() + 100)
                    context = text[start:end].replace("\n", " ")

                    extraction_log["initial_boxes"].append(
                        {
                            "type": "filled_box",
                            "text": initial_text.strip(),
                            "context": context,
                            "is_filled": True,
                            "is_likely_initial": len(initial_text.strip()) <= 3,
                            "detection_method": f"pattern_{initial_box_patterns.index(pattern) + 1}",
                        }
                    )

        # Store potential initials for backward compatibility
        for box in extraction_log["initial_boxes"]:
            if box["is_filled"] and box.get("is_likely_initial", False):
                extraction_log["potential_initials"].append(
                    {
                        "text": box["text"],
                        "is_likely_initial": True,
                        "context": box["context"],
                    }
                )

    def detect_service_options(self, text: str, extraction_log: Dict) -> None:
        """Detect New England service options in the document.
        This is only applicable for New England region LOAs.
        """
        if self.region != "New England":
            return

        # CRITICAL: Special handling for utilities that don't require service options validation
        # List of utilities that bypass service options validation
        bypass_utilities = [
            "BHE",  # Bangor Hydro Electric
            "CMP",  # Central Maine Power
            "GSECO",  # Granite State Electric
            "LIBERTY",  # Liberty Utilities
            "FGE",  # Fitchburg Gas & Electric
            "MECO",  # Massachusetts Electric
            "NANT",  # Nantucket Electric
            "NECO",  # Narragansett Electric
            "NHEC",  # New Hampshire Electric Co-op
            "PSNH",  # Public Service of New Hampshire
            "UES",  # Unitil Energy Systems
            "UI",  # United Illuminating
        ]

        # Additional full names for matching
        full_names = [
            "BANGOR HYDRO ELECTRIC",
            "CENTRAL MAINE POWER",
            "GRANITE STATE ELECTRIC",
            "LIBERTY UTILITIES",
            "FITCHBURG GAS & ELECTRIC",
            "MASSACHUSETTS ELECTRIC",
            "NANTUCKET ELECTRIC",
            "NARRAGANSETT ELECTRIC",
            "NEW HAMPSHIRE ELECTRIC CO-OP",
            "PUBLIC SERVICE OF NEW HAMPSHIRE",
            "UNITIL ENERGY SYSTEMS",
            "UNITED ILLUMINATING",
        ]

        # Check if provided UDC is in bypass list
        bypass_service_options = False
        if self.provided_udc:
            udc_upper = self.provided_udc.upper()

            # Check abbreviated names
            if any(utility in udc_upper for utility in bypass_utilities):
                bypass_service_options = True

            # Check full names
            if any(full_name in udc_upper for full_name in full_names):
                bypass_service_options = True

        # Special case for Liberty in New Hampshire
        is_liberty_nh = (
            self.provided_udc
            and ("LIBERTY" in self.provided_udc.upper())
            and (
                "NH" in self.provided_udc.upper()
                or "NEW HAMPSHIRE" in self.provided_udc.upper()
            )
        )
        if is_liberty_nh:
            bypass_service_options = True

        if bypass_service_options:
            # For these documents, automatically mark one service option as selected to bypass the validation
            extraction_log["service_options"] = {
                "detected": True,
                "one_time_selected": True,  # Default to One Time Request selected
                "annual_subscription_selected": False,
                "selection_count": 1,
                "bypass_reason": f"Service options validation bypassed for utility: {self.provided_udc}",
            }
            return

        # For non-BHE documents, continue with normal service option detection
        # Look for service options text
        one_time_pattern = r"One\s+Time\s+Request,\s+\$50\.00\s+per\s+account\s+number"
        annual_pattern = (
            r"Annual\s+Subscription,\s+\$300\.00\s+per\s+account\s+per\s+year"
        )

        one_time_match = re.search(one_time_pattern, text, re.IGNORECASE)
        annual_match = re.search(annual_pattern, text, re.IGNORECASE)

        # If either option is found, service options are detected
        extraction_log["service_options"]["detected"] = bool(
            one_time_match or annual_match
        )

        if not extraction_log["service_options"]["detected"]:
            return

        # Now determine which options are selected by analyzing selection marks
        selection_marks = extraction_log.get("selection_marks", [])
        selected_marks = [
            mark for mark in selection_marks if mark.get("state") == "selected"
        ]

        # Get paragraphs containing service options
        one_time_paragraph = None
        annual_paragraph = None

        if one_time_match:
            # Get context around one time option
            start = max(0, one_time_match.start() - 100)
            end = min(len(text), one_time_match.end() + 200)
            one_time_paragraph = text[start:end]

        if annual_match:
            # Get context around annual option
            start = max(0, annual_match.start() - 100)
            end = min(len(text), annual_match.end() + 200)
            annual_paragraph = text[start:end]

        # Check for selection marks near service options text
        # First look for explicit checkbox indicators like "☑" or "☒" or "[X]"
        one_time_checkbox_pattern = r"[☑☒✓✗X]\s*One\s+Time\s+Request"
        annual_checkbox_pattern = r"[☑☒✓✗X]\s*Annual\s+Subscription"

        one_time_selected = bool(
            re.search(one_time_checkbox_pattern, text, re.IGNORECASE)
        )
        annual_selected = bool(re.search(annual_checkbox_pattern, text, re.IGNORECASE))

        # If explicit checkbox indicators are found, use them
        if one_time_selected or annual_selected:
            extraction_log["service_options"]["one_time_selected"] = one_time_selected
            extraction_log["service_options"][
                "annual_subscription_selected"
            ] = annual_selected
            extraction_log["service_options"]["selection_count"] = sum(
                [one_time_selected, annual_selected]
            )
            return

        # If no explicit checkboxes are found in text, use selection marks from layout analysis
        if selected_marks and (one_time_paragraph or annual_paragraph):
            # Look for the actual text patterns showing which option is selected
            # Handle both ":selected: Option" and "Option :selected:" patterns
            one_time_selected_pattern_before = r":selected:\s*One\s+Time\s+Request"
            one_time_selected_pattern_after = r"One\s+Time\s+Request[^:]*:selected:"
            annual_selected_pattern_before = r":selected:\s*Annual\s+Subscription"
            annual_selected_pattern_after = r"Annual\s+Subscription[^:]*:selected:"
            one_time_unselected_pattern = r":unselected:\s*One\s+Time\s+Request"
            annual_unselected_pattern = r":unselected:\s*Annual\s+Subscription"

            # Check text patterns first - look for both before and after patterns
            one_time_selected_text = bool(
                re.search(one_time_selected_pattern_before, text, re.IGNORECASE)
            ) or bool(re.search(one_time_selected_pattern_after, text, re.IGNORECASE))
            annual_selected_text = bool(
                re.search(annual_selected_pattern_before, text, re.IGNORECASE)
            ) or bool(re.search(annual_selected_pattern_after, text, re.IGNORECASE))
            one_time_unselected_text = bool(
                re.search(one_time_unselected_pattern, text, re.IGNORECASE)
            )
            annual_unselected_text = bool(
                re.search(annual_unselected_pattern, text, re.IGNORECASE)
            )

            # If we find explicit :selected: or :unselected: patterns, use them
            if (
                one_time_selected_text
                or annual_selected_text
                or one_time_unselected_text
                or annual_unselected_text
            ):
                extraction_log["service_options"][
                    "one_time_selected"
                ] = one_time_selected_text
                extraction_log["service_options"][
                    "annual_subscription_selected"
                ] = annual_selected_text
                extraction_log["service_options"]["selection_count"] = sum(
                    [one_time_selected_text, annual_selected_text]
                )
            else:
                # Fallback to original heuristic logic
                early_selected_marks = [
                    mark for mark in selected_marks if mark.get("page", 1) == 1
                ]

                if len(early_selected_marks) == 1:
                    # If only one mark is selected, determine which option based on text order
                    if one_time_paragraph and one_time_match and annual_match:
                        if one_time_match.start() < annual_match.start():
                            extraction_log["service_options"][
                                "one_time_selected"
                            ] = True
                        else:
                            extraction_log["service_options"][
                                "annual_subscription_selected"
                            ] = True
                    else:
                        # Default to annual subscription if we can't determine order
                        extraction_log["service_options"][
                            "annual_subscription_selected"
                        ] = True

                    extraction_log["service_options"]["selection_count"] = 1

                elif len(early_selected_marks) == 2:
                    # If two marks are selected, both options are selected
                    extraction_log["service_options"]["one_time_selected"] = True
                    extraction_log["service_options"][
                        "annual_subscription_selected"
                    ] = True
                    extraction_log["service_options"]["selection_count"] = 2

                else:
                    # If no marks or more than two, count all early selected marks
                    extraction_log["service_options"]["selection_count"] = len(
                        early_selected_marks
                    )

        # If we still haven't determined the selections, search for X marks or check marks in text
        if extraction_log["service_options"]["selection_count"] == 0:
            # Look for patterns like [X] or (X) or X_ before service options
            x_one_time = bool(
                re.search(r"[\[\(]?\s*[Xx]\s*[\]\)]?\s+One\s+Time", text, re.IGNORECASE)
            )
            x_annual = bool(
                re.search(r"[\[\(]?\s*[Xx]\s*[\]\)]?\s+Annual", text, re.IGNORECASE)
            )

            extraction_log["service_options"]["one_time_selected"] = x_one_time
            extraction_log["service_options"]["annual_subscription_selected"] = x_annual
            extraction_log["service_options"]["selection_count"] = sum(
                [x_one_time, x_annual]
            )

    def detect_nhec_request_type_options(self, text: str, extraction_log: Dict) -> None:
        """Detect NHEC-specific request type options (Request Type - Select One).
        NHEC LOAs have 2 request type options and exactly ONE must be selected.
        This is only applicable for NHEC UDC in New England region.
        """
        # Look for NHEC request type option patterns (2 options)
        adhoc_pattern = r"Ad-hoc\s+Request\s+for\s+Historic(?:al)?\s+Data"
        subscription_pattern = r"Subscription\s+Request\s+for\s+Future\s+Data"

        adhoc_match = re.search(adhoc_pattern, text, re.IGNORECASE)
        subscription_match = re.search(subscription_pattern, text, re.IGNORECASE)

        # Initialize NHEC request type options structure
        extraction_log["nhec_request_type_options"] = {
            "detected": False,
            "adhoc_selected": False,
            "subscription_selected": False,
            "selection_count": 0,
        }

        # If any option is found, NHEC request type options are detected
        extraction_log["nhec_request_type_options"]["detected"] = bool(
            adhoc_match or subscription_match
        )

        if not extraction_log["nhec_request_type_options"]["detected"]:
            return

        # Now determine which options are selected
        # Look for X marks or checkboxes near each option
        adhoc_checkbox_pattern = r"[☑☒✓✗X]\s*Ad-hoc\s+Request"
        subscription_checkbox_pattern = r"[☑☒✓✗X]\s*Subscription\s+Request"

        adhoc_selected = bool(re.search(adhoc_checkbox_pattern, text, re.IGNORECASE))
        subscription_selected = bool(
            re.search(subscription_checkbox_pattern, text, re.IGNORECASE)
        )

        # Update the extraction log
        extraction_log["nhec_request_type_options"]["adhoc_selected"] = adhoc_selected
        extraction_log["nhec_request_type_options"][
            "subscription_selected"
        ] = subscription_selected
        extraction_log["nhec_request_type_options"]["selection_count"] = sum(
            [adhoc_selected, subscription_selected]
        )

    def detect_neco_subscription_options(self, text: str, extraction_log: Dict) -> None:
        """Detect NECO-specific subscription options (Type of Interval Data Request).
        NECO LOAs have 2 subscription options and exactly ONE must be selected.
        This is only applicable for NECO UDC in New England region.
        """
        # Look for NECO subscription option patterns (2 options only)
        two_weeks_pattern = r"Two\s+Weeks\s+Online\s+Access\s+to\s+data"
        one_year_pattern = r"One\s+Year\s+Online\s+Access\s+to\s+Data"

        two_weeks_match = re.search(two_weeks_pattern, text, re.IGNORECASE)
        one_year_match = re.search(one_year_pattern, text, re.IGNORECASE)

        # Initialize NECO subscription options structure
        extraction_log["neco_subscription_options"] = {
            "detected": False,
            "two_weeks_selected": False,
            "one_year_selected": False,
            "selection_count": 0,
        }

        # If any option is found, NECO subscription options are detected
        extraction_log["neco_subscription_options"]["detected"] = bool(
            two_weeks_match or one_year_match
        )

        if not extraction_log["neco_subscription_options"]["detected"]:
            return

        # Now determine which options are selected
        # Look for X marks or checkboxes near each option
        two_weeks_checkbox_pattern = r"[☑☒✓✗X]\s*Two\s+Weeks\s+Online"
        one_year_checkbox_pattern = r"[☑☒✓✗X]\s*One\s+Year\s+Online"

        two_weeks_selected = bool(
            re.search(two_weeks_checkbox_pattern, text, re.IGNORECASE)
        )
        one_year_selected = bool(
            re.search(one_year_checkbox_pattern, text, re.IGNORECASE)
        )

        # Update the extraction log
        extraction_log["neco_subscription_options"][
            "two_weeks_selected"
        ] = two_weeks_selected
        extraction_log["neco_subscription_options"][
            "one_year_selected"
        ] = one_year_selected
        extraction_log["neco_subscription_options"]["selection_count"] = sum(
            [two_weeks_selected, one_year_selected]
        )

    def validate_neco_customer_name_field(
        self, text: str, extraction_log: Dict
    ) -> List[str]:
        """Validate NECO-specific customer name field requirement.
        NECO documents must have 'Customer Name (as it appears on the bill):' field filled in.
        Also validates other required customer fields: Printed Name, Title, Company Name, Date.

        Args:
            text: Extracted text from the document
            extraction_log: The extraction log to store validation results

        Returns:
            List of validation issues found (empty if validation passes)
        """
        validation_issues = []

        # Initialize NECO customer name validation structure
        extraction_log["neco_customer_name_validation"] = {
            "field_found": False,
            "field_filled": False,
            "customer_name": None,
            "printed_name_found": False,
            "title_found": False,
            "company_name_found": False,
            "customer_date_found": False,
            "validation_method": "code_level_check",
        }

        # Extract the customer section from the document
        customer_section_pattern = r"To\s+be\s+completed\s+by\s+Customer.*?(?=To\s+be\s+completed\s+by\s+Supplier|Supplier/Third\s+Party|$)"
        customer_section_match = re.search(
            customer_section_pattern, text, re.IGNORECASE | re.DOTALL
        )

        if not customer_section_match:
            validation_issues.append(
                "Customer section 'To be completed by Customer' not found in document"
            )
            return validation_issues

        customer_section = customer_section_match.group(0)

        # 1. Check for "Customer Name (as it appears on the bill)" field - AT TOP OF FORM
        # Use [ \t]* to match only spaces/tabs, not newlines (prevents matching next line)
        customer_name_patterns = [
            r"Customer\s+Name\s+\(as\s+it\s+appears\s+on\s+the\s+bill\):[ \t]*([^\n]*)",
            r"Customer\s+Name\s+\(as\s+it\s+appears\s+on\s+bill\):[ \t]*([^\n]*)",
            r"Customer\s+Name\s+\(\s*as\s+on\s+bill\s*\):[ \t]*([^\n]*)",
        ]

        # Use utility function to extract customer name
        customer_placeholders = [
            "please",
            "fill",
            "enter",
            "name here",
            "as appears",
            "customer name",
        ]
        try:
            customer_name_text = extract_field(
                customer_name_patterns, text, placeholders=customer_placeholders
            )
        except Exception as e:
            customer_name_text = None
            extraction_log["neco_customer_name_validation"][
                "customer_name_extraction_error"
            ] = str(e)

        if customer_name_text:
            extraction_log["neco_customer_name_validation"]["field_found"] = True
            extraction_log["neco_customer_name_validation"]["field_filled"] = True
            extraction_log["neco_customer_name_validation"][
                "customer_name"
            ] = customer_name_text
            customer_name_found = True
        else:
            customer_name_found = False
            # Check if field exists but is empty/invalid
            field_exists = any(
                re.search(pattern, text, re.IGNORECASE)
                for pattern in customer_name_patterns
            )
            if field_exists:
                extraction_log["neco_customer_name_validation"]["field_found"] = True
                validation_issues.append(self.ERROR_MESSAGES["customer_name_empty"])
            else:
                validation_issues.append(self.ERROR_MESSAGES["customer_name_missing"])

        # 2. Check for *Printed Name field in customer section
        printed_name_patterns = [r"\*Printed\s+Name[:\s]*([^\n*]+)"]
        try:
            printed_name_text = extract_field(
                printed_name_patterns, customer_section, min_length=2
            )
        except Exception as e:
            printed_name_text = None
            extraction_log["neco_customer_name_validation"][
                "printed_name_extraction_error"
            ] = str(e)
        if printed_name_text:
            extraction_log["neco_customer_name_validation"]["printed_name_found"] = True

        # NOTE: Printed Name, Title, and Company Name are often handwritten
        # Code-level regex validation cannot reliably detect handwritten text from OCR
        # These fields are validated by GPT-4o in the system prompt instead
        # We only validate the "Customer Name (as appears on bill)" which is typically typed

        # CRITICAL: Enhanced broker detection in title field
        # Multiple patterns to handle different OCR variations
        title_patterns = [
            r"\*Title[:\s]*([^\n*]+)",  # *Title: Owner
            r"Title[:\s]*([^\n*]+)",  # Title: Owner (without asterisk)
            r"\*\s*Title[:\s]*([^\n*]+)",  # * Title: Owner (with space)
            r"&\s*Title[:\s]*([^\n*]+)",  # & Title: Broker (alternate format)
        ]

        # Broker-related keywords to check for (case-insensitive)
        # Note: "Owner", "President", "Manager" etc. are NOT broker indicators
        broker_keywords = ["broker", "agent", "consultant"]

        # Use utility function to extract title
        title_placeholders = ["title", "your title", "enter title", "please print"]
        try:
            title_text = extract_field(
                title_patterns,
                customer_section,
                min_length=1,
                placeholders=title_placeholders,
            )
        except Exception as e:
            title_text = None
            extraction_log["neco_customer_name_validation"][
                "title_extraction_error"
            ] = str(e)

        if title_text:
            extraction_log["neco_customer_name_validation"]["title_found"] = True
            extraction_log["neco_customer_name_validation"]["title_text"] = title_text

            # Check if title CONTAINS any broker-related keywords
            title_lower = title_text.lower()
            for broker_keyword in broker_keywords:
                if broker_keyword in title_lower:
                    validation_issues.append(
                        self.ERROR_MESSAGES["broker_title"].format(
                            role=broker_keyword.title(), title_text=title_text
                        )
                    )
                    extraction_log["neco_customer_name_validation"][
                        "broker_title_detected"
                    ] = True
                    extraction_log["neco_customer_name_validation"][
                        "broker_keyword_found"
                    ] = broker_keyword
                    break

        # 4. Check for *Company Name field in customer section
        company_name_patterns = [
            r"\*Company\s+Name[:\s]*([^\n*]+)",
            r"Company\s+Name[:\s]*([^\n*]+)",
            r"\*\s*Company\s+Name[:\s]*([^\n*]+)",
        ]

        # Use utility function to extract company name
        company_placeholders = ["company", "name", "enter", "please"]
        try:
            company_name_text = extract_field(
                company_name_patterns,
                customer_section,
                placeholders=company_placeholders,
            )
        except Exception as e:
            company_name_text = None
            extraction_log["neco_customer_name_validation"][
                "company_name_extraction_error"
            ] = str(e)

        if company_name_text:
            extraction_log["neco_customer_name_validation"]["company_name_found"] = True
            extraction_log["neco_customer_name_validation"][
                "company_name_text"
            ] = company_name_text

        # CRITICAL: Check if Company Name is an energy-related company different from Customer Name
        # This indicates a broker/third-party is signing instead of the customer
        if company_name_text and customer_name_found:
            customer_name_from_top = extraction_log["neco_customer_name_validation"][
                "customer_name"
            ]

            # Energy-related keywords that suggest a broker/third-party company
            energy_company_keywords = [
                "energy",
                "power",
                "electric",
                "utility",
                "utilities",
                "consulting",
                "consultant",
                "broker",
                "brokerage",
            ]

            # Check if company name contains energy-related keywords
            company_lower = company_name_text.lower()
            has_energy_keyword = any(
                keyword in company_lower for keyword in energy_company_keywords
            )

            # Check if company name is substantially different from customer name
            # Normalize for comparison (remove common suffixes, make lowercase)
            customer_normalized = re.sub(
                r"\s+(inc|llc|corp|corporation|company|co\.?|ltd|limited)\.?$",
                "",
                customer_name_from_top.lower(),
                flags=re.IGNORECASE,
            ).strip()
            company_normalized = re.sub(
                r"\s+(inc|llc|corp|corporation|company|co\.?|ltd|limited)\.?$",
                "",
                company_name_text.lower(),
                flags=re.IGNORECASE,
            ).strip()

            # They're different if normalized names don't match (allowing for minor variations)
            are_different = (
                customer_normalized not in company_normalized
                and company_normalized not in customer_normalized
            )

            # If company name has energy keywords AND is different from customer name, it's likely a broker
            if has_energy_keyword and are_different:
                validation_issues.append(
                    self.ERROR_MESSAGES["broker_company_mismatch"].format(
                        company_name=company_name_text,
                        customer_name=customer_name_from_top,
                    )
                )
                extraction_log["neco_customer_name_validation"][
                    "energy_company_mismatch_detected"
                ] = True
                extraction_log["neco_customer_name_validation"]["mismatch_details"] = {
                    "customer_name": customer_name_from_top,
                    "company_name": company_name_text,
                    "energy_keywords_found": [
                        kw for kw in energy_company_keywords if kw in company_lower
                    ],
                }

        # 5. Check for *Date field in customer section (separate from signature date extraction)
        # This is just to ensure the field exists and has a value
        customer_date_pattern = r"\*Date[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
        date_match = re.search(customer_date_pattern, customer_section)
        if date_match:
            extraction_log["neco_customer_name_validation"][
                "customer_date_found"
            ] = True

        # Note: We don't add validation issue for date here because GPT-4o handles date extraction separately

        return validation_issues

    def validate_neco_account_numbers(
        self, text: str, extraction_log: Dict
    ) -> List[str]:
        """Validate NECO account number requirements.
        NECO documents must have at least one account number provided.

        Args:
            text: Extracted text from the document
            extraction_log: The extraction log to store validation results

        Returns:
            List of validation issues found (empty if validation passes)
        """
        validation_issues = []

        # Initialize NECO account number validation structure
        extraction_log["neco_account_validation"] = {
            "accounts_found": False,
            "account_count": 0,
            "account_numbers": [],
            "validation_method": "code_level_check",
        }

        # NECO account number patterns - ENHANCED to detect numbers in tables and lists
        # NECO accounts are typically 8-20 digits
        account_patterns = [
            # Pattern 1: Account numbers with label
            r"Account\s*(?:Number|#|No\.?|Num)[:\s]*(\d{8,})",
            r"Acct\s*(?:Number|#|No\.?|Num)[:\s]*(\d{8,})",
            # Pattern 2: Generic standalone numbers (8-20 digits) - for table/list format
            r"\b(\d{8,20})\b",
            # Pattern 3: Numbers after colon or in structured format
            r"Account[:\s]+(\d{8,})",
            r":\s*(\d{8,20})\s*(?:\n|$|,)",  # Numbers after colon at end of line
            # Pattern 4: Numbers in table rows (detect standalone digit sequences)
            r"^\s*(\d{8,20})\s*$",  # Numbers on their own line (table format)
            r"^\s*(\d{8,20})\s+\w",  # Numbers followed by text (table row format)
        ]

        # Use utility function to extract account numbers
        account_numbers = extract_account_numbers(
            account_patterns, text, min_length=8, max_length=20
        )

        # Update extraction log
        if account_numbers:
            extraction_log["neco_account_validation"]["accounts_found"] = True
            extraction_log["neco_account_validation"]["account_count"] = len(
                account_numbers
            )
            extraction_log["neco_account_validation"][
                "account_numbers"
            ] = account_numbers
        else:
            # Check if document indicates accounts are attached
            attachment_indicators = [
                "see attached",
                "attached spreadsheet",
                "please attach",
            ]
            has_attachment_note = any(
                indicator in text.lower() for indicator in attachment_indicators
            )

            if not has_attachment_note:
                validation_issues.append(self.ERROR_MESSAGES["no_accounts_found"])

        return validation_issues

    def validate_neco_supplier_fields(
        self, text: str, extraction_log: Dict
    ) -> List[str]:
        """Validate NECO supplier/requestor information requirements.
        NECO documents must have complete supplier information including signature date.

        Args:
            text: Extracted text from the document
            extraction_log: The extraction log to store validation results

        Returns:
            List of validation issues found (empty if validation passes)
        """
        validation_issues = []

        # Initialize NECO supplier validation structure
        extraction_log["neco_supplier_validation"] = {
            "supplier_name_found": False,
            "supplier_contact_found": False,
            "supplier_email_found": False,
            "supplier_signature_date_found": False,
            "validation_method": "code_level_check",
        }

        # Check for supplier/third party section
        supplier_section_patterns = [
            r"To\s+be\s+completed\s+by\s+Supplier/Third\s+Party.*?(?=To\s+be\s+completed\s+by\s+Customer|\Z)",
            r"Supplier/Third\s+Party.*?(?=Customer|\Z)",
            r"Requestor\s+&\s+Billing.*?(?=Customer|\Z)",
        ]

        supplier_section = ""
        for pattern in supplier_section_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                supplier_section = match.group(0)
                break

        if not supplier_section:
            validation_issues.append(self.ERROR_MESSAGES["supplier_section_missing"])
            return validation_issues

        # Check supplier name/company using utility function
        supplier_name_patterns = [
            r"Supplier/Third\s+Party\s+Name[:\s]*([^\n]+)",
            r"Company\s+Name[:\s]*([^\n]+)",
            r"Supplier\s+Name[:\s]*([^\n]+)",
        ]

        supplier_placeholders = ["n/a", "none"]
        try:
            supplier_name = extract_field(
                supplier_name_patterns,
                supplier_section,
                placeholders=supplier_placeholders,
            )
        except Exception as e:
            supplier_name = None
            extraction_log["neco_supplier_validation"][
                "supplier_name_extraction_error"
            ] = str(e)

        if supplier_name:
            extraction_log["neco_supplier_validation"]["supplier_name_found"] = True
        else:
            validation_issues.append(self.ERROR_MESSAGES["supplier_name_missing"])

        # Check supplier contact using utility function
        contact_patterns = [
            r"Supplier/Third\s+Party\s+Contact[:\s]*([^\n]+)",
            r"Contact\s+Name[:\s]*([^\n]+)",
            r"Contact[:\s]*([^\n]+)",
        ]

        contact_placeholders = ["n/a", "none"]
        try:
            supplier_contact = extract_field(
                contact_patterns, supplier_section, placeholders=contact_placeholders
            )
        except Exception as e:
            supplier_contact = None
            extraction_log["neco_supplier_validation"][
                "supplier_contact_extraction_error"
            ] = str(e)

        if supplier_contact:
            extraction_log["neco_supplier_validation"]["supplier_contact_found"] = True
        else:
            validation_issues.append(self.ERROR_MESSAGES["supplier_contact_missing"])

        # Check supplier email
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        email_matches = re.findall(email_pattern, supplier_section)

        if email_matches:
            extraction_log["neco_supplier_validation"]["supplier_email_found"] = True
        else:
            validation_issues.append(self.ERROR_MESSAGES["supplier_email_missing"])

        # Check supplier signature using utility function
        signature_patterns = [
            r"Supplier/Third\s+Party\s+Signature[:\s]*([^\n]+)",
            r"Signature[:\s]*([^\n]+)",
        ]

        # Custom validation for signature - need to exclude "Date:" label
        signature_text = None
        for pattern in signature_patterns:
            sig_match = re.search(pattern, supplier_section, re.IGNORECASE)
            if sig_match:
                sig_text = sig_match.group(1).strip()
                # Check if signature field has content (not blank, not just underscores, not just "Date:")
                is_only_underscores = re.match(r"^_+$", sig_text)
                is_just_date_label = sig_text.lower().startswith("date")

                if (
                    len(sig_text) > 2
                    and not is_only_underscores
                    and not is_just_date_label
                ):
                    extraction_log["neco_supplier_validation"][
                        "supplier_signature_found"
                    ] = True
                    signature_text = sig_text
                    break

        if not signature_text:
            validation_issues.append(self.ERROR_MESSAGES["supplier_signature_missing"])

        # Check supplier signature date using utility function
        signature_date_patterns = [
            r"Date[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"Dated?\s+Signed[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",  # Generic date pattern in supplier section
        ]

        try:
            signature_date = extract_field(
                signature_date_patterns, supplier_section, min_length=8
            )
        except Exception as e:
            signature_date = None
            extraction_log["neco_supplier_validation"][
                "signature_date_extraction_error"
            ] = str(e)

        if signature_date:
            extraction_log["neco_supplier_validation"][
                "supplier_signature_date_found"
            ] = True
        else:
            validation_issues.append(
                self.ERROR_MESSAGES["supplier_signature_date_missing"]
            )

        return validation_issues

    def detect_cmp_billing_options(self, text: str, extraction_log: Dict) -> None:
        """Detect CMP-specific billing options (Check One - Billing).
        CMP LOAs have 2 billing options and exactly ONE must be selected.
        This is only applicable for CMP UDC in New England region.
        """
        # Look for CMP billing option patterns (2 options)
        invoice_customer_pattern = r"Invoice\s+the\s+customer"
        invoice_supplier_pattern = r"Invoice\s+the\s+supplier/broker\s+as\s+follows"

        invoice_customer_match = re.search(
            invoice_customer_pattern, text, re.IGNORECASE
        )
        invoice_supplier_match = re.search(
            invoice_supplier_pattern, text, re.IGNORECASE
        )

        # Initialize CMP billing options structure
        extraction_log["cmp_billing_options"] = {
            "detected": False,
            "invoice_customer_selected": False,
            "invoice_supplier_selected": False,
            "selection_count": 0,
        }

        # If any option is found, CMP billing options are detected
        extraction_log["cmp_billing_options"]["detected"] = bool(
            invoice_customer_match or invoice_supplier_match
        )

        if not extraction_log["cmp_billing_options"]["detected"]:
            return

        # Now determine which options are selected
        # Look for X marks or checkboxes near each option
        invoice_customer_checkbox_pattern = r"[☑☒✓✗X]\s*Invoice\s+the\s+customer"
        invoice_supplier_checkbox_pattern = r"[☑☒✓✗X]\s*Invoice\s+the\s+supplier/broker"

        invoice_customer_selected = bool(
            re.search(invoice_customer_checkbox_pattern, text, re.IGNORECASE)
        )
        invoice_supplier_selected = bool(
            re.search(invoice_supplier_checkbox_pattern, text, re.IGNORECASE)
        )

        # Update the extraction log
        extraction_log["cmp_billing_options"][
            "invoice_customer_selected"
        ] = invoice_customer_selected
        extraction_log["cmp_billing_options"][
            "invoice_supplier_selected"
        ] = invoice_supplier_selected
        extraction_log["cmp_billing_options"]["selection_count"] = sum(
            [invoice_customer_selected, invoice_supplier_selected]
        )

    def validate_comed_required_fields(
        self, text: str, extraction_log: Dict
    ) -> List[str]:
        """Validate COMED-specific required fields.
        COMED LOAs can have different formats/structures, so this validation is flexible
        and searches for required fields anywhere in the document.

        Required fields for COMED:
        1. Customer Name
        2. Customer Address
        3. Authorized Person
        4. Authorized Person Title
        5. Signature
        6. Signature Date
        7. Account Numbers (or indication of attached accounts)
        8. Interval Authorization
        9. Supplier (Constellation) information

        Args:
            text: Extracted text from the document
            extraction_log: The extraction log to store validation results

        Returns:
            List of validation issues found (empty if validation passes)
        """
        validation_issues = []
        # Determine utility type ONCE at the start (used by multiple validations)

        provided_udc_upper = self.provided_udc.upper() if self.provided_udc else ""

        is_ameren = any(
            x in provided_udc_upper for x in ["AMEREN", "CILCO", "CIPS", "IP"]
        )

        # Initialize COMED validation structure
        extraction_log["comed_validation"] = {
            "customer_name_found": False,
            "customer_address_found": False,
            "authorized_person_found": False,
            "authorized_person_title_found": False,
            "signature_found": False,
            "signature_date_found": False,
            "account_numbers_found": False,
            "interval_authorization_found": False,
            "supplier_info_found": False,
            "validation_method": "code_level_check",
        }

        # 1. Customer Name - flexible patterns to handle different formats
        # CRITICAL FIX: Exclude "Constellation" as it's our company (supplier), not the customer
        customer_name_patterns = [
            r"Customer\s+Name[:\s]*([^\n]{3,})",
            r"Name\s+of\s+Customer[:\s]*([^\n]{3,})",
            r"Account\s+Holder[:\s]*([^\n]{3,})",
            r"Company\s+Name[:\s]*([^\n]{3,})",
            r"Business\s+Name[:\s]*([^\n]{3,})",
            r"Legal\s+Name[:\s]*([^\n]{3,})",
        ]

        # Exclude Constellation (our company) and common placeholders
        customer_placeholders = [
            "customer",
            "name",
            "enter",
            "please",
            "fill",
            "constellation",
        ]

        # Also add explicit exclusion after extraction
        customer_name = None
        try:
            extracted_name = extract_field(
                customer_name_patterns,
                text,
                min_length=3,
                placeholders=customer_placeholders,
            )

            # CRITICAL: Validate that the extracted name is NOT "Constellation" or our company variations
            # Constellation is the supplier, never the customer
            if extracted_name:
                name_lower = extracted_name.lower()
                # Exclude if it's our company name
                if "constellation" not in name_lower:
                    customer_name = extracted_name
                else:
                    # This is our company name, not the customer - treat as not found
                    customer_name = None
                    extraction_log["comed_validation"][
                        "customer_name_rejected_reason"
                    ] = "Extracted name is Constellation (our company, not customer)"
        except Exception as e:
            customer_name = None
            extraction_log["comed_validation"]["customer_name_extraction_error"] = str(
                e
            )

        if customer_name:
            extraction_log["comed_validation"]["customer_name_found"] = True
            extraction_log["comed_validation"]["customer_name"] = customer_name
        else:
            validation_issues.append(self.ERROR_MESSAGES["comed_customer_name_missing"])

        # 2. Customer Address - flexible patterns
        address_patterns = [
            r"(?:Customer\s+)?Address[:\s]*([^\n]{10,})",
            r"(?:Service\s+)?Location[:\s]*([^\n]{10,})",
            r"Street\s+Address[:\s]*([^\n]{10,})",
            r"Mailing\s+Address[:\s]*([^\n]{10,})",
        ]

        address_placeholders = ["address", "street", "enter", "please"]
        try:
            customer_address = extract_field(
                address_patterns, text, min_length=10, placeholders=address_placeholders
            )
        except Exception as e:
            customer_address = None
            extraction_log["comed_validation"]["customer_address_extraction_error"] = (
                str(e)
            )

        if customer_address:
            extraction_log["comed_validation"]["customer_address_found"] = True
            extraction_log["comed_validation"]["customer_address"] = customer_address
        else:
            validation_issues.append(
                self.ERROR_MESSAGES["comed_customer_address_missing"]
            )

        # 3. Authorized Person - flexible patterns
        authorized_person_patterns = [
            r"Authorized\s+(?:Person|Representative|Signer)[:\s]*([^\n]{2,})",
            r"Authorized\s+(?:Person|Representative)\s+Name[:\s]*([^\n]{2,})",
            r"Name\s+of\s+Authorized\s+(?:Person|Representative)[:\s]*([^\n]{2,})",
            r"Signatory\s+Name[:\s]*([^\n]{2,})",
            r"Printed\s+Name[:\s]*([^\n]{2,})",  # Common in signature sections
            r"(?:Customer\s+)?Contact\s+Name[:\s]*([^\n]{2,})",  # Contact Name or Customer Contact Name
            r"Contact[:\s]*([^\n]{2,})",  # Generic Contact field
        ]

        auth_person_placeholders = ["name", "authorized", "print", "please"]
        try:
            authorized_person = extract_field(
                authorized_person_patterns,
                text,
                min_length=2,
                placeholders=auth_person_placeholders,
            )
        except Exception as e:
            authorized_person = None
            extraction_log["comed_validation"]["authorized_person_extraction_error"] = (
                str(e)
            )

        if authorized_person:
            extraction_log["comed_validation"]["authorized_person_found"] = True
            extraction_log["comed_validation"]["authorized_person"] = authorized_person
        else:
            validation_issues.append(
                self.ERROR_MESSAGES["comed_authorized_person_missing"]
            )

        # 4. Authorized Person Title - flexible patterns
        # CRITICAL: Only reject if the field label EXISTS but is EMPTY
        # If no "Title" field exists in the document at all, don't reject
        title_label_patterns = [
            r"(?:Authorized\s+Person\s+)?Title\s*:",
            r"Position\s*:",
            r"Job\s+Title\s*:",
            r"Role\s*:",
        ]

        # First check if any title field label exists
        title_field_exists = any(
            re.search(pattern, text, re.IGNORECASE) for pattern in title_label_patterns
        )

        # Then extract the value if field exists
        title_patterns = [
            r"(?:Authorized\s+Person\s+)?Title[:\s]*([^\n]{2,})",
            r"Position[:\s]*([^\n]{2,})",
            r"Job\s+Title[:\s]*([^\n]{2,})",
            r"Role[:\s]*([^\n]{2,})",
        ]

        title_placeholders = ["title", "position", "enter"]
        try:
            authorized_title = extract_field(
                title_patterns, text, min_length=2, placeholders=title_placeholders
            )
        except Exception as e:
            authorized_title = None
            extraction_log["comed_validation"]["authorized_title_extraction_error"] = (
                str(e)
            )

        # Store whether field label exists

        extraction_log["comed_validation"][
            "authorized_person_title_field_exists"
        ] = title_field_exists

        if authorized_title:

            extraction_log["comed_validation"]["authorized_person_title_found"] = True

            extraction_log["comed_validation"][
                "authorized_person_title"
            ] = authorized_title

        elif title_field_exists:

            # Field label exists but is empty

            extraction_log["comed_validation"]["authorized_person_title_found"] = False

            # Only REJECT for ComEd

            # Ameren business rules (line 117) only require "Authorized Person Name", not Title

            if not is_ameren:

                validation_issues.append(
                    self.ERROR_MESSAGES["comed_authorized_person_title_missing"]
                )

        else:

            # No title field label exists at all - this is OK for Ameren, don't reject

            extraction_log["comed_validation"]["authorized_person_title_found"] = False

            extraction_log["comed_validation"][
                "authorized_person_title_optional"
            ] = True

        # 5. Signature - look for signature indicators (handwritten signatures are hard to detect via OCR)
        signature_indicators = [
            "signature",
            "signed",
            "/s/",
            "digitally signed",
            "electronically signed",
            "executed by",
        ]

        # Also look for signature fields with content
        signature_field_patterns = [
            r"(?:Customer\s+)?Signature[:\s]*([^\n]{2,})",
            r"Signed\s+by[:\s]*([^\n]{2,})",
            r"/s/\s*([^\n]{2,})",
        ]

        # Check for signature indicators or filled signature fields
        has_signature_indicator = any(
            indicator in text.lower() for indicator in signature_indicators
        )
        try:
            signature_field = extract_field(
                signature_field_patterns, text, min_length=2
            )
        except Exception as e:
            signature_field = None
            extraction_log["comed_validation"]["signature_field_extraction_error"] = (
                str(e)
            )

        if has_signature_indicator or signature_field:
            extraction_log["comed_validation"]["signature_found"] = True
            if signature_field:
                extraction_log["comed_validation"]["signature_text"] = signature_field
        else:
            validation_issues.append(self.ERROR_MESSAGES["comed_signature_missing"])

        # 6. Signature Date - flexible patterns
        signature_date_patterns = [
            r"(?:Signature\s+)?Date[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"Dated[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"Date\s+Signed[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})",  # Generic date pattern
        ]

        try:
            signature_date = extract_field(signature_date_patterns, text, min_length=8)
        except Exception as e:
            signature_date = None
            extraction_log["comed_validation"]["signature_date_extraction_error"] = str(
                e
            )

        if signature_date:
            extraction_log["comed_validation"]["signature_date_found"] = True
            extraction_log["comed_validation"]["signature_date"] = signature_date
        else:
            validation_issues.append(
                self.ERROR_MESSAGES["comed_signature_date_missing"]
            )

        # 7. Account Numbers - COMED specific patterns
        account_patterns = [
            r"Account\s*(?:Number|#|No\.?|Num)[:\s]*(\d{8,})",
            r"Acct[:\s]*(\d{8,})",
            r"\b(\d{10,20})\b",  # COMED accounts are typically 10-20 digits
            r"Account[:\s]+(\d{8,})",
        ]

        account_numbers = extract_account_numbers(
            account_patterns, text, min_length=8, max_length=20
        )

        # Check for attachment indicators if no accounts found
        attachment_indicators = [
            "see attached",
            "attached",
            "attachment",
            "attached account",
            "list attached",
        ]
        has_attachment_note = any(
            indicator in text.lower() for indicator in attachment_indicators
        )

        if account_numbers or has_attachment_note:
            extraction_log["comed_validation"]["account_numbers_found"] = True
            if account_numbers:
                extraction_log["comed_validation"]["account_numbers"] = account_numbers
                extraction_log["comed_validation"]["account_count"] = len(
                    account_numbers
                )
            if has_attachment_note:
                extraction_log["comed_validation"]["has_attachment_indicator"] = True
        else:
            validation_issues.append(
                self.ERROR_MESSAGES["comed_account_numbers_missing"]
            )

        # 8. Interval Authorization - flexible patterns
        interval_patterns = [
            r"interval\s+data",
            r"interval\s+usage",
            r"interval\s+meter",
            r"15-minute\s+interval",
            r"hourly\s+interval",
            r"usage\s+data",
            r"meter\s+data",
            r"authorize.*?interval",
            r"release.*?interval",
            r"access.*?interval",
        ]

        has_interval_authorization = any(
            re.search(pattern, text, re.IGNORECASE) for pattern in interval_patterns
        )

        if has_interval_authorization:
            extraction_log["comed_validation"]["interval_authorization_found"] = True
        else:
            validation_issues.append(
                self.ERROR_MESSAGES["comed_interval_authorization_missing"]
            )

        # 9. Supplier (Constellation) Information - check for Constellation/CRES provider
        supplier_patterns = [
            r"constellation",
            r"cres\s+provider",
            r"retail\s+electric\s+supplier",
            r"supplier\s+name",
            r"energy\s+supplier",
        ]

        # Also look for Constellation email domains
        constellation_email_pattern = r"[a-zA-Z0-9._%+-]+@constellation(?:energy)?\.com"

        has_constellation_mention = any(
            re.search(pattern, text, re.IGNORECASE) for pattern in supplier_patterns
        )
        has_constellation_email = bool(
            re.search(constellation_email_pattern, text, re.IGNORECASE)
        )

        if has_constellation_mention or has_constellation_email:
            extraction_log["comed_validation"]["supplier_info_found"] = True
            if has_constellation_email:
                extraction_log["comed_validation"]["constellation_email_found"] = True
        else:
            validation_issues.append(self.ERROR_MESSAGES["comed_supplier_info_missing"])

        # 10. Interval Usage Authorization - REQUIRED for COMED LOAs
        # COMED LOAs must include permission to access interval data (Interval Usage Authorization clause)

        # FIRST: Check for "Usage Data Type" radio button section (rare format)
        # Some ComEd LOAs have explicit radio buttons: Summary vs Interval
        usage_data_type_section_pattern = r"Usage\s+Data\s+Type"
        has_usage_data_type_section = bool(
            re.search(usage_data_type_section_pattern, text, re.IGNORECASE)
        )

        if has_usage_data_type_section:
            # This LOA has the "Usage Data Type" section with radio buttons
            # Check if "Interval" radio button is selected

            # Look for selection indicators near "Interval" option
            interval_selected_patterns = [
                r"[☑☒✓✗X]\s*Interval",  # Checkbox/mark before Interval
                r"Interval.*?:selected:",  # OCR marker after Interval
                r":selected:.*?Interval",  # OCR marker before Interval
                r"\([Xx]\)\s*Interval",  # (X) Interval format
                r"Interval\s*\([Xx]\)",  # Interval (X) format
            ]

            interval_radio_selected = any(
                re.search(pattern, text, re.IGNORECASE)
                for pattern in interval_selected_patterns
            )

            # Store the detection results
            extraction_log["comed_validation"]["usage_data_type_section_found"] = True
            extraction_log["comed_validation"][
                "interval_radio_selected"
            ] = interval_radio_selected

            if interval_radio_selected:
                # Interval radio button is selected - VALID
                extraction_log["comed_validation"][
                    "interval_authorization_found"
                ] = True
                extraction_log["comed_validation"][
                    "illinois_authorization_found"
                ] = True
                extraction_log["comed_validation"][
                    "authorization_type"
                ] = "interval_radio_button"
                extraction_log["comed_validation"]["interval_data_in_auth"] = True
            else:
                # Interval radio button is NOT selected - INVALID
                extraction_log["comed_validation"][
                    "interval_authorization_found"
                ] = False
                extraction_log["comed_validation"][
                    "illinois_authorization_found"
                ] = False
                extraction_log["comed_validation"][
                    "authorization_type"
                ] = "summary_only"
                extraction_log["comed_validation"]["interval_data_in_auth"] = False
                validation_issues.append(
                    self.ERROR_MESSAGES["comed_illinois_authorization_missing"]
                )
        else:
            # No "Usage Data Type" section - use standard interval authorization text patterns
            # Check for Interval Usage Authorization keywords
            # CRITICAL: Patterns must match authorization keywords and data types in close proximity (same sentence/clause)
            # This prevents false positives where "interval" appears in title but "authorize" appears elsewhere

            # Use word boundary limits to ensure they're in the same sentence/clause (max ~200 chars apart)
            interval_auth_patterns = [
                # CRITICAL: ComEd-specific phrase that explicitly indicates interval data authorization
                # This phrase appears in most ComEd LOAs and clearly indicates 30-minute interval data
                r"EUI\s+includes\s+your\s+electricity\s+usage\s+levels\s+for\s+distinct\s+time\s+periods\s+as\s+short\s+as\s+(?:15|30|60)[-\s]minutes?",
                r"electricity\s+usage\s+levels\s+for\s+distinct\s+time\s+periods\s+as\s+short\s+as\s+(?:15|30|60)[-\s]minutes?",
                r"usage\s+levels\s+for\s+distinct\s+time\s+periods\s+as\s+short\s+as\s+(?:15|30|60)[-\s]minutes?",
                # Authorization + interval/usage data (within same sentence - limited reach)
                r"(?:authorize|permission|access|release).{0,200}?(?:interval|usage|meter)\s+data",
                r"(?:interval|usage|meter)\s+data.{0,200}?(?:authorize|permission|access|release)",
                # Specific interval data mentions with authorization
                r"(?:authorize|permission|access|release).{0,150}?interval.{0,50}?data",
                r"interval.{0,50}?data.{0,150}?(?:authorize|permission|access|release)",
                # Time granularity + authorization (must be close together)
                r"(?:15|30|60)[-\s]minute.{0,100}?(?:authorize|permission|access|release)",
                r"(?:authorize|permission|access|release).{0,100}?(?:15|30|60)[-\s]minute",
                r"hourly.{0,100}?(?:authorize|permission|access|release)",
                r"(?:authorize|permission|access|release).{0,100}?hourly",
                # EUI (Electricity Usage Information) authorization
                r"authorize.{0,50}?EUI",
                r"EUI.{0,50}?authorize",
                r"access.{0,50}?EUI",
                r"EUI.{0,50}?access",
                # Specific phrase patterns that are valid
                r"authorize.{0,50}?(?:the\s+)?release.{0,100}?interval",
                r"interval.{0,100}?(?:usage|data).{0,50}?(?:authorize|release)",
            ]

            # Check if any interval authorization pattern is found (removed re.DOTALL to be more strict)
            has_interval_authorization = any(
                re.search(pattern, text, re.IGNORECASE)
                for pattern in interval_auth_patterns
            )

            extraction_log["comed_validation"]["usage_data_type_section_found"] = False

            if has_interval_authorization:
                # Interval usage authorization found - VALID
                extraction_log["comed_validation"][
                    "interval_authorization_found"
                ] = True
                extraction_log["comed_validation"][
                    "illinois_authorization_found"
                ] = True
                extraction_log["comed_validation"][
                    "authorization_type"
                ] = "interval_usage_authorization"
                extraction_log["comed_validation"]["interval_data_in_auth"] = True
            else:
                # No interval usage authorization found - INVALID
                extraction_log["comed_validation"][
                    "interval_authorization_found"
                ] = False
                extraction_log["comed_validation"][
                    "illinois_authorization_found"
                ] = False
                extraction_log["comed_validation"]["authorization_type"] = "none"
                extraction_log["comed_validation"]["interval_data_in_auth"] = False
                validation_issues.append(
                    self.ERROR_MESSAGES["comed_illinois_authorization_missing"]
                )

        # 11. Illinois Utility Name Mention - REQUIRED for Illinois LOAs
        # Document must explicitly mention the utility company (ComEd or Ameren)
        # Determine which utility we're checking
        if is_ameren:
            # Ameren utilities - check for Ameren-specific names
            ameren_utility_patterns = [
                r"\bAmeren\b",  # Ameren (exact word)
                r"\bCILCO\b",  # CILCO
                r"\bCentral\s+Illinois\s+Light",  # Central Illinois Light Company
                r"\bCIPS\b",  # CIPS
                r"\bCentral\s+Illinois\s+Public\s+Service",  # Central Illinois Public Service
                r"\b IP\b",  # IP (with word boundaries to avoid false matches)
                r"\bIllinois\s+Power\b",  # Illinois Power
            ]
            utility_mentioned = any(
                re.search(pattern, text, re.IGNORECASE)
                for pattern in ameren_utility_patterns
            )
            if utility_mentioned:
                extraction_log["comed_validation"]["comed_utility_mentioned"] = True
                extraction_log["comed_validation"]["utility_name_detected"] = "Ameren"
            else:
                extraction_log["comed_validation"]["comed_utility_mentioned"] = False
                extraction_log["comed_validation"]["utility_name_detected"] = "None"
                validation_issues.append(
                    self.ERROR_MESSAGES["comed_utility_not_mentioned"]
                )
        else:
            # ComEd - check for ComEd-specific names
            comed_utility_patterns = [
                r"\bComEd\b",  # ComEd (exact word)
                r"\bCom\s*Ed\b",  # Com Ed (with optional space)
                r"\bCommonwealth\s+Edison",  # Commonwealth Edison
            ]
            utility_mentioned = any(
                re.search(pattern, text, re.IGNORECASE)
                for pattern in comed_utility_patterns
            )
            if utility_mentioned:
                extraction_log["comed_validation"]["comed_utility_mentioned"] = True
                extraction_log["comed_validation"]["utility_name_detected"] = "ComEd"
            else:
                extraction_log["comed_validation"]["comed_utility_mentioned"] = False
                extraction_log["comed_validation"]["utility_name_detected"] = "None"
                validation_issues.append(
                    self.ERROR_MESSAGES["comed_utility_not_mentioned"]
                )

        # 12. Agent Authorization Checkbox - CHECK IF PRESENT AND MARKED
        # Some COMED LOAs have an agent authorization checkbox that must be checked if present
        # Pattern to detect the agent authorization text
        agent_auth_pattern = (
            r"By\s+checking\s+this\s+box.*?Authorized\s+Person.*?indicates.*?"
            r"(?:s/he\s+is|is)\s+an\s+agent\s+for\s+the\s+Customer.*?"
            r"(?:written\s+agreement|granted\s+the\s+authority).*?"
            r"(?:indemnifies|executing\s+this\s+Authorization)"
        )

        # Check if agent authorization section exists
        has_agent_auth_section = bool(
            re.search(agent_auth_pattern, text, re.IGNORECASE | re.DOTALL)
        )

        if has_agent_auth_section:
            # Agent authorization section found - now check if checkbox is marked
            extraction_log["comed_validation"]["agent_auth_section_found"] = True

            # Look for checkbox marker near the agent authorization text
            # This could be :selected:, checkmark symbols, or X marks near "By checking this box"
            agent_checkbox_patterns = [
                r":selected:.*?By\s+checking\s+this\s+box",
                r"By\s+checking\s+this\s+box.*?:selected:",
                r"[☑✓✗X]\s*By\s+checking\s+this\s+box",
                r"By\s+checking\s+this\s+box\s+the\s+Authorized\s+Person",  # Presence of text suggests it might be checked
            ]

            # Check if checkbox appears to be marked
            checkbox_marked = any(
                re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                for pattern in agent_checkbox_patterns
            )

            # Also check selection_marks from OCR near the agent authorization text
            # If we have selection marks and one is "selected" near the text, consider it marked
            if not checkbox_marked and extraction_log.get("selection_marks"):
                # Find the position of the agent auth text
                agent_match = re.search(
                    agent_auth_pattern, text, re.IGNORECASE | re.DOTALL
                )
                if agent_match:
                    agent_start = agent_match.start()
                    agent_match.end()

                    # Check if any selection marks are nearby (within reasonable text distance)
                    # This is a heuristic - if there's a selected mark in the first 500 chars of the doc, it might be this one
                    early_selected_marks = [
                        mark
                        for mark in extraction_log["selection_marks"]
                        if mark.get("state") == "selected" and mark.get("page", 1) == 1
                    ]

                    if (
                        early_selected_marks and agent_start < 1000
                    ):  # If agent auth text is early in document
                        checkbox_marked = True

            extraction_log["comed_validation"][
                "agent_checkbox_marked"
            ] = checkbox_marked

            if not checkbox_marked:
                validation_issues.append(
                    self.ERROR_MESSAGES["comed_agent_checkbox_not_marked"]
                )
        else:
            # Agent authorization section not found - this is OK, not all COMED LOAs have it
            extraction_log["comed_validation"]["agent_auth_section_found"] = False

        return validation_issues

    def validate_firstenergy_required_fields(
        self, text: str, extraction_log: Dict
    ) -> List[str]:
        """Validate FirstEnergy-specific Account/SDI Numbers and Ohio phrase utility using code-level checks.
        SIMPLIFIED VERSION: Only validates Account/SDI Numbers and Ohio phrase utility name.
        All other fields (customer name, phone, address, CRES provider, etc.) are handled by GPT-4o Vision in Layer 2.

        Args:
            text: Extracted text from the document
            extraction_log: The extraction log to store validation results

        Returns:
            List of validation issues found (empty if validation passes)
        """
        validation_issues = []

        # Initialize FirstEnergy validation structure (minimal - only for fields validated here)
        extraction_log["firstenergy_validation"] = {
            "account_numbers_found": False,
            "ohio_phrase_utility_valid": False,
            "validation_method": "code_level_check",
        }

        # Extract reference to avoid repeated dictionary lookups
        fe_validation = extraction_log["firstenergy_validation"]

        # ACCOUNT/SDI NUMBERS VALIDATION (Code-level check as backup)
        account_patterns = [
            r"Account[/\s]*SDI\s+(?:Number|#|No\.?)[:\s]*(\d{8,})",
            r"Account\s+(?:Number|#|No\.?)[:\s]*(\d{8,})",
            r"SDI\s+(?:Number|#|No\.?)[:\s]*(\d{8,})",
            r"\b(\d{8,20})\b",  # Generic number pattern
        ]

        account_numbers = extract_account_numbers(
            account_patterns, text, min_length=8, max_length=20
        )

        # Check for attachment indicators
        attachment_indicators = [
            "see attached",
            "see below",
            "attached",
            "attachment",
            "see acceptable attachments",
        ]
        text_lower = text.lower()  # Extract once to avoid repeated calls
        has_attachment_note = any(
            indicator in text_lower for indicator in attachment_indicators
        )

        if account_numbers or has_attachment_note:
            fe_validation["account_numbers_found"] = True
            if account_numbers:
                fe_validation["account_numbers"] = account_numbers
                fe_validation["account_count"] = len(account_numbers)
            if has_attachment_note:
                fe_validation["has_attachment_indicator"] = True
        else:
            validation_issues.append(
                self.ERROR_MESSAGES["firstenergy_account_numbers_missing"]
            )

        # OHIO AUTHORIZATION STATEMENT VALIDATION (Only check utility name in Ohio phrase)
        # Look for Ohio authorization statement
        ohio_phrase_pattern = (
            r"I\s+realize\s+that\s+under\s+the\s+rules\s+and\s+regulations.*?(?=\n\n|$)"
        )
        ohio_section_match = re.search(
            ohio_phrase_pattern, text, re.IGNORECASE | re.DOTALL
        )

        if ohio_section_match:
            ohio_section = ohio_section_match.group(0)

            # Validate utility name in Ohio phrase (Only thing we validate in Ohio statement)
            # Must be CEI, OE, TE, Toledo Edison, or The Illuminating Company (or Illuminating Co.)
            # CRITICAL FIX: Do NOT accept generic "FirstEnergy" or "First Energy" - only specific UDCs
            # CEI = Cleveland Electric Illuminating, OE = Ohio Edison, TE = Toledo Edison
            valid_fe_utilities = [
                "CEI",
                "CLEVELAND ELECTRIC ILLUMINATING",
                "CLEVELAND ILLUMINATING",
                "OE",
                "OHIO EDISON",
                "TE",
                "TOLEDO EDISON",
                "THE ILLUMINATING COMPANY",
                "THE ILLUMINATING CO",
                "ILLUMINATING COMPANY",
                "ILLUMINATING CO",
            ]

            # Look for utility mentions in the Ohio phrase
            utility_name_in_phrase = None

            # Check for invalid generic names that should be rejected
            invalid_generic_names = ["FIRSTENERGY", "FIRST ENERGY", "FE"]
            ohio_section_upper = (
                ohio_section.upper()
            )  # Extract once to avoid repeated calls
            has_invalid_generic = any(
                generic in ohio_section_upper for generic in invalid_generic_names
            )

            for valid_utility in valid_fe_utilities:
                if valid_utility in ohio_section_upper:
                    utility_name_in_phrase = valid_utility
                    break

            # CRITICAL: Reject if generic FirstEnergy is found OR if no valid utility is found
            if utility_name_in_phrase is not None and not has_invalid_generic:
                fe_validation["ohio_phrase_utility_valid"] = True
                fe_validation["ohio_phrase_utility_name"] = utility_name_in_phrase
            else:
                fe_validation["ohio_phrase_utility_valid"] = False
                if has_invalid_generic:
                    # More specific error message for generic FirstEnergy
                    fe_validation["invalid_generic_utility_found"] = True
                    validation_issues.append(
                        self.ERROR_MESSAGES["firstenergy_wrong_utility_in_ohio_phrase"]
                    )
                else:
                    validation_issues.append(
                        self.ERROR_MESSAGES["firstenergy_wrong_utility_in_ohio_phrase"]
                    )

        # NOTE: All other field validations (customer name, phone, address, authorized person/title,
        # CRES provider fields, Ohio signature/date, form type) are handled by GPT-4o Vision in Layer 2
        # This code-level function only validates Account/SDI Numbers and Ohio phrase utility

        return validation_issues

    def validate_aep_required_fields(
        self, text: str, extraction_log: Dict
    ) -> List[str]:
        """Validate AEP-specific Account/SDI Numbers and Ohio phrase utility using code-level checks.
        SIMPLIFIED VERSION: Only validates Account/SDI Numbers and Ohio phrase utility name.
        All other fields (customer name, phone, address, CRES provider, etc.) are handled by GPT-4o Vision in Layer 2.

        Args:
            text: Extracted text from the document
            extraction_log: The extraction log to store validation results

        Returns:
            List of validation issues found (empty if validation passes)
        """
        validation_issues = []

        # Initialize AEP validation structure (minimal - only for fields validated here)
        extraction_log["aep_validation"] = {
            "account_numbers_found": False,
            "ohio_phrase_utility_valid": False,
            "validation_method": "code_level_check",
        }

        # Extract reference to avoid repeated dictionary lookups
        aep_validation = extraction_log["aep_validation"]

        # ACCOUNT/SDI NUMBERS VALIDATION (Code-level check as backup)
        # CRITICAL: For AEP, "see attached" means NOTHING - we need ACTUAL account numbers
        # Multi-page scan will find actual accounts in attachments
        account_patterns = [
            r"Account[/\s]*SDI\s+(?:Number|#|No\.?)[:\s]*(\d{8,})",
            r"Account\s+(?:Number|#|No\.?)[:\s]*(\d{8,})",
            r"SDI\s+(?:Number|#|No\.?)[:\s]*(\d{8,})",
            r"\b(\d{8,20})\b",  # Generic number pattern
        ]

        account_numbers = extract_account_numbers(
            account_patterns, text, min_length=8, max_length=20
        )

        # CRITICAL FIX: For AEP, IGNORE "see attached" text - it doesn't mean anything
        # Only accept ACTUAL account numbers found via multi-page scan
        if account_numbers:
            aep_validation["account_numbers_found"] = True
            aep_validation["account_numbers"] = account_numbers
            aep_validation["account_count"] = len(account_numbers)
        else:
            # No account numbers found in initial scan
            # Multi-page scan will run later and update this if accounts are found
            aep_validation["account_numbers_found"] = False
            aep_validation["account_field_empty"] = True

        # OHIO AUTHORIZATION STATEMENT VALIDATION (Only check utility name in Ohio phrase)
        # Look for Ohio authorization statement
        ohio_phrase_pattern = (
            r"I\s+realize\s+that\s+under\s+the\s+rules\s+and\s+regulations.*?(?=\n\n|$)"
        )
        ohio_section_match = re.search(
            ohio_phrase_pattern, text, re.IGNORECASE | re.DOTALL
        )

        if ohio_section_match:
            ohio_section = ohio_section_match.group(0)

            # Validate utility name in Ohio phrase (Only thing we validate in Ohio statement)
            # Must be AEP, AEP Ohio, CSPC, OPC, Columbus Southern Power, or Ohio Power Company
            # CRITICAL: Do NOT accept generic names - only specific AEP UDCs
            valid_aep_utilities = [
                "AEP",
                "AEP OHIO",
                "AMERICAN ELECTRIC POWER",
                "CSPC",
                "COLUMBUS SOUTHERN POWER COMPANY",
                "COLUMBUS SOUTHERN POWER",
                "OPC",
                "OHIO POWER COMPANY",
                "OHIO POWER",
            ]

            # Look for utility mentions in the Ohio phrase
            utility_name_in_phrase = None
            ohio_section_upper = (
                ohio_section.upper()
            )  # Extract once to avoid repeated calls

            for valid_utility in valid_aep_utilities:
                if valid_utility in ohio_section_upper:
                    utility_name_in_phrase = valid_utility
                    break

            # Validate if valid AEP utility found
            if utility_name_in_phrase is not None:
                aep_validation["ohio_phrase_utility_valid"] = True
                aep_validation["ohio_phrase_utility_name"] = utility_name_in_phrase
            else:
                aep_validation["ohio_phrase_utility_valid"] = False
                validation_issues.append(
                    self.ERROR_MESSAGES["aep_wrong_utility_in_ohio_phrase"]
                )

        # NOTE: All other field validations (customer name, phone, address, authorized person/title,
        # CRES provider fields, Ohio signature/date, form type) are handled by GPT-4o Vision in Layer 2
        # This code-level function only validates Account/SDI Numbers and Ohio phrase utility

        return validation_issues

    def validate_cinergy_required_fields(
        self, text: str, extraction_log: Dict
    ) -> List[str]:
        """Validate CINERGY/DUKE ENERGY-specific required fields (Ohio utility).

        CINERGY (Duke Energy Ohio) requirements:
        1. Account number must be exactly 22 digits
        2. Account number must start with '910'
        3. Account number must have 'Z' as the 13th character (position 12 in 0-indexed)
        4. Signature must be within 1 year validity (Ohio requirement: 12 months)

        Args:
            text: Extracted text from the document
            extraction_log: The extraction log to store validation results

        Returns:
            List of validation issues found (empty if validation passes)
        """
        validation_issues = []

        # Initialize CINERGY validation structure
        extraction_log["cinergy_validation"] = {
            "account_format_valid": False,
            "account_number": None,
            "signature_date_found": False,
            "signature_date_valid": False,
            "signature_date": None,
            "validation_method": "code_level_check",
        }

        # 1. Extract and validate account number format
        # Pattern: 22 characters (21 digits + Z at position 13)
        # Example: 910117129533Z109008636 or 910-117129533-Z-109008636

        # First check if there's an indication of attached spreadsheet for multiple accounts
        # If "See Attached" or "See list below" is present, we need to verify account numbers exist
        attachment_patterns = [
            r"See\s+Attached",
            r"see\s+attached",
            r"See\s+list\s+below",
            r"see\s+list\s+below",
            r"Attached\s+(?:spreadsheet|list|file)",
            r"please\s+(?:see|refer\s+to)\s+(?:attached|attachment|list\s+below)",
            r"multiple\s+account.*\s+(?:attached|spreadsheet)",
            r"attach(?:ed)?\s+spreadsheet",
            r"list(?:ed)?\s+below",
        ]

        has_attachment = any(
            re.search(pattern, text, re.IGNORECASE) for pattern in attachment_patterns
        )

        # Search for account numbers in the document (regardless of attachment indication)
        account_patterns = [
            r"Account\s*(?:Number|#|No\.?|Num)[:\s]*([\d\-\s]*910[\d\-\s]*Z[\d\-\s]*)",  # With label
            r"Acct[:\s]*([\d\-\s]*910[\d\-\s]*Z[\d\-\s]*)",  # Short label
            r"\b(910[\d\-\s]{9,15}Z[\d\-\s]{9,15})\b",  # Pattern with Z, allowing for hyphens/spaces
            r"\b(910\d{9}Z\d{9})\b",  # Explicit pattern with Z at position 13 (no separators)
        ]

        found_accounts = []
        for pattern in account_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found_accounts.extend(matches)

        # Validate each found account
        valid_account = None
        for account in found_accounts:
            # Remove hyphens and spaces but keep letters (for Z)
            account_clean = re.sub(r"[-\s]", "", account)

            # Check if it matches Cinergy format: 22 characters (21 digits + Z at position 13)
            # Pattern: 910XXXXXXXXXZXXXXXXXX where X = digit
            if (
                len(account_clean) == 22
                and account_clean.startswith("910")
                and account_clean[12] == "Z"
                and account_clean[:12].isdigit()
                and account_clean[13:].isdigit()
            ):
                valid_account = account_clean
                extraction_log["cinergy_validation"]["account_format_valid"] = True
                extraction_log["cinergy_validation"]["account_number"] = valid_account
                break

        # Decision logic based on attachment indication and accounts found
        if has_attachment:
            # "See Attached" or "See list below" is present
            if not found_accounts:
                # No accounts found - REJECT (attachment indicated but no accounts provided)
                validation_issues.append(
                    "CINERGY/DUKE ENERGY: Account number field says 'See Attached' or 'See list below' but no account numbers were found in the document"
                )
                extraction_log["cinergy_validation"]["account_format_valid"] = False
            elif not valid_account:
                # Found accounts but none match the required format
                extraction_log["cinergy_validation"][
                    "invalid_accounts"
                ] = found_accounts
                validation_issues.append(
                    self.ERROR_MESSAGES["cinergy_account_format_invalid"]
                )
        else:
            # No attachment indication
            if not found_accounts:
                # No attachment indication AND no accounts found - REJECT
                validation_issues.append(self.ERROR_MESSAGES["cinergy_account_missing"])
            elif not valid_account:
                # Found accounts but none match the required format
                extraction_log["cinergy_validation"][
                    "invalid_accounts"
                ] = found_accounts
                validation_issues.append(
                    self.ERROR_MESSAGES["cinergy_account_format_invalid"]
                )

        # 2. Extract and validate signature date (must be within 1 year for Ohio)
        signature_date_patterns = [
            r"(?:Signature\s+)?Date[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"Dated[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"Date\s+Signed[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})",  # Generic date pattern
        ]

        signature_date = None
        for pattern in signature_date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                signature_date = match.group(1)
                break

        if signature_date:
            extraction_log["cinergy_validation"]["signature_date_found"] = True
            extraction_log["cinergy_validation"]["signature_date"] = signature_date

            # Validate signature date is within 1 year (Ohio requirement)
            # Use existing calculate_signature_validity method
            validity_result = self.calculate_signature_validity(
                signature_date, state="OH", utility="CINERGY"
            )

            if validity_result.get("is_valid"):
                extraction_log["cinergy_validation"]["signature_date_valid"] = True
                extraction_log["cinergy_validation"][
                    "validity_result"
                ] = validity_result
            else:
                extraction_log["cinergy_validation"]["signature_date_valid"] = False
                extraction_log["cinergy_validation"][
                    "validity_result"
                ] = validity_result
                validation_issues.append(
                    self.ERROR_MESSAGES["cinergy_signature_expired"]
                )
        else:
            validation_issues.append(
                self.ERROR_MESSAGES["cinergy_signature_date_missing"]
            )

        # 3. Check that the word "Electric" appears somewhere in the form
        if not re.search(r"\bElectric\b", text, re.IGNORECASE):
            validation_issues.append(
                "CINERGY/DUKE ENERGY: The word 'Electric' must appear in the form to confirm this is an electric utility authorization"
            )

        # 4. Validate utility name in Ohio authorization statement
        # Look for Ohio authorization statement (similar to FirstEnergy)
        ohio_phrase_pattern = r"I\s+realize\s+that\s+under\s+the\s+rules\s+and\s+regulations.*?(?=\n\n|Signature|Date|$)"
        ohio_section_match = re.search(
            ohio_phrase_pattern, text, re.IGNORECASE | re.DOTALL
        )

        if ohio_section_match:
            ohio_section = ohio_section_match.group(0)
            ohio_section_upper = ohio_section.upper()

            # Valid CINERGY/DUKE utility names
            valid_cinergy_utilities = [
                "CINERGY",
                "DUKE ENERGY",
                "DUKE ENERGY OHIO",
                "DUKE",
                "CINERGY CORP",
            ]

            # Invalid utility names that should be rejected
            invalid_cinergy_utilities = [
                "DPL",
                "DAYTON POWER",
                "DAYTON POWER & LIGHT",
                "CONSTELLATION",
                "CONSTELLATION ENERGY",
                "FIRSTENERGY",
                "FIRST ENERGY",
                "CEI",
                "OHIO EDISON",
                "TOLEDO EDISON",
                "AEP",
                "AMERICAN ELECTRIC POWER",
            ]

            # Check for invalid utilities first (these should be rejected)
            utility_name_found = None
            has_invalid_utility = False

            for invalid_util in invalid_cinergy_utilities:
                if invalid_util in ohio_section_upper:
                    has_invalid_utility = True
                    extraction_log["cinergy_validation"][
                        "invalid_utility_found"
                    ] = invalid_util
                    validation_issues.append(
                        f"CINERGY/DUKE ENERGY: Wrong utility name '{invalid_util}' found in Ohio authorization statement - must be CINERGY or DUKE ENERGY OHIO"
                    )
                    break

            # If no invalid utility found, check for valid CINERGY/DUKE names
            if not has_invalid_utility:
                for valid_utility in valid_cinergy_utilities:
                    if valid_utility in ohio_section_upper:
                        utility_name_found = valid_utility
                        extraction_log["cinergy_validation"][
                            "ohio_phrase_utility_valid"
                        ] = True
                        extraction_log["cinergy_validation"][
                            "ohio_phrase_utility_name"
                        ] = utility_name_found
                        break

                # If no valid utility name found either, reject
                if utility_name_found is None:
                    extraction_log["cinergy_validation"][
                        "ohio_phrase_utility_valid"
                    ] = False
                    validation_issues.append(
                        "CINERGY/DUKE ENERGY: Ohio authorization statement must reference CINERGY or DUKE ENERGY OHIO utility"
                    )
        else:
            # No Ohio phrase found at all
            validation_issues.append(
                "CINERGY/DUKE ENERGY: Ohio authorization statement not found in document"
            )

        return validation_issues

    def validate_dayton_required_fields(
        self, text: str, extraction_log: Dict
    ) -> List[str]:
        """Validate Dayton Power & Light-specific Ohio phrase utility requirement.

        SIMPLIFIED VERSION: Only validates Ohio phrase utility name.
        Account number validation is handled separately.

        Args:
            text: Extracted text from the document
            extraction_log: The extraction log to store validation results

        Returns:
            List of validation issues found (empty if validation passes)
        """
        validation_issues = []

        # Initialize Dayton validation structure
        extraction_log["dayton_validation"] = {
            "ohio_phrase_utility_valid": False,
            "validation_method": "code_level_check",
        }

        # Extract reference to avoid repeated dictionary lookups
        dayton_validation = extraction_log["dayton_validation"]

        # OHIO AUTHORIZATION STATEMENT VALIDATION
        ohio_phrase_pattern = (
            r"I\s+realize\s+that\s+under\s+the\s+rules\s+and\s+regulations.*?(?=\n\n|$)"
        )
        ohio_section_match = re.search(
            ohio_phrase_pattern, text, re.IGNORECASE | re.DOTALL
        )

        if ohio_section_match:
            ohio_section = ohio_section_match.group(0)

            # Valid Dayton utility names
            valid_dayton_utilities = [
                "DAYTON",
                "DAYTON POWER",
                "DAYTON POWER & LIGHT",
                "DAYTON POWER AND LIGHT",
                "DP&L",
                "DPL",
                "AES",
                "AES OHIO",
            ]

            # Look for utility mentions in the Ohio phrase
            utility_name_in_phrase = None
            ohio_section_upper = ohio_section.upper()

            for valid_utility in valid_dayton_utilities:
                if valid_utility in ohio_section_upper:
                    utility_name_in_phrase = valid_utility
                    break

            # Validate if valid Dayton utility found
            if utility_name_in_phrase is not None:
                dayton_validation["ohio_phrase_utility_valid"] = True
                dayton_validation["ohio_phrase_utility_name"] = utility_name_in_phrase
            else:
                dayton_validation["ohio_phrase_utility_valid"] = False
                validation_issues.append(
                    "Wrong Utility in Paragraph: Ohio authorization statement for Dayton LOAs must reference "
                    "DAYTON, Dayton Power & Light, or DP&L as the utility company - found different utility"
                )
        else:
            # No Ohio phrase found at all
            validation_issues.append(
                "Ohio authorization statement not found in document - Dayton LOAs require the Ohio PUCO statement"
            )

        return validation_issues

    def detect_meco_subscription_options(self, text: str, extraction_log: Dict) -> None:
        """Detect MECO-specific subscription options (Type of Interval Data Request).
        MECO LOAs have 3 subscription options and exactly ONE must be selected.
        This is only applicable for MECO UDC in New England region.
        """
        # Look for MECO subscription option patterns
        two_weeks_pattern = r"Two\s+Weeks\s+Online\s+Access\s+to\s+Data"
        one_year_pattern = r"One\s+Year\s+Online\s+Access\s+to\s+Data"
        auto_renewing_pattern = (
            r"Auto-Renewing,?\s+One\s+Year\s+Online\s+Access\s+to\s+Data"
        )

        two_weeks_match = re.search(two_weeks_pattern, text, re.IGNORECASE)
        one_year_match = re.search(one_year_pattern, text, re.IGNORECASE)
        auto_renewing_match = re.search(auto_renewing_pattern, text, re.IGNORECASE)

        # Initialize MECO subscription options structure
        extraction_log["meco_subscription_options"] = {
            "detected": False,
            "two_weeks_selected": False,
            "one_year_selected": False,
            "auto_renewing_selected": False,
            "selection_count": 0,
        }

        # If any option is found, MECO subscription options are detected
        extraction_log["meco_subscription_options"]["detected"] = bool(
            two_weeks_match or one_year_match or auto_renewing_match
        )

        if not extraction_log["meco_subscription_options"]["detected"]:
            return

        # Now determine which options are selected
        # Look for X marks or checkboxes near each option
        two_weeks_checkbox_pattern = r"[☑☒✓✗X]\s*Two\s+Weeks\s+Online"
        one_year_checkbox_pattern = r"[☑☒✓✗X]\s*One\s+Year\s+Online"
        auto_renewing_checkbox_pattern = r"[☑☒✓✗X]\s*Auto-Renewing"

        two_weeks_selected = bool(
            re.search(two_weeks_checkbox_pattern, text, re.IGNORECASE)
        )
        one_year_selected = bool(
            re.search(one_year_checkbox_pattern, text, re.IGNORECASE)
        )
        auto_renewing_selected = bool(
            re.search(auto_renewing_checkbox_pattern, text, re.IGNORECASE)
        )

        # Update the extraction log
        extraction_log["meco_subscription_options"][
            "two_weeks_selected"
        ] = two_weeks_selected
        extraction_log["meco_subscription_options"][
            "one_year_selected"
        ] = one_year_selected
        extraction_log["meco_subscription_options"][
            "auto_renewing_selected"
        ] = auto_renewing_selected
        extraction_log["meco_subscription_options"]["selection_count"] = sum(
            [two_weeks_selected, one_year_selected, auto_renewing_selected]
        )

    def detect_potential_initials(self, text: str, extraction_log: Dict) -> None:
        """Detect potential handwritten initials in the text."""

        # Pattern 1: Detecting potential initials after "Initial Box" text (immediate)
        initial_box_pattern = r"Initial Box[^:]*:\s*([A-Za-z]{1,3})"

        # Find all potential initials
        potential_initials = re.findall(initial_box_pattern, text)

        # Store the detected initials
        for initial in potential_initials:
            extraction_log["potential_initials"].append(
                {
                    "text": initial,
                    "is_likely_initial": len(initial)
                    <= 3,  # Most initials are 1-3 characters
                    "context": f"Initial Box: {initial}",
                }
            )

        # Pattern 2: Detect initials at start of line after "Initial Box" section
        # This handles cases like:
        # "Initial Box for release of specific account information to CRES provider listed above:
        #  CR Account/SDI Number Release:"
        initial_box_line_pattern = r"Initial Box[^\n]*:\s*\n\s*([A-Z]{1,3})\s+(?:Account|Residential|Commercial|Historical|Interval)"
        line_initials = re.findall(initial_box_line_pattern, text, re.MULTILINE)

        for initial in line_initials:
            # Avoid duplicates
            if not any(
                item.get("text") == initial
                for item in extraction_log["potential_initials"]
            ):
                extraction_log["potential_initials"].append(
                    {
                        "text": initial,
                        "is_likely_initial": True,
                        "context": f"Initial Box line start: {initial}",
                        "confidence": 0.95,
                    }
                )

        # Also look for standalone 1-2 character strings that might be initials
        # This is more aggressive and might have false positives
        standalone_pattern = r"\b([A-Z][A-Za-z]?)\b"
        standalone_matches = re.findall(standalone_pattern, text)

        for match in standalone_matches:
            if match not in [
                "I",
                "A",
                "OK",
                "NO",
                "US",
                "OH",
                "MI",
                "IL",
                "AZ",
                "NY",
                "CA",
                "TX",
            ]:  # Filter out common non-initial abbreviations
                extraction_log["potential_initials"].append(
                    {
                        "text": match,
                        "is_likely_initial": True,
                        "context": "Standalone potential initial",
                    }
                )

    def calculate_signature_validity(
        self, signature_date_str: str, state: str, utility: str = None
    ) -> Dict:
        """Calculate if signature date is valid based on state and utility-specific rules."""

        try:
            # Check if signature_date_str is None or empty
            if not signature_date_str:
                return {
                    "is_valid": False,
                    "reason": "No signature date provided",
                    "days_old": None,
                    "months_old": None,
                    "years_old": None,
                    "state_limit": None,
                    "calculation_details": "Signature date string is None or empty",
                }

            # Parse various date formats (including dash-separated, digital signature format, and full month names)
            signature_date = None
            date_formats = [
                "%m/%d/%Y",
                "%m/%d/%y",
                "%m-%d-%Y",
                "%m-%d-%y",
                "%Y-%m-%d",
                "%y-%m-%d",
                "%d/%m/%Y",
                "%d-%m-%Y",
                "%Y.%m.%d",
                "%B %d, %Y",
            ]

            for fmt in date_formats:
                try:
                    signature_date = datetime.strptime(signature_date_str.strip(), fmt)
                    break
                except ValueError:
                    continue

            if not signature_date:
                return {
                    "is_valid": False,
                    "reason": f"Could not parse signature date: {signature_date_str}",
                    "days_old": None,
                    "months_old": None,
                    "years_old": None,
                    "state_limit": None,
                    "calculation_details": f"Unable to parse date format: {signature_date_str}",
                }

            # Calculate time difference
            today = datetime.now()
            time_diff = today - signature_date
            days_old = time_diff.days
            months_old = days_old / 30.44  # Average days per month
            years_old = days_old / 365.25  # Average days per year

            # Define state-specific and utility-specific validity periods based on region
            state_utility_limits = self.get_state_utility_limits()

            # Get the appropriate limit based on state and utility
            state_upper = state.upper() if state else "default"

            # First check if the state is supported in the configured limits
            if state_upper in state_utility_limits:
                state_limits = state_utility_limits[state_upper]

                # If utility is specified and exists in state limits, use that
                if utility and utility.upper() in state_limits:
                    limit_info = state_limits[utility.upper()]
                else:
                    # Otherwise use the state default
                    limit_info = state_limits.get(
                        "default", state_utility_limits["default"]
                    )
            else:
                # State not found, use default limits
                limit_info = state_utility_limits["default"]

            max_months = limit_info["months"]

            # Determine validity
            is_valid = months_old <= max_months

            # Create detailed calculation
            calculation_details = f"""
        SIGNATURE DATE VALIDATION CALCULATION:
        - Signature Date: {signature_date.strftime('%m/%d/%Y')}
        - Today's Date: {today.strftime('%m/%d/%Y')}
        - Time Difference: {days_old} days
        - Months Old: {months_old:.1f} months
        - Years Old: {years_old:.1f} years
        - State: {state_upper}
        - Utility: {utility if utility else 'Not specified'}
        - Applicable Limit: {limit_info['name']}
        - Month Limit: {max_months} months
        - Is Valid: {'YES' if is_valid else 'NO'}
        - Reason: {'Within {max_months} month limit' if is_valid else f'Exceeds {max_months} month limit by {months_old - max_months:.1f} months'}
"""

            return {
                "is_valid": is_valid,
                "reason": f"Signature is {months_old:.1f} months old, {'within' if is_valid else 'exceeds'} {limit_info['name']} {max_months}-month limit",
                "days_old": days_old,
                "months_old": round(months_old, 1),
                "years_old": round(years_old, 1),
                "state_limit": max_months,
                "calculation_details": calculation_details,
                "signature_date": signature_date.strftime("%m/%d/%Y"),
                "today_date": today.strftime("%m/%d/%Y"),
            }

        except Exception as e:
            return {
                "is_valid": False,
                "reason": f"Error calculating signature validity: {str(e)}",
                "days_old": None,
                "months_old": None,
                "years_old": None,
                "state_limit": None,
                "calculation_details": f"Calculation error: {str(e)}",
            }

    def calculate_loa_expiration_date(
        self,
        signature_date_str: str,
        state: str,
        utility: str = None,
        extracted_text: str = "",
    ) -> Dict:
        """Calculate LOA expiration date based on signature date and state/utility rules.

        Args:
            signature_date_str: The signature date from the document
            state: The state for determining expiration rules
            utility: The utility company (optional)
            extracted_text: Full document text to check for explicit expiration statements

        Returns:
            Dict containing expiration date, months until expiration, and calculation details
        """

        try:
            # Check if signature_date_str is None or empty
            if not signature_date_str:
                return {
                    "expiration_date": None,
                    "expiration_date_formatted": "Could not calculate - no signature date provided",
                    "months_until_expiration": None,
                    "days_until_expiration": None,
                    "expiration_rule_used": "N/A",
                    "explicit_expiration_found": False,
                    "calculation_details": "Signature date string is None or empty",
                }

            # Parse various date formats (including dash-separated)
            signature_date = None
            date_formats = [
                "%m/%d/%Y",
                "%m/%d/%y",
                "%m-%d-%Y",
                "%m-%d-%y",
                "%Y-%m-%d",
                "%y-%m-%d",
                "%d/%m/%Y",
                "%d-%m-%Y",
            ]

            for fmt in date_formats:
                try:
                    signature_date = datetime.strptime(signature_date_str.strip(), fmt)
                    break
                except ValueError:
                    continue

            if not signature_date:
                return {
                    "expiration_date": None,
                    "expiration_date_formatted": "Could not calculate - invalid signature date",
                    "months_until_expiration": None,
                    "days_until_expiration": None,
                    "expiration_rule_used": "N/A",
                    "explicit_expiration_found": False,
                    "calculation_details": f"Unable to parse signature date format: {signature_date_str}",
                }

            # Check for explicit expiration statements in the document text
            explicit_expiration_months = None
            explicit_expiration_found = False

            # GSECO-specific phrase that indicates a 1-year expiration
            gseco_one_year_pattern = (
                r"Customer\'s signature (?:are|is) valid one year from the sign date"
            )
            is_gseco = utility and (
                "GSECO" in utility.upper() or "GRANITE STATE" in utility.upper()
            )

            # Check for GSECO-specific one-year phrase first
            if is_gseco and re.search(
                gseco_one_year_pattern, extracted_text, re.IGNORECASE
            ):
                explicit_expiration_months = 12  # 1 year = 12 months
                explicit_expiration_found = True
                expiration_rule_used = "GSECO-specific phrase: 'Customer's signature are valid one year from the sign date'"
            else:
                # Standard patterns to look for explicit expiration periods
                expiration_patterns = [
                    r"expire(?:s)?\s+(?:in\s+|after\s+)?(\d+)\s+months?",
                    r"valid\s+for\s+(\d+)\s+months?",
                    r"expires?\s+(\d+)\s+months?\s+(?:from|after)",
                    r"this\s+authorization\s+(?:will\s+)?expire(?:s)?\s+(?:in\s+)?(\d+)\s+months?",
                    r"loa\s+(?:will\s+)?expire(?:s)?\s+(?:in\s+)?(\d+)\s+months?",
                ]

                for pattern in expiration_patterns:
                    matches = re.findall(pattern, extracted_text, re.IGNORECASE)
                    if matches:
                        try:
                            explicit_expiration_months = int(matches[0])
                            explicit_expiration_found = True
                            break
                        except (ValueError, IndexError):
                            continue

            # Get state/utility specific expiration rules
            state_utility_limits = self.get_state_utility_limits()
            state_upper = state.upper() if state else "default"

            # Determine which expiration period to use
            if explicit_expiration_found and explicit_expiration_months is not None:
                # Use explicit expiration period from document
                expiration_months = explicit_expiration_months
                expiration_rule_used = (
                    f"Explicit statement in document: {expiration_months} months"
                )
            else:
                # Use state/utility specific rules
                if state_upper in state_utility_limits:
                    state_limits = state_utility_limits[state_upper]

                    if utility and utility.upper() in state_limits:
                        limit_info = state_limits[utility.upper()]
                    else:
                        limit_info = state_limits.get(
                            "default", state_utility_limits["default"]
                        )
                else:
                    limit_info = state_utility_limits["default"]

                expiration_months = limit_info["months"]
                expiration_rule_used = f"State/Utility rule: {limit_info['name']}"

            # Calculate expiration date
            expiration_date = signature_date + relativedelta(months=expiration_months)

            # Calculate time until expiration
            today = datetime.now()
            time_until_expiration = expiration_date - today
            days_until_expiration = time_until_expiration.days
            months_until_expiration = (
                days_until_expiration / 30.44
            )  # Average days per month

            # Create detailed calculation
            calculation_details = f"""
        LOA EXPIRATION DATE CALCULATION:
        - Signature Date: {signature_date.strftime('%m/%d/%Y')}
        - State: {state_upper}
        - Utility: {utility if utility else 'Not specified'}
        - Explicit Expiration in Document: {'YES - ' + str(explicit_expiration_months) + ' months' if explicit_expiration_found else 'NO'}
        - Expiration Period Used: {expiration_months} months
        - Rule Used: {expiration_rule_used}
        - Calculated Expiration Date: {expiration_date.strftime('%m/%d/%Y')}
        - Today's Date: {today.strftime('%m/%d/%Y')}
        - Days Until Expiration: {days_until_expiration} days
        - Months Until Expiration: {months_until_expiration:.1f} months
        - Status: {'ACTIVE' if days_until_expiration > 0 else 'EXPIRED'}
"""

            return {
                "expiration_date": expiration_date,
                "expiration_date_formatted": expiration_date.strftime("%m/%d/%Y"),
                "months_until_expiration": round(months_until_expiration, 1),
                "days_until_expiration": days_until_expiration,
                "expiration_months_used": expiration_months,
                "expiration_rule_used": expiration_rule_used,
                "explicit_expiration_found": explicit_expiration_found,
                "explicit_expiration_months": (
                    explicit_expiration_months if explicit_expiration_found else None
                ),
                "signature_date": signature_date.strftime("%m/%d/%Y"),
                "is_expired": days_until_expiration <= 0,
                "calculation_details": calculation_details,
            }

        except Exception as e:
            return {
                "expiration_date": None,
                "expiration_date_formatted": f"Error calculating expiration date: {str(e)}",
                "months_until_expiration": None,
                "days_until_expiration": None,
                "expiration_rule_used": "Error",
                "explicit_expiration_found": False,
                "calculation_details": f"Calculation error: {str(e)}",
            }

    def get_state_utility_limits(self) -> Dict:
        """Get state and utility-specific time limits based on the configured region."""

        # Great Lakes Region state-specific limits
        great_lakes_limits = {
            "OH": {"default": {"months": 12, "name": "Ohio (1 year)"}},
            "IL": {"default": {"months": 6, "name": "Illinois (6 months)"}},
            "MI": {"default": {"months": 12, "name": "Michigan (1 year)"}},
            "default": {"months": 12, "name": "Great Lakes Default (1 year)"},
        }

        # New England Region state and utility-specific limits
        new_england_limits = {
            "ME": {
                "default": {"months": 60, "name": "Maine (5 years)"},
                "BHE": {"months": 60, "name": "Bangor Hydro Electric (5 years)"},
                "CMP": {"months": 60, "name": "Central Maine Power (5 years)"},
            },
            "MA": {
                "default": {"months": 48, "name": "Massachusetts (4 years)"},
                "FGE": {"months": 48, "name": "Fitchburg Gas & Electric (4 years)"},
                "MECO": {"months": 12, "name": "Massachusetts Electric (1 year)"},
                "NANT": {"months": 12, "name": "Nantucket Electric (1 year)"},
                "BECO": {"months": 48, "name": "Boston Edison (4 years)"},
                "CECO": {"months": 48, "name": "Commonwealth Electric (4 years)"},
                "EVERSOURCE": {"months": 48, "name": "Eversource (4 years)"},
            },
            "NH": {
                "default": {"months": 48, "name": "New Hampshire (4 years)"},
                "NHEC": {"months": 12, "name": "New Hampshire Electric Co-op (1 year)"},
                "GSECO": {"months": 12, "name": "Granite State Electric (1 year)"},
                "PSNH": {
                    "months": 48,
                    "name": "Public Service of New Hampshire (4 years)",
                },
                "UES": {"months": 48, "name": "Unitil Energy Systems (4 years)"},
                "EVERSOURCE": {"months": 48, "name": "Eversource (4 years)"},
            },
            "RI": {
                "default": {"months": 24, "name": "Rhode Island (2 years)"},
                "NECO": {"months": 24, "name": "Narragansett Electric (2 years)"},
            },
            "CT": {
                "default": {"months": 48, "name": "Connecticut (4 years)"},
                "CLP": {"months": 48, "name": "Connecticut Light & Power (4 years)"},
                "UI": {"months": 48, "name": "United Illuminating (4 years)"},
                "VMECO": {
                    "months": 48,
                    "name": "Western Massachusetts Electric (4 years)",
                },
            },
            "default": {"months": 48, "name": "New England Default (4 years)"},
        }

        # Return the appropriate limits based on region
        if self.region == "New England":
            return new_england_limits
        else:
            return great_lakes_limits

    def get_utility_patterns(self) -> List[str]:
        """Get region-specific utility company regex patterns."""

        # Great Lakes utility patterns - Updated with official UDC Sub-Types and utilities
        great_lakes_patterns = [
            # AEP-Ohio utilities
            r"\bAEP\s*-?\s*Ohio\b|\bAEP\b",
            r"\bCSPC\b|\bColumbus\s+Southern\s+Power\s+Company\b",
            r"\bOPC\b|\bOhio\s+Power\s+Company\b",
            # First Energy-Ohio utilities
            r"\bFirst\s*Energy\s*-?\s*Ohio\b|\bFirstEnergy\b",
            r"\bCEI\b|\bCleveland\s+Illuminating\s+Co(?:mpany)?\b",
            r"\bOE\b|\bOhio\s+Edison\b",
            r"\bTE\b|\bToledo\s+Edison(?:\s*&?\s*Ohio\s+Edison)?\b",
            # Duke Energy
            r"\bDuke\s+Energy\b|\bCINERGY\b",
            # Dayton Power & Light
            r"\bDAYTON\b|\bDayton\s+Power\s*&\s*Light\b",
            # Ameren Illinois utilities
            r"\bAmeren\s+Illinois\b",
            r"\bCILCO\b|\bCentral\s+Illinois\s+Light\s+Company\b",
            r"\bCIPS\b|\bCentral\s+Illinois\s+Public\s+Service\s+Company\b",
            r"\bIP\b|\bIllinois\s+Power\b",
            # Commonwealth Edison
            r"\bCOMED\b|\bCommonwealth\s+Edison\b",
            # Indiana Michigan Power
            r"\bIMP\b|\bIndiana\s+Michigan\s+Power\b",
            # Consumers Energy
            r"\bCEC\b|\bConsumers\s+Energy\b",
            # Detroit Edison
            r"\bDTE\b|\bDetroit\s+Edison\b",
            # UMERC utilities
            r"\bUMERC\b",
            r"\bWE\b|\bWisconsin\s+Electric\b",
            r"\bWPSC\b|\bWisconsin\s+Public\s+Service\s+Corporation\b",
            # Upper Peninsula Power
            r"\bUPPCO\b|\bUpper\s+Peninsula\s+Power\s+Company\b",
            # Legacy patterns for backward compatibility
            r"\bColumbia\s+Gas(?:\s+of\s+Ohio)?\b",
            r"\bDominion\s+Energy(?:\s+Ohio)?\b",
        ]

        # New England utility patterns
        new_england_patterns = [
            # Maine
            r"\bBangor\s+Hydro\s+Electric\b|\bBHE\b",
            r"\bCentral\s+Maine\s+Power\b|\bCMP\b",
            # Massachusetts
            r"\bFitchburg\s+Gas\s+&\s+Electric\b|\bFGE\b",
            r"\bMassachusetts\s+Electric\b|\bMECO\b",
            r"\bBoston\s+Edison\b|\bBECO\b",
            r"\bCommonwealth\s+Electric\b|\bCECO\b",
            r"\bEversource\b",  # Multi-state utility
            # New Hampshire
            r"\bNew\s+Hampshire\s+Electric\s+Co-?op\b|\bNHEC\b",
            r"\bGranite\s+State\s+Electric\b|\bGSECO\b",
            r"\bPublic\s+Service\s+of\s+New\s+Hampshire\b|\bPSNH\b",
            r"\bUnitil\s+Energy\s+Systems\b|\bUES\b",
            # Rhode Island
            r"\bNarragansett\s+Electric\b|\bNECO\b",
            # Connecticut
            r"\bConnecticut\s+Light\s+&\s+Power\b|\bCLP\b",
            r"\bUnited\s+Illuminating\b|\bUI\b",
            r"\bWestern\s+Massachusetts\s+Electric\b|\bVMECO\b",
        ]

        # Return the appropriate patterns based on region
        if self.region == "New England":
            return new_england_patterns
        else:
            return great_lakes_patterns

    def detect_utility_companies(self, text: str, extraction_log: Dict) -> None:
        """Detect utility companies mentioned in the text using region-specific pattern matching."""

        # Get region-specific utility patterns
        specific_utilities = self.get_utility_patterns()

        # Generic utility keywords for fallback detection
        utility_keywords = [
            "energy",
            "power",
            "electric",
            "utility",
            "utilities",
            "edison",
            "illuminating",
            "company",
            "companies",
            "gas",
            "light",
            "hydro",
            "narragansett",
            "massachusetts",
            "maine",
            "new hampshire",
            "connecticut",
            "rhode island",
            "duke",
            "firstenergy",
            "aep",
        ]

        processed_matches = set()

        # First, look for specific known utility patterns
        for pattern in specific_utilities:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_match = match.strip()
                if clean_match and clean_match not in processed_matches:
                    processed_matches.add(clean_match)
                    # Count exact matches in text
                    count = len(re.findall(re.escape(clean_match), text, re.IGNORECASE))
                    extraction_log["detected_utilities"].append(
                        {
                            "name": clean_match,
                            "mentions": count,
                            "detection_method": "specific_pattern",
                        }
                    )

        # Second, look for utility company names using improved bounded pattern
        # This pattern looks for 1-4 words before a utility keyword, bounded by word boundaries
        improved_pattern = (
            r"\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,3})\s+(?:"
            + "|".join(utility_keywords)
            + r")(?:\s+(?:Company|Corp|Corporation|Inc|LLC))?\b"
        )
        utility_matches = re.findall(improved_pattern, text, re.IGNORECASE)

        for match in utility_matches:
            # Clean up the match and add the utility keyword back
            clean_match = match.strip()
            if len(clean_match) > 2 and clean_match not in processed_matches:
                # Find the full utility name in context
                full_match_pattern = (
                    re.escape(clean_match)
                    + r"\s+(?:"
                    + "|".join(utility_keywords)
                    + r")(?:\s+(?:Company|Corp|Corporation|Inc|LLC))?"
                )
                full_matches = re.findall(full_match_pattern, text, re.IGNORECASE)

                if full_matches:
                    full_name = full_matches[0].strip()
                    if full_name not in processed_matches:
                        processed_matches.add(full_name)
                        count = len(
                            re.findall(re.escape(full_name), text, re.IGNORECASE)
                        )
                        extraction_log["detected_utilities"].append(
                            {
                                "name": full_name,
                                "mentions": count,
                                "detection_method": "pattern_matching",
                            }
                        )

        # Third, look for simple utility references in Ohio statement context
        ohio_statement_pattern = r"(?:allow|give)\s+([A-Z][A-Za-z\s&\-\.]{2,25}?)\s+(?:to\s+release|permission)"
        ohio_matches = re.findall(ohio_statement_pattern, text, re.IGNORECASE)

        for match in ohio_matches:
            clean_match = match.strip()
            # Filter out obvious non-utility matches
            if (
                len(clean_match) > 2
                and clean_match not in processed_matches
                and not any(
                    word in clean_match.lower()
                    for word in [
                        "constellation",
                        "cres",
                        "provider",
                        "customer",
                        "above",
                        "information",
                    ]
                )
            ):

                processed_matches.add(clean_match)
                count = len(re.findall(re.escape(clean_match), text, re.IGNORECASE))
                extraction_log["detected_utilities"].append(
                    {
                        "name": clean_match,
                        "mentions": count,
                        "detection_method": "ohio_statement_context",
                    }
                )

        # Finally, look for generic utility references
        generic_utilities = [
            "utility",
            "utilities",
            "electric company",
            "power company",
            "energy company",
            "gas company",
            "local distribution company",
        ]

        for generic in generic_utilities:
            if generic.lower() in text.lower():
                extraction_log["detected_utilities"].append(
                    {
                        "name": "Generic Utility Reference",
                        "reference": generic,
                        "mentions": text.lower().count(generic.lower()),
                        "detection_method": "generic_reference",
                    }
                )

    def check_need_critical_checkbox_verification(
        self, extraction_log: Dict, extracted_text: str
    ) -> bool:
        """Determine if we need to perform second pass verification on critical checkboxes.

        Args:
            extraction_log: The current extraction log
            extracted_text: The extracted text from the document

        Returns:
            bool: True if we need to verify critical checkboxes
        """
        # Extract the selection marks
        selection_marks = extraction_log.get("selection_marks", [])

        # Check if there are any unselected marks
        unselected_marks = [
            mark for mark in selection_marks if mark.get("state") == "unselected"
        ]

        # Define critical keywords
        critical_keywords = [
            "Interval Historical Energy Usage Data",
            "Account/SDI Number",
            "Historical Usage Data",
            "Account / SDI Number Release",
            "Interval Historical Energy Usage Data Release",
        ]

        # Look for critical keywords in text (with null check)
        extracted_text_safe = extracted_text or ""
        critical_found = any(
            keyword.lower() in extracted_text_safe.lower()
            for keyword in critical_keywords
        )

        # Look for critical keywords in unselected marks' content
        critical_unselected = False
        for mark in unselected_marks:
            mark_content = mark.get("content") or ""
            mark_content = mark_content.lower() if mark_content else ""
            if any(keyword.lower() in mark_content for keyword in critical_keywords):
                critical_unselected = True
                break

        # Additional check: Look for specific phrases that indicate required checkboxes
        # that might be missing initials
        required_phrases = [
            r"Account.*?SDI.*?Number.*?Release",
            r"Interval.*?Historical.*?Energy.*?Usage.*?Data.*?Release",
            r"Historical.*?Usage.*?Data.*?Release",
        ]

        has_required_phrases = False
        for phrase in required_phrases:
            if extracted_text_safe and re.search(
                phrase, extracted_text_safe, re.IGNORECASE
            ):
                has_required_phrases = True
                break

        # Need verification if:
        # 1. Critical keywords are found in text AND we have unselected marks
        # 2. OR we have required phrases and unselected marks (suggesting missing initials)
        # 3. OR all marks are unselected and we found critical content (like Baldwin Wallace case)
        all_unselected = len(selection_marks) > 0 and len(unselected_marks) == len(
            selection_marks
        )

        should_verify = (
            (critical_found and len(unselected_marks) > 0)
            or (has_required_phrases and len(unselected_marks) > 0)
            or (all_unselected and critical_found)
            or critical_unselected
        )

        return should_verify

    def _get_gpt4o_analysis_with_fallback(
        self, system_prompt: str, user_prompt: str
    ) -> str:
        """Get GPT-4o analysis using the standard OpenAI service - no fallback logic needed here."""

        try:
            # Use the standard OpenAI service
            if self.openai_4o_service:
                gpt_response_data = self.openai_4o_service.process_with_prompts(
                    system_prompt, user_prompt, max_token=3000, raw_response=True
                )
                gpt_response = (
                    gpt_response_data[0]["ai_result"][0]["result"]
                    if gpt_response_data and gpt_response_data[0]["ai_result"]
                    else None
                )

                if gpt_response and gpt_response.strip():
                    return gpt_response

            # Return error message if no response
            return "Error: No response from OpenAI service"

        except Exception as e:
            # Return error for any unexpected errors
            return f"Error: GPT-4o analysis failed with exception: {str(e)}"

    def _validate_initial_boxes(
        self,
        pdf_path,
        extraction_log,
        extracted_text,
        udc_name,
        keywords,
        log_key,
        verify_method_name,
    ):
        """Helper method to validate initial boxes for a given UDC.

        Args:
            pdf_path: Path to the PDF file
            extraction_log: The extraction log dictionary
            extracted_text: The extracted text from the document
            udc_name: Name of the UDC for logging (e.g., 'CINERGY', 'DAYTON')
            keywords: List of keywords to match in UDC name (e.g., ['CINERGY', 'DUKE'])
            log_key: Key in extraction_log to store validation data (e.g., 'cinergy_validation')
            verify_method_name: Name of the GPT-4o verification method to call

        Returns:
            Tuple of (updated extraction_log, updated extracted_text)
        """
        if not (self.provided_udc and pdf_path):
            return extraction_log, extracted_text

        # Check if any keyword matches
        if not any(keyword in self.provided_udc.upper() for keyword in keywords):
            return extraction_log, extracted_text

        try:
            self.logger.info(
                f"{udc_name} document detected - Running GPT-4o initial box verification..."
            )

            # Call the verification method
            verify_method = getattr(
                self.gpt4o_verification_integration, verify_method_name
            )
            extraction_log = verify_method(pdf_path, extraction_log)

            # Check results
            validation_data = extraction_log.get(log_key, {})
            initial_boxes = validation_data.get("initial_boxes", {})

            # Check for X marks first
            if initial_boxes.get("x_mark_count", 0) > 0:
                extracted_text += f"\n\n{udc_name} INITIAL BOX VALIDATION: Found {initial_boxes.get('x_mark_count')} X mark(s) - REJECT\n"
                self.logger.warning(
                    f"{udc_name}: {initial_boxes.get('x_mark_count')} X mark(s) detected in initial boxes - will add to rejection reasons"
                )
            # Then check for empty boxes
            elif initial_boxes.get("empty_box_count", 0) > 0:
                extracted_text += f"\n\n{udc_name} INITIAL BOX VALIDATION: Found {initial_boxes.get('empty_box_count')} empty box(es) - REJECT\n"
                self.logger.warning(
                    f"{udc_name}: {initial_boxes.get('empty_box_count')} empty box(es) detected - will add to rejection reasons"
                )
            else:
                extracted_text += f"\n\n{udc_name} INITIAL BOX VALIDATION: Both boxes filled with valid letter initials - PASS\n"
                self.logger.info(
                    f"{udc_name}: Both initial boxes have valid letter initials"
                )

        except Exception as e:
            self.logger.error(
                f"{udc_name} initial box GPT-4o verification error: {str(e)}"
            )
            extracted_text += (
                f"\n\n{udc_name} INITIAL BOX GPT-4O VERIFICATION ERROR: {str(e)}\n"
            )

        return extraction_log, extracted_text

    def _quick_validate_gseco_document(
        self, extraction_log: Dict, document_id: str
    ) -> Dict:
        """
        Special quick validation method for GSECO documents that bypasses most validation requirements.
        Only checks for critical issues like missing signatures and bad email domains.

        Args:
            extraction_log: The extraction log with document data
            document_id: The ID of the document being validated

        Returns:
            Dict: A validation result with ACCEPT status unless critical issues are found
        """
        extracted_text = extraction_log["extracted_text"]

        # Default to ACCEPT for GSECO documents
        status = "ACCEPT"
        rejection_reasons = []

        # Check for signature - look for common signature indicators
        signature_indicators = ["signature", "signed", "/s/", "authorized by"]
        has_signature = any(
            indicator in extracted_text.lower() for indicator in signature_indicators
        )

        if not has_signature:
            status = "REJECT"
            rejection_reasons.append("Missing customer signature")

        # Check for bad email domains that should always be rejected
        bad_email_domains = [
            "@exelon.com",
            "@exeloncorp.com",
            "@strategic.com",
            "@integrys.com",
            "@pepco.com",
        ]
        for domain in bad_email_domains:
            if domain in extracted_text.lower():
                status = "REJECT"
                rejection_reasons.append(
                    f"Email domain in CRES provider section is non-Constellation: {domain}"
                )

        # Extract signature date and calculate expiration
        signature_date_pattern = r"(\d{1,2}/\d{1,2}/\d{2,4})"
        signature_dates = re.findall(signature_date_pattern, extracted_text)

        expiration_date_formatted = "Not calculated"
        expiration_details = None

        if signature_dates:
            # Use the first date found
            signature_date_str = signature_dates[0]

            try:
                # Try to parse the date
                date_formats = [
                    "%m/%d/%Y",
                    "%m/%d/%y",
                    "%Y-%m-%d",
                    "%y-%m-%d",
                    "%d/%m/%Y",
                ]
                signature_date = None

                for fmt in date_formats:
                    try:
                        signature_date = datetime.strptime(
                            signature_date_str.strip(), fmt
                        )
                        break
                    except ValueError:
                        continue

                if signature_date:
                    # Calculate expiration date (1 year for GSECO)
                    today = datetime.now()

                    # Calculate expiration date (1 year for GSECO)
                    expiration_date = signature_date + relativedelta(months=12)
                    expiration_date_formatted = expiration_date.strftime("%m/%d/%Y")

                    # Calculate time until expiration
                    time_until_expiration = expiration_date - today
                    days_until_expiration = time_until_expiration.days
                    months_until_expiration = days_until_expiration / 30.44

                    expiration_details = {
                        "expiration_date": expiration_date_formatted,
                        "months_until_expiration": round(months_until_expiration, 1),
                        "days_until_expiration": days_until_expiration,
                        "expiration_months_used": 12,
                        "expiration_rule_used": "GSECO rule: 1 year from signature date",
                        "signature_date": signature_date.strftime("%m/%d/%Y"),
                        "is_expired": days_until_expiration <= 0,
                        "calculation_details": f"GSECO: Signature {signature_date.strftime('%m/%d/%Y')} + 12 months = Expires {expiration_date_formatted}",
                    }
            except Exception:
                # If there's any issue parsing the date, just continue
                pass

        # Check if LOA is expired
        if expiration_details and expiration_details.get("is_expired"):
            status = "REJECT"
            rejection_reasons.append(f"LOA expired on {expiration_date_formatted}")

        # Check for account number - GSECO specific validation
        # GSECO account numbers: at least 8 numeric digits, no letters, can have dash between numbers
        # Example: 44624069 or 44624069-12345678

        # GSECO-specific pattern: 8+ digits, optional dash and more digits, no letters
        gseco_account_pattern = r"\b(\d{8,})(?:-\d{8,})?\b"

        account_number_found = False

        # First try GSECO-specific pattern
        gseco_matches = re.findall(gseco_account_pattern, extracted_text)
        if gseco_matches:
            account_number_found = True

        # If GSECO pattern didn't find anything, try general account number patterns
        if not account_number_found:
            general_account_patterns = [
                r"Account\s*(?:Number|#|No\.?)[\s:]*([0-9\-]{8,})",
                r"Acct\s*(?:Number|#|No\.?)[\s:]*([0-9\-]{8,})",
            ]

            for pattern in general_account_patterns:
                matches = re.findall(pattern, extracted_text, re.IGNORECASE)
                if matches:
                    for match in matches:
                        clean_match = match.strip()
                        # For GSECO: must be at least 8 digits, no letters
                        digits_only = re.sub(r"[^0-9]", "", clean_match)
                        if len(digits_only) >= 8:
                            account_number_found = True
                            break
                    if account_number_found:
                        break

        if not account_number_found:
            status = "REJECT"
            rejection_reasons.append("Missing account number")

        # Check for explicitly marked rejection terms
        if "REJECTED" in extracted_text.upper() or "VOID" in extracted_text.upper():
            status = "REJECT"
            rejection_reasons.append("Document explicitly marked as rejected or void")

        # Build the validation result
        validation_result = {
            "document_id": document_id,
            "fileName": document_id,
            "validation_status": status,
            "status": status,
            "rejectionReasons": rejection_reasons,
            "all_rejection_reasons": rejection_reasons,
            "expiration_date": expiration_date_formatted,
            "ocr_success": extraction_log["extraction_success"],
            "extracted_text_length": len(extracted_text),
            "processing_timestamp": datetime.now().isoformat(),
            "utility_identified": "GSECO",
            "state_identified": "NH",
            "bypass_mode": True,
            "bypass_reason": "GSECO documents have special validation bypass enabled",
        }

        # Add detailed expiration info if available
        if expiration_details:
            validation_result["expiration_details"] = expiration_details

        return validation_result

    def validate_with_universal_utility_recognition(
        self, extraction_log: Dict, document_id: str, pdf_path: str = None
    ) -> Dict:
        """Validate LOA using advanced form field detection with universal utility name validation."""

        # SPECIAL HANDLING FOR GSECO: Bypass most validation requirements
        if self.provided_udc and "GSECO" in self.provided_udc.upper():
            # GSECO documents get special handling with minimal validation
            return self._quick_validate_gseco_document(extraction_log, document_id)

        extracted_text = extraction_log["extracted_text"]
        selection_marks = extraction_log.get("selection_marks", [])
        key_value_pairs = extraction_log.get("key_value_pairs", [])
        potential_initials = extraction_log.get("potential_initials", [])

        # CRITICAL: Initialize ALL variables that might be used in validation context strings
        # These must be defined early to avoid UnboundLocalError regardless of code path taken
        selected_marks = [
            mark for mark in selection_marks if mark.get("state") == "selected"
        ]
        unselected_marks = [
            mark for mark in selection_marks if mark.get("state") == "unselected"
        ]
        x_marks_found = []
        filled_initial_boxes = []
        has_any_initials = False

        # UDC is now provided as input parameter - no need to detect from OCR

        # CRITICAL: Document Integrity Check - Run BEFORE any validation
        # This catches corrupted OCR, interleaved pages, and garbled text
        # TWO-LAYER: Text heuristics + GPT-4o Vision (always runs for maximum accuracy)
        self.logger.info("Running document integrity check...")
        integrity_checker = DocumentIntegrityChecker(
            min_confidence=0.7,
            gpt4o_verification_integration=self.gpt4o_verification_integration,
        )
        integrity_result = integrity_checker.check_document_integrity(
            extracted_text,
            ocr_result=None,  # Can be passed if available
            pdf_path=pdf_path,  # Enable GPT-4o Vision verification
        )

        # Store integrity check results in extraction_log for analysis
        extraction_log["integrity_check"] = integrity_result

        # Build integrity rejection reasons but DON'T return early - continue validation
        integrity_rejection_reasons = []
        if not integrity_result["is_valid"]:
            self.logger.warning(
                f"Document integrity check FAILED: {integrity_result['summary']}"
            )
            self.logger.warning(
                f"Critical issues found: {integrity_result['critical_count']}"
            )

            # Build detailed rejection reasons from integrity issues
            for issue in integrity_result["issues"]:
                if issue["severity"] == "CRITICAL":
                    integrity_rejection_reasons.append(
                        f"Document Integrity Issue - {issue['category']}: {issue['description']}"
                    )

            # Store these to be combined with other rejection reasons later
            extraction_log["integrity_rejection_reasons"] = integrity_rejection_reasons

            # Add prominent notice to extracted text about integrity issues
            extracted_text += "\n\n" + "=" * 80 + "\n"
            extracted_text += "CRITICAL: DOCUMENT INTEGRITY ISSUES DETECTED\n"
            extracted_text += "=" * 80 + "\n"
            for reason in integrity_rejection_reasons:
                extracted_text += f"- {reason}\n"
            extracted_text += "\nThese issues MUST be included in rejection reasons.\n"
            extracted_text += "Continue validation to find additional issues.\n"
            extracted_text += "=" * 80 + "\n\n"
        else:
            self.logger.info(
                f"Document integrity check PASSED: Confidence {integrity_result['confidence']}"
            )

        # If extraction failed, create a basic validation result
        if not extraction_log["extraction_success"]:
            return {
                "document_id": document_id,
                "validation_status": "REJECT",
                "ocr_success": False,
                "ocr_error": extraction_log["error_details"],
                "extracted_text_length": 0,
                "extracted_text": extraction_log["extracted_text"],
                "validation_results": [
                    {
                        "category": "OCR_FAILURE",
                        "status": "FAIL",
                        "details": f"Layout analysis failed: {extraction_log['error_details']}",
                        "rejection_reason": "Document could not be processed due to layout analysis failure",
                        "relevant_text": "N/A - Layout analysis failed",
                        "text_evidence": "No text could be extracted from the document",
                    }
                ],
                "all_rejection_reasons": [
                    f"Layout analysis failed: {extraction_log['error_details']}"
                ],
                "processing_timestamp": datetime.now().isoformat(),
                "gpt_response_raw": "N/A - Layout analysis failed",
                "gpt_parsing_error": None,
            }

        # Check for broker signatures EARLY (before any validation that uses this variable)
        # This must run before FirstEnergy validation which checks broker_signature_found
        broker_signature_patterns = [
            r"on behalf of",
            r"for and on behalf of",
            # r'as agent for', New England Uses agent for non broker representative
            r"authorized agent",
            r"energy consultant",
            r"consultant",
            r"broker",
            r"utilities group",
            r"energy group",
            r"power group",
        ]

        # Check for New England specific authorized agent terms that are valid (not broker signatures)
        ne_agent_terms = [
            r"agent for customer",
            r"agent for the customer",
            r"customer\'s agent",
            r"customer representative",
            r"authorized representative",
            r"duly authorized",
            r"authorized to execute",
        ]

        # Remove audit trail sections from text before checking for broker patterns
        # Audit trail sections typically contain metadata about document processing
        audit_trail_patterns = [
            r"Audit trail.*?(?=\n\n|\Z)",
            r"Document History.*?(?=\n\n|\Z)",
            r"Sent for signature.*?(?=\n|\Z)",
            r"Viewed by.*?(?=\n|\Z)",
            r"Signed by.*?(?=\n|\Z)",
            r"The document has been completed.*?(?=\n|\Z)",
            r"Powered by.*?(?=\n|\Z)",
            r"Dropbox Sign.*?(?=\n|\Z)",
            r"from\s+[^\s]+@[^\s]+.*?(?=\n|\Z)",
        ]

        # Create a copy of extracted text without audit trail sections
        text_without_audit_trail = extracted_text
        for pattern in audit_trail_patterns:
            text_without_audit_trail = re.sub(
                pattern, "", text_without_audit_trail, flags=re.IGNORECASE | re.DOTALL
            )

        broker_signature_found = any(
            re.search(pattern, text_without_audit_trail, re.IGNORECASE)
            for pattern in broker_signature_patterns
        )

        # Check for authorized person patterns
        authorized_person_patterns = [
            r"authorized person",
            r"authorized representative",
            r"authorized signatory",
            r"Authorized Person/Title:",
        ]

        authorized_person_found = any(
            re.search(pattern, extracted_text, re.IGNORECASE)
            for pattern in authorized_person_patterns
        )

        # Fallback Scenario 0: FirstEnergy Interval Data Granularity Detection (FirstEnergy UDCs Only)
        # CRITICAL: FirstEnergy documents often have interval granularity text (e.g., "IDR, Train/cap, summary, interval")
        # in unusual positions that OCR misses - use GPT-4o Vision to reliably detect this text
        is_firstenergy_udc = self.provided_udc and self.provided_udc.upper() in [
            "CEI",
            "OE",
            "TE",
        ]

        if is_firstenergy_udc and pdf_path:
            try:
                self.logger.info(
                    f"FirstEnergy UDC detected ({self.provided_udc}) - Running GPT-4o interval granularity verification..."
                )
                fe_result = self.gpt4o_verification_integration.verify_firstenergy_interval_granularity_with_gpt4o(
                    pdf_path, extraction_log
                )

                # Update extraction_log with results
                extraction_log = fe_result

                # Add prominent notice to extracted text if granularity was found
                if extraction_log.get("firstenergy_interval_granularity", {}).get(
                    "text_found"
                ):
                    granularity_text = extraction_log[
                        "firstenergy_interval_granularity"
                    ].get("extracted_text", "interval data specifications")
                    extracted_text += f"\n\nGPT-4O FIRSTENERGY INTERVAL GRANULARITY DETECTION: Found text '{granularity_text}' specifying data granularity.\n"
                    self.logger.info(
                        f"GPT-4o detected FirstEnergy interval granularity: {granularity_text}"
                    )
                else:
                    # CRITICAL: If no granularity text found, this is a validation failure
                    extracted_text += "\n\nGPT-4O FIRSTENERGY INTERVAL GRANULARITY DETECTION: No interval granularity specifications found.\n"
                    extracted_text += "REJECTION REQUIRED: FirstEnergy LOAs must specify interval data granularity (e.g., 'interval', 'summary', 'IDR').\n"
                    self.logger.warning(
                        "GPT-4o did not find interval granularity text in FirstEnergy document - will add to rejection reasons"
                    )

                    # Store this as a FirstEnergy validation issue to be added later
                    if "firstenergy_granularity_missing" not in extraction_log:
                        extraction_log["firstenergy_granularity_missing"] = True

            except Exception as e:
                error_msg = f"GPT-4o FirstEnergy interval granularity verification failed: {str(e)}"
                self.logger.error(error_msg)
                extracted_text += f"\n\nGPT-4O FIRSTENERGY INTERVAL GRANULARITY DETECTION ERROR: {str(e)}\n"

        # Fallback Scenario 0b: AEP Interval Data Granularity Detection (AEP UDCs Only)
        # CRITICAL: AEP documents (like FirstEnergy) have interval granularity text in unusual positions
        is_aep_udc = self.provided_udc and self.provided_udc.upper() in [
            "CSPC",
            "OPC",
            "AEP",
        ]

        if is_aep_udc and pdf_path:
            try:
                self.logger.info(
                    f"AEP UDC detected ({self.provided_udc}) - Running GPT-4o interval granularity verification..."
                )
                aep_result = self.gpt4o_verification_integration.verify_aep_interval_granularity_with_gpt4o(
                    pdf_path, extraction_log
                )

                # Update extraction_log with results
                extraction_log = aep_result

                # Add prominent notice to extracted text if granularity was found
                if extraction_log.get("aep_interval_granularity", {}).get("text_found"):
                    granularity_text = extraction_log["aep_interval_granularity"].get(
                        "extracted_text", "interval data specifications"
                    )
                    extracted_text += f"\n\nGPT-4O AEP INTERVAL GRANULARITY DETECTION: Found text '{granularity_text}' specifying data granularity.\n"
                    self.logger.info(
                        f"GPT-4o detected AEP interval granularity: {granularity_text}"
                    )
                else:
                    # CRITICAL: If no granularity text found, this is a validation failure
                    extracted_text += "\n\nGPT-4O AEP INTERVAL GRANULARITY DETECTION: No interval granularity specifications found.\n"
                    extracted_text += "REJECTION REQUIRED: AEP LOAs must specify interval data granularity (e.g., 'interval', 'summary', 'IDR').\n"
                    self.logger.warning(
                        "GPT-4o did not find interval granularity text in AEP document - will add to rejection reasons"
                    )

                    # Store this as an AEP validation issue to be added later
                    if "aep_granularity_missing" not in extraction_log:
                        extraction_log["aep_granularity_missing"] = True

            except Exception as e:
                error_msg = (
                    f"GPT-4o AEP interval granularity verification failed: {str(e)}"
                )
                self.logger.error(error_msg)
                extracted_text += (
                    f"\n\nGPT-4O AEP INTERVAL GRANULARITY DETECTION ERROR: {str(e)}\n"
                )

        # Fallback Scenario 1: GPT-4o Vision for Initial Box Detection (Great Lakes Region)
        # CRITICAL: Always use GPT-4o Vision for initial box and X mark detection
        # GPT-4o Vision is more accurate than Azure OCR regex patterns for detecting X marks vs actual initials
        gpt4o_fallback_used = False

        # Always run GPT-4o for Great Lakes region to get accurate initial box detection
        if self.region == "Great Lakes" and pdf_path:
            try:
                # Use GPT-4o Vision to detect initial boxes and X marks accurately
                gpt4o_result = (
                    self.gpt4o_ocr_integration.process_pdf_with_gpt4o_fallback(
                        pdf_path, extraction_log
                    )
                )
                if gpt4o_result.get("success"):
                    # Update extraction log with GPT-4o results
                    updated_extraction_log = gpt4o_result.get(
                        "extraction_log", extraction_log
                    )
                    extraction_log["selection_marks"] = updated_extraction_log.get(
                        "selection_marks", []
                    )
                    extraction_log["initial_boxes"] = updated_extraction_log.get(
                        "initial_boxes", []
                    )
                    extraction_log["potential_initials"] = updated_extraction_log.get(
                        "potential_initials", []
                    )

                    # Also populate potential_initials from initial_boxes if not already done
                    if (
                        not extraction_log["potential_initials"]
                        and extraction_log["initial_boxes"]
                    ):
                        for box in extraction_log["initial_boxes"]:
                            if box.get("is_filled", False) and box.get("text"):
                                extraction_log["potential_initials"].append(
                                    {
                                        "text": box["text"],
                                        "is_likely_initial": True,
                                        "context": box.get(
                                            "context", "GPT-4o detected initial box"
                                        ),
                                    }
                                )

                    # Update local variables
                    selection_marks = extraction_log["selection_marks"]
                    potential_initials = extraction_log["potential_initials"]
                    gpt4o_fallback_used = True
                    extracted_text += "\n\nGPT-4O VISION APPLIED FOR INITIAL BOX DETECTION: Analyzed document for accurate X mark detection."
            except Exception as e:
                extracted_text += f"\n\nGPT-4O VISION ERROR: {str(e)}"

        # Fallback Scenario 2: Critical Checkbox Verification (Great Lakes Region Only)
        # When Azure found selection marks but critical checkboxes appear unselected
        if (
            self.region == "Great Lakes"
            and not gpt4o_fallback_used
            and len(extraction_log["selection_marks"]) > 0
        ):

            need_verification = self.check_need_critical_checkbox_verification(
                extraction_log, extracted_text
            )
            if need_verification and pdf_path:
                try:
                    # Define critical keywords for verification
                    critical_keywords = [
                        "Interval Historical Energy Usage Data Release",
                        "Account/SDI Number Release",
                        "Historical Usage Data",
                    ]

                    # Use GPT-4o critical checkbox verification - returns updated extraction_log
                    updated_extraction_log = (
                        self.gpt4o_verification_integration.verify_critical_checkboxes(
                            pdf_path, extraction_log, critical_keywords
                        )
                    )

                    # The updated extraction_log contains the verification results directly
                    if "gpt4o_checkbox_verification" in updated_extraction_log:
                        # Update local extraction_log reference
                        extraction_log = updated_extraction_log

                        # Update local selection_marks variable for the rest of the function
                        selection_marks = extraction_log["selection_marks"]

                        # Add note to extracted_text about the verification
                        critical_checkboxes_found = len(
                            extraction_log.get("gpt4o_checkbox_verification", {}).get(
                                "critical_checkboxes_found", []
                            )
                        )
                        if critical_checkboxes_found > 0:
                            extracted_text += f"\n\nGPT-4O CRITICAL CHECKBOX VERIFICATION APPLIED: Verified {critical_checkboxes_found} critical checkboxes."
                except Exception as e:
                    extracted_text += (
                        f"\n\nGPT-4O CRITICAL CHECKBOX VERIFICATION ERROR: {str(e)}"
                    )

        # Fallback Scenario 3: New England Service Options Verification
        # CRITICAL: CLP/BECO/WMECO always use GPT-4o, others use it conditionally
        if self.region == "New England" and self.provided_udc and pdf_path:
            # Skip BHE - service options not required for BHE
            is_bhe = (
                "BHE" in self.provided_udc.upper() or self.provided_udc.upper() == "BHE"
            )

            if not is_bhe:
                # Define UDCs that always need GPT-4o verification
                always_verify_udcs = ["CLP", "BECO", "WMECO"]
                is_always_verify = any(
                    udc in self.provided_udc.upper() for udc in always_verify_udcs
                )

                # Determine if we need to run GPT-4o verification
                should_verify = False

                if is_always_verify:
                    # Always verify for CLP/BECO/WMECO - regex unreliable for these
                    should_verify = True
                else:
                    # For other NE utilities, only verify if there's a selection issue
                    service_options = extraction_log.get("service_options", {})
                    if (
                        service_options.get("detected")
                        and service_options.get("selection_count", 1) != 1
                    ):
                        should_verify = True

                if should_verify:
                    try:
                        self.logger.info(
                            f"{self.provided_udc} document detected - Running GPT-4o service options verification..."
                        )
                        ne_verification_result = self.gpt4o_verification_integration.verify_ne_service_options_with_gpt4o(
                            pdf_path, extraction_log
                        )
                        if ne_verification_result.get("success"):
                            # Update service options based on verification
                            if ne_verification_result.get("service_options_clarified"):
                                extraction_log["service_options"] = (
                                    ne_verification_result["service_options"]
                                )
                                extracted_text += f"\n\nGPT-4O {self.provided_udc} SERVICE OPTIONS VERIFICATION APPLIED: Verified service option selection."
                                self.logger.info(
                                    f"GPT-4o {self.provided_udc} verification complete"
                                )
                        else:
                            self.logger.warning(
                                f"GPT-4o {self.provided_udc} verification did not return success"
                            )
                    except Exception as e:
                        error_msg = f"GPT-4o {self.provided_udc} service options verification failed: {str(e)}"
                        self.logger.error(error_msg)
                        extracted_text += f"\n\nGPT-4O {self.provided_udc} SERVICE OPTIONS VERIFICATION ERROR: {str(e)}"

        # Fallback Scenario 4: MECO/NANT Subscription Options Verification (MECO/NANT UDC Only - 3 options)
        # CRITICAL: Always use GPT-4o for MECO/NANT - regex detection is unreliable
        if (
            self.region == "New England"
            and self.provided_udc
            and (
                "MECO" in self.provided_udc.upper()
                or "NANT" in self.provided_udc.upper()
            )
            and pdf_path
        ):
            # Always run GPT-4o for MECO/NANT, regardless of whether regex detected anything
            # This is because regex pattern matching is unreliable for checkbox detection
            try:
                self.logger.info(
                    f"{self.provided_udc} document detected - Running GPT-4o subscription options verification..."
                )
                extraction_log = self.gpt4o_verification_integration.verify_meco_subscription_options_with_gpt4o(
                    pdf_path, extraction_log
                )
                if extraction_log.get("meco_subscription_options", {}).get(
                    "gpt4o_verified"
                ):
                    selection_count = extraction_log["meco_subscription_options"][
                        "selection_count"
                    ]
                    extracted_text += f"\n\nGPT-4O MECO/NANT SUBSCRIPTION OPTIONS VERIFICATION APPLIED: Verified {selection_count} option(s) selected."
                    self.logger.info(
                        f"GPT-4o MECO/NANT verification complete: {selection_count} option(s) selected"
                    )
                else:
                    self.logger.warning(
                        "GPT-4o MECO/NANT verification did not return verified results"
                    )
            except Exception as e:
                error_msg = f"GPT-4o MECO/NANT subscription options verification failed: {str(e)}"
                self.logger.error(error_msg)
                extracted_text += f"\n\nGPT-4O MECO/NANT SUBSCRIPTION OPTIONS VERIFICATION ERROR: {str(e)}"

        # Fallback Scenario 5: NECO Subscription Options Verification (NECO UDC Only - 2 options)
        # CRITICAL: Always use GPT-4o for NECO - regex detection is unreliable
        if (
            self.region == "New England"
            and self.provided_udc
            and "NECO" in self.provided_udc.upper()
            and pdf_path
        ):
            # Always run GPT-4o for NECO, regardless of whether regex detected anything
            # This is because regex pattern matching is unreliable for checkbox detection
            try:
                self.logger.info(
                    "NECO document detected - Running GPT-4o subscription options verification..."
                )
                extraction_log = self.gpt4o_verification_integration.verify_neco_subscription_options_with_gpt4o(
                    pdf_path, extraction_log
                )
                if extraction_log.get("neco_subscription_options", {}).get(
                    "gpt4o_verified"
                ):
                    gpt4o_selection_count = extraction_log["neco_subscription_options"][
                        "selection_count"
                    ]

                    # CRITICAL: GPT-4o OCR now takes PRIORITY over regex-based detection
                    # GPT-4o vision is more accurate than regex patterns for checkbox detection
                    # Only use code-level detection as a fallback if GPT-4o results seem incorrect
                    code_level_two_weeks = bool(
                        re.search(
                            r":selected:.*?Two\s+Weeks\s+Online",
                            extracted_text,
                            re.IGNORECASE | re.DOTALL,
                        )
                    )
                    code_level_one_year = bool(
                        re.search(
                            r":selected:.*?One\s+Year\s+Online",
                            extracted_text,
                            re.IGNORECASE | re.DOTALL,
                        )
                    )
                    code_level_count = sum([code_level_two_weeks, code_level_one_year])

                    # If code-level and GPT-4o disagree, GPT-4o takes precedence (reversed priority)
                    if code_level_count != gpt4o_selection_count:
                        self.logger.warning(
                            f"NECO subscription count mismatch - Code-level: {code_level_count}, GPT-4o: {gpt4o_selection_count}. Using GPT-4o (vision is more accurate)."
                        )
                        # Keep GPT-4o result, log the mismatch for analysis
                        extraction_log["neco_subscription_options"][
                            "code_level_mismatch"
                        ] = True
                        extraction_log["neco_subscription_options"][
                            "code_level_count"
                        ] = code_level_count
                        extracted_text += f"\n\n**GPT-4O PRIORITY**: Using GPT-4o vision result ({gpt4o_selection_count}) over regex detection ({code_level_count})\n\n"

                    extracted_text += f"\n\nGPT-4O NECO SUBSCRIPTION OPTIONS VERIFICATION APPLIED: Verified {extraction_log['neco_subscription_options']['selection_count']} option(s) selected."
                    self.logger.info(
                        f"GPT-4o NECO verification complete: {extraction_log['neco_subscription_options']['selection_count']} option(s) selected"
                    )
                else:
                    self.logger.warning(
                        "GPT-4o NECO verification did not return verified results"
                    )
            except Exception as e:
                error_msg = (
                    f"GPT-4o NECO subscription options verification failed: {str(e)}"
                )
                self.logger.error(error_msg)
                extracted_text += (
                    f"\n\nGPT-4O NECO SUBSCRIPTION OPTIONS VERIFICATION ERROR: {str(e)}"
                )

            # NEW: Run code-level NECO field validations
            try:
                self.logger.info("Running code-level NECO field validations...")

                # Validate customer name field
                customer_name_issues = self.validate_neco_customer_name_field(
                    extracted_text, extraction_log
                )

                # Validate account numbers
                account_issues = self.validate_neco_account_numbers(
                    extracted_text, extraction_log
                )

                # Validate supplier information
                supplier_issues = self.validate_neco_supplier_fields(
                    extracted_text, extraction_log
                )

                # Validate NECO subscription options selection count
                subscription_issues = []
                if extraction_log.get("neco_subscription_options", {}).get(
                    "gpt4o_verified"
                ):
                    selection_count = extraction_log["neco_subscription_options"][
                        "selection_count"
                    ]
                    if selection_count == 0:
                        subscription_issues.append(
                            self.ERROR_MESSAGES["neco_subscription_none"]
                        )
                    elif selection_count > 1:
                        subscription_issues.append(
                            self.ERROR_MESSAGES["neco_subscription_multiple"]
                        )

                # Combine all NECO validation issues
                neco_validation_issues = (
                    customer_name_issues
                    + account_issues
                    + supplier_issues
                    + subscription_issues
                )

                if neco_validation_issues:
                    # Store for later injection into prompt
                    extraction_log["neco_code_level_validation_issues"] = (
                        neco_validation_issues
                    )
                    self.logger.info(
                        f"Code-level NECO validation found {len(neco_validation_issues)} issue(s)"
                    )

                    # Add prominent context about these issues
                    extracted_text += "\n\n" + "=" * 80 + "\n"
                    extracted_text += "CODE-LEVEL NECO FIELD VALIDATION RESULTS\n"
                    extracted_text += "=" * 80 + "\n"
                    extracted_text += (
                        "The following REQUIRED fields were checked at code-level:\n\n"
                    )

                    for i, issue in enumerate(neco_validation_issues, 1):
                        extracted_text += f"{i}. {issue}\n"

                    extracted_text += "\n**CRITICAL INSTRUCTION:**\n"
                    extracted_text += (
                        "These validation issues were detected by code-level checks.\n"
                    )
                    extracted_text += "You MUST include ALL of these issues in your rejectionReasons.\n"
                    extracted_text += (
                        "DO NOT skip or ignore any of these pre-validated issues.\n"
                    )
                    extracted_text += "=" * 80 + "\n\n"
                else:
                    self.logger.info(
                        "Code-level NECO validation passed - all required fields present"
                    )

            except Exception as e:
                self.logger.error(f"Code-level NECO validation error: {str(e)}")

        # NEW: Run code-level COMED field validations (Great Lakes Region - Illinois)
        # COMED LOAs have flexible formats so we check for required fields anywhere in the document
        # Run code-level Illinois field validations (ComEd and Ameren use same rules)
        provided_udc_upper = self.provided_udc.upper() if self.provided_udc else ""
        illinois_udcs = ["COMMED", "AMEREN", "CILCO", "CIPS", "IP"]

        if provided_udc_upper and any(
            udc in provided_udc_upper for udc in illinois_udcs
        ):
            try:
                utility_name = (
                    "ComEd/Ameren"
                    if any(
                        x in provided_udc_upper
                        for x in ["AMEREN", "CILCO", "CIPS", "IP"]
                    )
                    else "ComEd"
                )
                self.logger.info(
                    f"Running code-level Illinois field validations for {utility_name}..."
                )
                # First run code-level validation
                comed_validation_issues = self.validate_comed_required_fields(
                    extracted_text, extraction_log
                )

                # CRITICAL: Use GPT-4o Vision to verify actual signature presence (not just field labels)
                if pdf_path:
                    try:
                        self.logger.info(
                            "COMED document detected - Running GPT-4o signature verification..."
                        )
                        signature_result = self.gpt4o_verification_integration.extract_signatures_with_gpt4o(
                            pdf_path
                        )

                        if signature_result.get("success"):
                            # Store signature detection results
                            extraction_log["comed_signature_detection"] = {
                                "customer_signature_present": signature_result.get(
                                    "customer_signature_present", False
                                ),
                                "customer_signature_text": signature_result.get(
                                    "customer_signature_text"
                                ),
                                "requestor_signature_present": signature_result.get(
                                    "requestor_signature_present", False
                                ),
                                "requestor_signature_text": signature_result.get(
                                    "requestor_signature_text"
                                ),
                                "gpt4o_verified": True,
                            }

                            # Update the signature_found status based on GPT-4o detection
                            if "comed_validation" not in extraction_log:
                                extraction_log["comed_validation"] = {}

                            extraction_log["comed_validation"]["signature_found"] = (
                                signature_result.get(
                                    "customer_signature_present", False
                                )
                            )
                            extraction_log["comed_validation"]["signature_text"] = (
                                signature_result.get("customer_signature_text")
                            )
                            extraction_log["comed_validation"][
                                "signature_verification_method"
                            ] = "gpt4o_vision"

                            self.logger.info("GPT-4o COMED signature detection:")
                            self.logger.info(
                                f"  - Customer signature present: {signature_result.get('customer_signature_present')}"
                            )
                            self.logger.info(
                                f"  - Customer signature text: '{signature_result.get('customer_signature_text')}'"
                            )
                        else:
                            self.logger.warning(
                                "GPT-4o COMED signature verification did not return success"
                            )
                    except Exception as e:
                        self.logger.error(
                            f"GPT-4o COMED signature verification error: {str(e)}"
                        )

                # CRITICAL: Always use GPT-4o for COMED - NO CODE-LEVEL FALLBACK
                # GPT-4o MUST succeed or validation returns ERROR status
                if pdf_path:
                    try:
                        self.logger.info(
                            "COMED document detected - Running GPT-4o comprehensive field verification..."
                        )
                        gpt4o_result = self.gpt4o_verification_integration.verify_comed_required_fields_with_gpt4o(
                            pdf_path, extraction_log
                        )

                        if gpt4o_result.get("success"):
                            # Update extraction_log with GPT-4o results
                            extraction_log = gpt4o_result.get(
                                "extraction_log", extraction_log
                            )

                            # Re-run validation with GPT-4o extracted fields
                            comed_validation_issues = []
                            comed_data = extraction_log.get("comed_validation", {})

                            # Check each required field
                            if not comed_data.get("customer_name_found"):
                                comed_validation_issues.append(
                                    self.ERROR_MESSAGES["comed_customer_name_missing"]
                                )
                            if not comed_data.get("customer_address_found"):
                                comed_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "comed_customer_address_missing"
                                    ]
                                )
                            if not comed_data.get("authorized_person_found"):
                                comed_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "comed_authorized_person_missing"
                                    ]
                                )
                            if not comed_data.get("authorized_person_title_found"):
                                comed_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "comed_authorized_person_title_missing"
                                    ]
                                )
                            if not comed_data.get("signature_found"):
                                comed_validation_issues.append(
                                    self.ERROR_MESSAGES["comed_signature_missing"]
                                )
                            if not comed_data.get("signature_date_found"):
                                comed_validation_issues.append(
                                    self.ERROR_MESSAGES["comed_signature_date_missing"]
                                )
                            if not comed_data.get("account_numbers_found"):
                                comed_validation_issues.append(
                                    self.ERROR_MESSAGES["comed_account_numbers_missing"]
                                )
                            if not comed_data.get("interval_authorization_found"):
                                comed_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "comed_interval_authorization_missing"
                                    ]
                                )
                            if not comed_data.get("supplier_info_found"):
                                comed_validation_issues.append(
                                    self.ERROR_MESSAGES["comed_supplier_info_missing"]
                                )

                            # Check Illinois authorization with interval data
                            if not comed_data.get(
                                "illinois_authorization_found"
                            ) or not comed_data.get("interval_data_in_auth"):
                                comed_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "comed_illinois_authorization_missing"
                                    ]
                                )

                            # Check agent checkbox (conditional)
                            if comed_data.get(
                                "agent_auth_section_found"
                            ) and not comed_data.get("agent_checkbox_marked"):
                                comed_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "comed_agent_checkbox_not_marked"
                                    ]
                                )

                            # Check ComEd utility mention (use code-level detection since GPT-4o doesn't check this)
                            if not comed_data.get("comed_utility_mentioned"):
                                comed_validation_issues.append(
                                    self.ERROR_MESSAGES["comed_utility_not_mentioned"]
                                )

                            self.logger.info(
                                f"GPT-4o COMED verification complete - {len(comed_validation_issues)} issues found"
                            )
                        else:
                            # CRITICAL: GPT-4o failed - return ERROR status immediately
                            self.logger.error(
                                "GPT-4o COMED verification failed - returning ERROR status"
                            )
                            return {
                                "document_id": document_id,
                                "fileName": document_id,
                                "validation_status": "ERROR",
                                "status": "ERROR",
                                "rejectionReasons": [
                                    "GPT-4o vision verification failed - cannot validate COMED document without vision analysis"
                                ],
                                "all_rejection_reasons": [
                                    "GPT-4o vision verification failed"
                                ],
                                "expiration_date": "N/A",
                                "ocr_success": extraction_log["extraction_success"],
                                "extracted_text_length": len(extracted_text),
                                "processing_timestamp": datetime.now().isoformat(),
                                "error": "GPT-4o comprehensive field verification did not return success",
                                "gpt4o_failure": True,
                            }
                    except Exception as e:
                        # CRITICAL: GPT-4o exception - return ERROR status immediately
                        self.logger.error(
                            f"GPT-4o COMED field verification exception: {str(e)}"
                        )
                        return {
                            "document_id": document_id,
                            "fileName": document_id,
                            "validation_status": "ERROR",
                            "status": "ERROR",
                            "rejectionReasons": [
                                f"GPT-4o vision verification exception: {str(e)}"
                            ],
                            "all_rejection_reasons": [
                                f"GPT-4o vision verification failed: {str(e)}"
                            ],
                            "expiration_date": "N/A",
                            "ocr_success": extraction_log["extraction_success"],
                            "extracted_text_length": len(extracted_text),
                            "processing_timestamp": datetime.now().isoformat(),
                            "error": f"GPT-4o comprehensive field verification exception: {str(e)}",
                            "gpt4o_failure": True,
                            "exception_details": str(e),
                        }

                if comed_validation_issues:
                    # Store for later injection into prompt
                    extraction_log["comed_code_level_validation_issues"] = (
                        comed_validation_issues
                    )
                    self.logger.info(
                        f"Code-level COMED validation found {len(comed_validation_issues)} issue(s)"
                    )

                    # Add prominent context about these issues
                    extracted_text += "\n\n" + "=" * 80 + "\n"
                    extracted_text += "CODE-LEVEL COMED FIELD VALIDATION RESULTS\n"
                    extracted_text += "=" * 80 + "\n"
                    extracted_text += (
                        "COMED (ComEd/Commonwealth Edison) - Illinois Utility\n"
                    )
                    extracted_text += (
                        "IMPORTANT: COMED LOAs do NOT have a fixed form format.\n"
                    )
                    extracted_text += "Different structures/formats are acceptable as long as required fields are present.\n"
                    extracted_text += "\n"
                    extracted_text += (
                        "The following REQUIRED fields were checked at code-level:\n\n"
                    )

                    for i, issue in enumerate(comed_validation_issues, 1):
                        extracted_text += f"{i}. {issue}\n"

                    extracted_text += "\n**CRITICAL INSTRUCTION:**\n"
                    extracted_text += (
                        "These validation issues were detected by code-level checks.\n"
                    )
                    extracted_text += "You MUST include ALL of these issues in your rejectionReasons.\n"
                    extracted_text += (
                        "DO NOT skip or ignore any of these pre-validated issues.\n"
                    )
                    extracted_text += "=" * 80 + "\n\n"
                else:
                    self.logger.info(
                        "Code-level COMED validation passed - all required fields present"
                    )

                    # Add success context
                    extracted_text += "\n\n" + "=" * 80 + "\n"
                    extracted_text += "CODE-LEVEL COMED FIELD VALIDATION RESULTS\n"
                    extracted_text += "=" * 80 + "\n"
                    extracted_text += "✓ ALL REQUIRED COMED FIELDS PRESENT:\n"
                    comed_data = extraction_log.get("comed_validation", {})
                    if comed_data.get("customer_name_found"):
                        extracted_text += f"  ✓ Customer Name: {comed_data.get('customer_name', 'Found')}\n"
                    if comed_data.get("customer_address_found"):
                        extracted_text += f"  ✓ Customer Address: {comed_data.get('customer_address', 'Found')}\n"
                    if comed_data.get("authorized_person_found"):
                        extracted_text += f"  ✓ Authorized Person: {comed_data.get('authorized_person', 'Found')}\n"
                    if comed_data.get("authorized_person_title_found"):
                        extracted_text += f"  ✓ Authorized Person Title: {comed_data.get('authorized_person_title', 'Found')}\n"
                    if comed_data.get("signature_found"):
                        extracted_text += "  ✓ Signature: Present\n"
                    if comed_data.get("signature_date_found"):
                        extracted_text += f"  ✓ Signature Date: {comed_data.get('signature_date', 'Found')}\n"
                    if comed_data.get("account_numbers_found"):
                        account_count = comed_data.get("account_count", 0)
                        has_attachment = comed_data.get(
                            "has_attachment_indicator", False
                        )
                        extracted_text += (
                            f"  ✓ Account Numbers: {account_count} found"
                            + (" + attachment indicated" if has_attachment else "")
                            + "\n"
                        )
                    if comed_data.get("interval_authorization_found"):
                        extracted_text += "  ✓ Interval Data Authorization: Present\n"
                    if comed_data.get("supplier_info_found"):
                        extracted_text += (
                            "  ✓ Supplier (Constellation) Information: Present\n"
                        )
                    extracted_text += "\n"
                    extracted_text += "COMED validation passed - document contains all required fields.\n"
                    extracted_text += "=" * 80 + "\n\n"

            except Exception as e:
                self.logger.error(f"Code-level COMED validation error: {str(e)}")
                extracted_text += f"\n\nCODE-LEVEL COMED VALIDATION ERROR: {str(e)}\n"

        # NEW: Run code-level CINERGY/DUKE ENERGY field validations (Great Lakes Region - Ohio)
        # CINERGY (Duke Energy Ohio) has specific account format requirements and signature validity rules
        if self.provided_udc and (
            "CINERGY" in self.provided_udc.upper()
            or "DUKE" in self.provided_udc.upper()
        ):
            try:
                self.logger.info(
                    "Running code-level CINERGY/DUKE ENERGY field validations..."
                )

                # Run code-level validation for Cinergy-specific fields
                cinergy_validation_issues = self.validate_cinergy_required_fields(
                    extracted_text, extraction_log
                )

                if cinergy_validation_issues:
                    # Store for later injection into prompt
                    extraction_log["cinergy_code_level_validation_issues"] = (
                        cinergy_validation_issues
                    )
                    self.logger.info(
                        f"Code-level CINERGY validation found {len(cinergy_validation_issues)} issue(s)"
                    )

                    # Add prominent context about these issues (optimized with single join)
                    issues_list = [
                        f"{i}. {issue}"
                        for i, issue in enumerate(cinergy_validation_issues, 1)
                    ]
                    extracted_text += "\n".join(
                        [
                            "\n\n" + "=" * 80,
                            "CODE-LEVEL CINERGY/DUKE ENERGY FIELD VALIDATION RESULTS",
                            "=" * 80,
                            "CINERGY (Duke Energy Ohio) - Ohio Utility",
                            "IMPORTANT: CINERGY LOAs have strict Ohio-specific requirements:",
                            "  - Account number: 22 digits, starts with '910', 'Z' at position 13",
                            "  - Signature validity: 1 year (12 months) for Ohio",
                            "",
                            "The following validation checks were performed:\n",
                            "\n".join(issues_list),
                            "\n**CRITICAL INSTRUCTION:**",
                            "These validation issues were detected by code-level checks.",
                            "You MUST include ALL of these issues in your rejectionReasons.",
                            "DO NOT skip or ignore any of these pre-validated issues.",
                            "=" * 80 + "\n\n",
                        ]
                    )
                else:
                    self.logger.info(
                        "Code-level CINERGY validation passed - all required fields valid"
                    )

                    # Add success context (optimized with single join)
                    cinergy_data = extraction_log.get("cinergy_validation", {})
                    field_details = []
                    if cinergy_data.get("account_format_valid"):
                        field_details.append(
                            "  ✓ Account Number Format: Valid (22 digits, '910' prefix, 'Z' at position 13)"
                        )
                        field_details.append(
                            f"    Account: {cinergy_data.get('account_number', 'Found')}"
                        )
                    if cinergy_data.get("signature_date_valid"):
                        field_details.append(
                            "  ✓ Signature Date: Valid (within 1 year for Ohio)"
                        )
                        field_details.append(
                            f"    Date: {cinergy_data.get('signature_date', 'Found')}"
                        )

                    extracted_text += "\n".join(
                        [
                            "\n\n" + "=" * 80,
                            "CODE-LEVEL CINERGY/DUKE ENERGY FIELD VALIDATION RESULTS",
                            "=" * 80,
                            "✓ ALL CINERGY REQUIREMENTS MET:",
                            "\n".join(field_details),
                            "",
                            "CINERGY validation passed - document meets Ohio-specific requirements.",
                            "=" * 80 + "\n\n",
                        ]
                    )

            except Exception as e:
                self.logger.error(f"Code-level CINERGY validation error: {str(e)}")
                extracted_text += f"\n\nCODE-LEVEL CINERGY VALIDATION ERROR: {str(e)}\n"

        # NEW: Run CINERGY initial box validation via GPT-4o (Great Lakes Region - Ohio)
        # CINERGY/Duke Energy Ohio has TWO initial boxes that must be filled with letter initials (same as AEP)
        extraction_log, extracted_text = self._validate_initial_boxes(
            pdf_path,
            extraction_log,
            extracted_text,
            udc_name="CINERGY",
            keywords=["CINERGY", "DUKE"],
            log_key="cinergy_validation",
            verify_method_name="verify_cinergy_initial_boxes_with_gpt4o",
        )

        # NEW: Run code-level Dayton validation (Great Lakes Region - Ohio)
        # DAYTON (Dayton Power & Light) has specific Ohio phrase utility requirements
        if self.provided_udc and "DAYTON" in self.provided_udc.upper():
            try:
                self.logger.info("Running code-level DAYTON field validations...")

                # Run code-level validation for Dayton-specific fields
                dayton_validation_issues = self.validate_dayton_required_fields(
                    extracted_text, extraction_log
                )

                if dayton_validation_issues:
                    # Store for later injection into prompt
                    extraction_log["dayton_code_level_validation_issues"] = (
                        dayton_validation_issues
                    )
                    self.logger.info(
                        f"Code-level DAYTON validation found {len(dayton_validation_issues)} issue(s)"
                    )

                    # Add prominent context about these issues
                    issues_list = [
                        f"{i}. {issue}"
                        for i, issue in enumerate(dayton_validation_issues, 1)
                    ]
                    extracted_text += "\n".join(
                        [
                            "\n\n" + "=" * 80,
                            "CODE-LEVEL DAYTON POWER & LIGHT FIELD VALIDATION RESULTS",
                            "=" * 80,
                            "DAYTON (Dayton Power & Light) - Ohio Utility",
                            "IMPORTANT: Dayton LOAs have Ohio-specific requirements:",
                            "  - Ohio authorization statement must reference DAYTON, DP&L, or Dayton Power & Light",
                            "",
                            "The following validation checks were performed:\n",
                            "\n".join(issues_list),
                            "\n**CRITICAL INSTRUCTION:**",
                            "These validation issues were detected by code-level checks.",
                            "You MUST include ALL of these issues in your rejectionReasons.",
                            "DO NOT skip or ignore any of these pre-validated issues.",
                            "=" * 80 + "\n\n",
                        ]
                    )
                else:
                    self.logger.info(
                        "Code-level DAYTON validation passed - Ohio phrase utility validation passed"
                    )

            except Exception as e:
                self.logger.error(f"Code-level DAYTON validation error: {str(e)}")
                extracted_text += f"\n\nCODE-LEVEL DAYTON VALIDATION ERROR: {str(e)}\n"

        # NEW: Run Dayton initial box validation via GPT-4o (Great Lakes Region - Ohio)
        # Dayton Power & Light has TWO initial boxes that must be filled with letter initials (same as AEP)
        extraction_log, extracted_text = self._validate_initial_boxes(
            pdf_path,
            extraction_log,
            extracted_text,
            udc_name="DAYTON",
            keywords=["DAYTON"],
            log_key="dayton_validation",
            verify_method_name="verify_dayton_initial_boxes_with_gpt4o",
        )

        # NEW: Run Dayton multi-page account number scan
        # ALWAYS scan all pages for Dayton account numbers (format: 11-13 digits + Z + 9-11 digits)
        if self.provided_udc and "DAYTON" in self.provided_udc.upper() and pdf_path:
            try:
                self.logger.info(
                    "DAYTON document detected - Scanning ALL pages for account numbers..."
                )
                extraction_log = self.gpt4o_verification_integration.scan_all_pages_for_dayton_accounts_with_gpt4o(
                    pdf_path, extraction_log
                )

                # Check results and add to extracted text
                dayton_data = extraction_log.get("dayton_validation", {})

                if dayton_data.get("account_numbers_found"):
                    account_count = dayton_data.get("account_count", 0)
                    extracted_text += f"\n\nDAYTON MULTI-PAGE ACCOUNT SCAN: Found {account_count} valid account(s)\n"
                    self.logger.info(
                        f"Dayton multi-page scan found {account_count} valid account(s)"
                    )
                else:
                    # No valid accounts found - this will be added as a rejection reason
                    extracted_text += "\n\nDAYTON MULTI-PAGE ACCOUNT SCAN: No valid Dayton account numbers found\n"
                    extracted_text += "REJECTION REQUIRED: Dayton LOAs must include valid account numbers (format: 11-13 digits + Z + 9-11 digits)\n"
                    self.logger.warning(
                        "Dayton multi-page scan found NO valid account numbers - will add to rejection reasons"
                    )

                    # Store as validation issue
                    if "dayton_code_level_validation_issues" not in extraction_log:
                        extraction_log["dayton_code_level_validation_issues"] = []
                    extraction_log["dayton_code_level_validation_issues"].append(
                        "DAYTON: No valid account numbers found in document (format required: 11-13 digits + Z + 9-11 digits, e.g., 123456789012Z1234567890)"
                    )

            except Exception as e:
                self.logger.error(f"Dayton multi-page account scan error: {str(e)}")
                extracted_text += f"\n\nDAYTON ACCOUNT SCAN ERROR: {str(e)}\n"

        # NEW: Run comprehensive AEP validation (Great Lakes Region - Ohio)
        # THREE-LAYER APPROACH (same as FirstEnergy/ComEd)
        if self.provided_udc and self.provided_udc.upper() in ["CSPC", "OPC", "AEP"]:
            try:
                self.logger.info(
                    f"AEP document detected ({self.provided_udc}) - Running comprehensive three-layer validation..."
                )

                # LAYER 1: Code-level structural validation
                try:
                    self.logger.info(
                        "Layer 1: Running code-level AEP structural validation..."
                    )
                    structural_issues = self.validate_aep_required_fields(
                        extracted_text, extraction_log
                    )
                    extraction_log["aep_structural_validation_issues"] = (
                        structural_issues
                    )
                    self.logger.info(
                        f"Layer 1 complete: {len(structural_issues)} structural issue(s) found"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Layer 1 AEP structural validation error: {str(e)}"
                    )

                # LAYER 2: GPT-4o Vision extraction
                if pdf_path:
                    try:
                        self.logger.info(
                            "Layer 2: Running GPT-4o Vision field extraction..."
                        )
                        gpt4o_result = self.gpt4o_verification_integration.verify_aep_comprehensive_with_gpt4o(
                            pdf_path, extraction_log
                        )

                        if gpt4o_result.get("success"):
                            extraction_log = gpt4o_result.get(
                                "extraction_log", extraction_log
                            )
                            self.logger.info(
                                "Layer 2 complete: GPT-4o successfully extracted all fields"
                            )

                            # LAYER 3: Code validation
                            self.logger.info(
                                "Layer 3: Validating GPT-4o extracted fields..."
                            )
                            aep_validation_issues = []
                            aep_data = extraction_log.get("aep_validation", {})

                            # CRES Provider fields (MUST be code-level enforced)
                            if not aep_data.get("cres_name_found"):
                                aep_validation_issues.append(
                                    self.ERROR_MESSAGES["aep_cres_name_missing"]
                                )
                            if not aep_data.get("cres_address_found"):
                                aep_validation_issues.append(
                                    self.ERROR_MESSAGES["aep_cres_address_missing"]
                                )
                            if not aep_data.get("cres_phone_found"):
                                aep_validation_issues.append(
                                    self.ERROR_MESSAGES["aep_cres_phone_missing"]
                                )
                            if not aep_data.get("cres_email_found"):
                                aep_validation_issues.append(
                                    self.ERROR_MESSAGES["aep_cres_email_missing"]
                                )

                            # Customer fields
                            if not aep_data.get("customer_name_found"):
                                aep_validation_issues.append(
                                    self.ERROR_MESSAGES["aep_customer_name_missing"]
                                )
                            if not aep_data.get("customer_address_found"):
                                aep_validation_issues.append(
                                    self.ERROR_MESSAGES["aep_customer_address_missing"]
                                )
                            if not aep_data.get("authorized_person_title_found"):
                                aep_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "aep_authorized_person_title_missing"
                                    ]
                                )

                            # Ohio Statement fields
                            if not aep_data.get("ohio_signature_found"):
                                aep_validation_issues.append(
                                    self.ERROR_MESSAGES["aep_ohio_signature_missing"]
                                )
                            if not aep_data.get("ohio_date_found"):
                                aep_validation_issues.append(
                                    self.ERROR_MESSAGES["aep_ohio_date_missing"]
                                )

                            # CRITICAL: Initial Box validation for AEP LOAs (matches FirstEnergy logic)
                            initial_boxes = aep_data.get("initial_boxes", {})
                            filled_box_count = initial_boxes.get("filled_box_count", 0)
                            empty_box_count = initial_boxes.get("empty_box_count", 0)
                            x_mark_count = initial_boxes.get("x_mark_count", 0)

                            # Check for X marks in initial boxes
                            if x_mark_count > 0:
                                # Reject for X marks
                                aep_validation_issues.append(
                                    "AEP: Unclear or ambiguous initials or x mark in initial boxes"
                                )
                                self.logger.info(
                                    f"Added rejection for {x_mark_count} X mark(s) in initial boxes"
                                )

                            # CRITICAL: Account number extraction using Azure OCR
                            # Azure OCR is more accurate, faster, and doesn't hallucinate
                            try:
                                self.logger.info(
                                    "AEP: Extracting account numbers from Azure OCR (all pages)..."
                                )

                                # Use Azure OCR for account extraction
                                azure_account_result = self.gpt4o_verification_integration.extract_aep_accounts_from_azure_ocr(
                                    extraction_log
                                )

                                # Update aep_data with Azure OCR results
                                aep_data = extraction_log.get("aep_validation", {})

                                if azure_account_result.get("success"):
                                    aep_data["account_numbers"] = azure_account_result[
                                        "valid_accounts"
                                    ]
                                    aep_data["account_count"] = azure_account_result[
                                        "account_count"
                                    ]
                                    aep_data["account_numbers_found"] = (
                                        azure_account_result["account_count"] > 0
                                    )
                                    aep_data["invalid_length_accounts"] = (
                                        azure_account_result["invalid_accounts"]
                                    )
                                    aep_data["account_length_valid"] = (
                                        azure_account_result["format_validation_passed"]
                                    )
                                    aep_data["extraction_method"] = (
                                        "azure_document_intelligence"
                                    )

                                    self.logger.info(
                                        f"Azure OCR found {aep_data['account_count']} valid account(s)"
                                    )

                                    if azure_account_result["invalid_accounts"]:
                                        self.logger.warning(
                                            f"Found {len(azure_account_result['invalid_accounts'])} invalid account(s): "
                                            f"{azure_account_result['invalid_accounts']}"
                                        )
                                else:
                                    # No accounts found
                                    aep_data["account_field_empty"] = True
                                    aep_data["account_numbers_found"] = False
                                    aep_data["extraction_method"] = (
                                        "azure_document_intelligence"
                                    )
                                    self.logger.warning(
                                        "Azure OCR found NO account numbers - marking for rejection"
                                    )

                            except Exception as e:
                                self.logger.error(
                                    f"Azure OCR account extraction error for AEP: {str(e)}"
                                )
                                # Fallback: mark as empty
                                aep_data["account_field_empty"] = True
                                aep_data["account_numbers_found"] = False

                            # Account validation
                            # Check if accounts were found and validate format
                            if aep_data.get("account_field_empty") is True:
                                aep_validation_issues.append(
                                    self.ERROR_MESSAGES["aep_account_numbers_missing"]
                                )
                            elif not aep_data.get("account_length_valid"):
                                # Reject if invalid account formats found
                                aep_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "aep_account_numbers_invalid_length"
                                    ]
                                )

                            # CRITICAL: Form Type Validation - AEP Form Phrase
                            # Check if GPT-4o found the required AEP form identification phrase
                            if not aep_data.get("form_type_valid"):
                                aep_validation_issues.append(
                                    self.ERROR_MESSAGES["aep_wrong_form"]
                                )
                                self.logger.warning(
                                    "AEP form phrase not found - wrong form type detected"
                                )

                            # Ohio phrase utility (CODE-LEVEL DOUBLE-CHECK)
                            ohio_phrase_utility = (
                                aep_data.get("ohio_phrase_utility") or ""
                            ).upper()
                            valid_aep_utilities = [
                                "AEP",
                                "AEP OHIO",
                                "AMERICAN ELECTRIC POWER",
                                "CSPC",
                                "COLUMBUS SOUTHERN POWER COMPANY",
                                "COLUMBUS SOUTHERN POWER",
                                "OPC",
                                "OHIO POWER COMPANY",
                                "OHIO POWER",
                            ]
                            is_valid_utility = any(
                                valid_util in ohio_phrase_utility
                                for valid_util in valid_aep_utilities
                            )

                            if ohio_phrase_utility and not is_valid_utility:
                                aep_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "aep_wrong_utility_in_ohio_phrase"
                                    ]
                                )
                                aep_data["ohio_phrase_utility_valid"] = False
                            elif not aep_data.get("ohio_phrase_utility_valid"):
                                aep_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "aep_wrong_utility_in_ohio_phrase"
                                    ]
                                )

                            # Check interval granularity
                            if extraction_log.get("aep_granularity_missing"):
                                aep_validation_issues.append(
                                    "AEP: Interval data granularity is not specified (e.g., 'interval', 'summary', 'IDR')"
                                )

                            # Check broker signatures
                            # CRITICAL FIX: Ensure strings are never None before calling .lower()
                            authorized_person_text = (
                                aep_data.get("authorized_person_title") or ""
                            )
                            ohio_signature_text = (
                                aep_data.get("ohio_signature_text") or ""
                            )
                            broker_indicators = [
                                "on behalf of",
                                "for and on behalf of",
                                "utilities group",
                                "energy group",
                                "power group",
                                "broker",
                                "consultant",
                            ]

                            if any(
                                indicator in authorized_person_text.lower()
                                for indicator in broker_indicators
                            ):
                                aep_validation_issues.append(
                                    "Document signed by broker/third-party, not by customer - broker signature detected in Authorized Person field"
                                )
                            if any(
                                indicator in ohio_signature_text.lower()
                                for indicator in broker_indicators
                            ):
                                aep_validation_issues.append(
                                    "Document signed by broker/third-party, not by customer - broker signature detected in Ohio authorization statement"
                                )

                            self.logger.info(
                                f"Layer 3 complete: {len(aep_validation_issues)} field validation issue(s) found"
                            )

                            # Prominent prompt injection
                            if aep_validation_issues:
                                extraction_log["aep_code_level_validation_issues"] = (
                                    aep_validation_issues
                                )

                                extracted_text += "\n\n" + "=" * 80 + "\n"
                                extracted_text += (
                                    "CODE-LEVEL AEP COMPREHENSIVE VALIDATION RESULTS\n"
                                )
                                extracted_text += "=" * 80 + "\n"
                                extracted_text += (
                                    f"AEP ({self.provided_udc}) - Ohio Utility\n"
                                )
                                extracted_text += "IMPORTANT: AEP LOAs have standard Ohio form structure with required fields.\n\n"
                                extracted_text += "The following REQUIRED fields/validations were checked:\n\n"

                                for i, issue in enumerate(aep_validation_issues, 1):
                                    extracted_text += f"{i}. {issue}\n"

                                extracted_text += "\n**CRITICAL INSTRUCTION:**\n"
                                extracted_text += "These validation issues were detected by code-level checks with GPT-4o Vision.\n"
                                extracted_text += "You MUST include ALL of these issues in your rejectionReasons.\n"
                                extracted_text += "DO NOT skip or ignore any of these pre-validated issues.\n"
                                extracted_text += "=" * 80 + "\n\n"
                            else:
                                self.logger.info(
                                    "GPT-4o AEP validation passed - all required fields present"
                                )
                        else:
                            # GPT-4o failed - return ERROR status
                            self.logger.error(
                                "GPT-4o AEP verification failed - returning ERROR status"
                            )
                            return {
                                "document_id": document_id,
                                "fileName": document_id,
                                "validation_status": "ERROR",
                                "status": "ERROR",
                                "rejectionReasons": [
                                    "GPT-4o vision verification failed - cannot validate AEP document without vision analysis"
                                ],
                                "all_rejection_reasons": [
                                    "GPT-4o vision verification failed"
                                ],
                                "expiration_date": "N/A",
                                "ocr_success": extraction_log["extraction_success"],
                                "extracted_text_length": len(extracted_text),
                                "processing_timestamp": datetime.now().isoformat(),
                                "error": "GPT-4o AEP field verification did not return success",
                                "gpt4o_failure": True,
                            }
                    except Exception as e:
                        # GPT-4o exception - return ERROR status
                        self.logger.error(
                            f"GPT-4o AEP field verification exception: {str(e)}"
                        )
                        return {
                            "document_id": document_id,
                            "fileName": document_id,
                            "validation_status": "ERROR",
                            "status": "ERROR",
                            "rejectionReasons": [
                                f"GPT-4o vision verification exception: {str(e)}"
                            ],
                            "all_rejection_reasons": [
                                f"GPT-4o vision verification failed: {str(e)}"
                            ],
                            "expiration_date": "N/A",
                            "ocr_success": extraction_log["extraction_success"],
                            "extracted_text_length": len(extracted_text),
                            "processing_timestamp": datetime.now().isoformat(),
                            "error": f"GPT-4o AEP field verification exception: {str(e)}",
                            "gpt4o_failure": True,
                            "exception_details": str(e),
                        }
            except Exception as e:
                self.logger.error(f"AEP GPT-4o validation error: {str(e)}")
                extracted_text += f"\n\nAEP GPT-4O VALIDATION ERROR: {str(e)}\n"

        # NEW: Run comprehensive FirstEnergy validation (Great Lakes Region - Ohio)
        # THREE-LAYER APPROACH (same as ComEd) to defeat non-determinism:
        # Layer 1: Code-level validation of structure and form type
        # Layer 2: GPT-4o Vision extraction of all fields
        # Layer 3: Code validation of extracted fields + Prominent prompt injection
        if self.provided_udc and self.provided_udc.upper() in ["CEI", "OE", "TE"]:
            try:
                self.logger.info(
                    f"FirstEnergy document detected ({self.provided_udc}) - Running comprehensive three-layer validation..."
                )

                # LAYER 1: Code-level structural validation (form type, Ohio phrase utility)
                # This runs FIRST to catch wrong-form issues before GPT-4o
                try:
                    self.logger.info(
                        "Layer 1: Running code-level FirstEnergy structural validation..."
                    )
                    structural_issues = self.validate_firstenergy_required_fields(
                        extracted_text, extraction_log
                    )

                    # Store structural validation results
                    extraction_log["firstenergy_structural_validation_issues"] = (
                        structural_issues
                    )
                    self.logger.info(
                        f"Layer 1 complete: {len(structural_issues)} structural issue(s) found"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Layer 1 FirstEnergy structural validation error: {str(e)}"
                    )

                # LAYER 2: GPT-4o Vision extraction of all fields
                # This extracts the actual field values that will be validated
                if pdf_path:
                    try:
                        self.logger.info(
                            "Layer 2: Running GPT-4o Vision field extraction..."
                        )
                        gpt4o_result = self.gpt4o_verification_integration.verify_firstenergy_comprehensive_with_gpt4o(
                            pdf_path, extraction_log
                        )

                        if gpt4o_result.get("success"):
                            # Update extraction_log with GPT-4o extracted fields
                            extraction_log = gpt4o_result.get(
                                "extraction_log", extraction_log
                            )
                            self.logger.info(
                                "Layer 2 complete: GPT-4o successfully extracted all fields"
                            )

                            # LAYER 3: Code validation of GPT-4o extracted fields
                            self.logger.info(
                                "Layer 3: Validating GPT-4o extracted fields..."
                            )
                            firstenergy_validation_issues = []
                            fe_data = extraction_log.get("firstenergy_validation", {})

                            # LAYER 3 validates CRITICAL fields that must be present
                            # These include: Initial boxes, Account numbers, CRES Provider info, Ohio phrase utility

                            # CRITICAL: CRES Provider field validation (MUST be code-level enforced)
                            # GPT-4o prompt validation is unreliable for missing fields - we MUST check at code level
                            if not fe_data.get("cres_name_found"):
                                firstenergy_validation_issues.append(
                                    self.ERROR_MESSAGES["firstenergy_cres_name_missing"]
                                )
                                self.logger.info(
                                    "Added rejection for missing CRES Provider Name"
                                )

                            if not fe_data.get("cres_address_found"):
                                firstenergy_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "firstenergy_cres_address_missing"
                                    ]
                                )
                                self.logger.info(
                                    "Added rejection for missing CRES Provider Address"
                                )

                            if not fe_data.get("cres_phone_found"):
                                firstenergy_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "firstenergy_cres_phone_missing"
                                    ]
                                )
                                self.logger.info(
                                    "Added rejection for missing CRES Provider Phone"
                                )

                            # CRITICAL: CRES Email validation - this is the most commonly missing field
                            if not fe_data.get("cres_email_found"):
                                firstenergy_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "firstenergy_cres_email_missing"
                                    ]
                                )
                                self.logger.info(
                                    "Added rejection for missing CRES Provider Email"
                                )

                            # CRITICAL: Customer Information fields validation (top of document)
                            # Customer Name
                            if not fe_data.get("customer_name_found"):
                                firstenergy_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "firstenergy_customer_name_missing"
                                    ]
                                )
                                self.logger.info(
                                    "Added rejection for missing Customer Name"
                                )

                            # Customer Address
                            if not fe_data.get("customer_address_found"):
                                firstenergy_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "firstenergy_customer_address_missing"
                                    ]
                                )
                                self.logger.info(
                                    "Added rejection for missing Customer Address"
                                )

                            # Authorized Person/Title
                            if not fe_data.get("authorized_person_title_found"):
                                firstenergy_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "firstenergy_authorized_person_title_missing"
                                    ]
                                )
                                self.logger.info(
                                    "Added rejection for missing Authorized Person/Title"
                                )

                            # CRITICAL: Ohio Statement Signature and Date validation (bottom of document)
                            # Ohio Signature
                            if not fe_data.get("ohio_signature_found"):
                                firstenergy_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "firstenergy_ohio_signature_missing"
                                    ]
                                )
                                self.logger.info(
                                    "Added rejection for missing Ohio statement signature"
                                )

                            # Ohio Date
                            if not fe_data.get("ohio_date_found"):
                                firstenergy_validation_issues.append(
                                    self.ERROR_MESSAGES["firstenergy_ohio_date_missing"]
                                )
                                self.logger.info(
                                    "Added rejection for missing Ohio statement date"
                                )

                            # CRITICAL: Initial Box validation for FirstEnergy LOAs
                            # FirstEnergy LOAs have TWO initial boxes that MUST both be filled with LETTER INITIALS (not X marks)

                            # CRITICAL FIX: Get initial box data from NESTED structure
                            # GPT-4o stores this data in fe_data['initial_boxes'], not at top level
                            initial_boxes = fe_data.get("initial_boxes", {})
                            filled_box_count = initial_boxes.get("filled_box_count", 0)
                            empty_box_count = initial_boxes.get("empty_box_count", 0)
                            x_mark_count = initial_boxes.get("x_mark_count", 0)

                            # STEP 1: Check for X marks first (highest priority rejection)
                            if x_mark_count > 0:
                                # Reject for X marks - this takes priority over empty boxes
                                firstenergy_validation_issues.append(
                                    "First Energy: Unclear or ambiguous initials or x mark in initial boxes"
                                )
                                self.logger.info(
                                    f"Added rejection for {x_mark_count} X mark(s) in initial boxes"
                                )
                            # STEP 2: If no X marks, then check for empty boxes
                            elif empty_box_count > 0:
                                # FirstEnergy LOAs require 2 initial boxes to be filled
                                firstenergy_validation_issues.append(
                                    f"First Energy: {empty_box_count} initial box(es) are empty - both initial boxes must be initialed"
                                )
                                self.logger.info(
                                    f"Added rejection for {empty_box_count} empty initial box(es): Filled={filled_box_count}, Empty={empty_box_count}"
                                )

                            # CRITICAL: Multi-page account number scan
                            # Scan ALL pages if: account field is empty OR there's an attachment indicator
                            # "See attached" means we need to find and validate accounts in attachment pages
                            if fe_data.get(
                                "account_field_empty"
                            ) is True or fe_data.get("has_attachment_indicator", False):
                                try:
                                    self.logger.info(
                                        "FirstEnergy: Scanning ALL pages for account numbers (empty field or attachment indicated)..."
                                    )
                                    # CRITICAL FIX: scan_all_pages_for_account_numbers_with_gpt4o returns updated extraction_log
                                    extraction_log = self.gpt4o_verification_integration.scan_all_pages_for_account_numbers_with_gpt4o(
                                        pdf_path, extraction_log
                                    )
                                    # Update fe_data reference to get the updated values
                                    fe_data = extraction_log.get(
                                        "firstenergy_validation", {}
                                    )

                                    if fe_data.get("account_numbers_found"):
                                        # Accounts were found - multipage scan succeeded
                                        self.logger.info(
                                            f"Multi-page scan found {fe_data.get('account_count', 0)} account(s)"
                                        )
                                        self.logger.info(
                                            f"Account numbers: {fe_data.get('account_numbers', [])}"
                                        )
                                    else:
                                        self.logger.warning(
                                            "Multi-page scan found NO valid account numbers"
                                        )
                                except Exception as e:
                                    self.logger.error(
                                        f"Multi-page account scan error: {str(e)}"
                                    )

                            # CRITICAL: Account field validation - check TWO separate issues
                            # Issue 1: Is the account field completely empty? (no numbers AND no attachment)
                            if fe_data.get("account_field_empty") is True:
                                firstenergy_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "firstenergy_account_numbers_missing"
                                    ]
                                )
                                self.logger.info(
                                    "Added rejection for empty account field"
                                )
                            # Issue 2: Are visible account numbers the wrong length? (not 20 digits)
                            elif fe_data.get("account_length_valid") is False:
                                firstenergy_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "firstenergy_account_numbers_invalid_length"
                                    ]
                                )
                                self.logger.info(
                                    f"Added rejection for invalid account length: {fe_data.get('invalid_length_accounts', [])}"
                                )

                            # Form type validation (GPT-4o Vision determines this)
                            if not fe_data.get("form_type_valid"):
                                firstenergy_validation_issues.append(
                                    self.ERROR_MESSAGES["firstenergy_wrong_form"]
                                )

                            # CRITICAL FIX: Ohio phrase utility validation - CODE-LEVEL DOUBLE-CHECK
                            # Don't just trust GPT-4o's judgment - validate the actual utility name it extracted
                            # CEI = Cleveland Electric Illuminating, OE = Ohio Edison, TE = Toledo Edison
                            ohio_phrase_utility = (
                                fe_data.get("ohio_phrase_utility") or ""
                            ).upper()
                            valid_fe_utilities = [
                                "CEI",
                                "CLEVELAND ELECTRIC ILLUMINATING",
                                "CLEVELAND ILLUMINATING",
                                "OE",
                                "OHIO EDISON",
                                "TE",
                                "TOLEDO EDISON",
                                "THE ILLUMINATING COMPANY",
                                "THE ILLUMINATING CO",
                                "ILLUMINATING COMPANY",
                                "ILLUMINATING CO",
                            ]
                            invalid_generic_names = [
                                "FIRSTENERGY",
                                "FIRST ENERGY",
                                "FE",
                            ]

                            # Check if the extracted utility name is valid
                            is_valid_utility = any(
                                valid_util in ohio_phrase_utility
                                for valid_util in valid_fe_utilities
                            )
                            has_invalid_generic = any(
                                invalid in ohio_phrase_utility
                                for invalid in invalid_generic_names
                            )

                            if has_invalid_generic or (
                                ohio_phrase_utility and not is_valid_utility
                            ):
                                # Override GPT-4o's judgment - this is invalid
                                firstenergy_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "firstenergy_wrong_utility_in_ohio_phrase"
                                    ]
                                )
                                fe_data["ohio_phrase_utility_valid"] = (
                                    False  # Update the data
                                )
                                self.logger.warning(
                                    f"CODE-LEVEL OVERRIDE: Ohio phrase utility '{ohio_phrase_utility}' is NOT valid. Only CEI, OE, TE, or Illuminating Company are accepted."
                                )
                            elif not fe_data.get("ohio_phrase_utility_valid"):
                                # GPT-4o already flagged it as invalid
                                firstenergy_validation_issues.append(
                                    self.ERROR_MESSAGES[
                                        "firstenergy_wrong_utility_in_ohio_phrase"
                                    ]
                                )

                            # CRITICAL: Check for missing interval data granularity
                            # This was detected earlier in the FirstEnergy interval granularity detection
                            if extraction_log.get("firstenergy_granularity_missing"):
                                firstenergy_validation_issues.append(
                                    "First Energy: Interval data granularity is not specified (e.g., 'interval', 'summary', 'IDR')"
                                )
                                self.logger.info(
                                    "Added rejection for missing FirstEnergy interval data granularity"
                                )

                            # CRITICAL: Check for broker signatures in FirstEnergy documents
                            # For FirstEnergy, check BOTH the authorized person field AND Ohio signature text
                            # GPT-4o extracts both fields which may contain broker language
                            # CRITICAL FIX: Ensure strings are never None before calling .lower()
                            authorized_person_text = (
                                fe_data.get("authorized_person_title") or ""
                            )
                            ohio_signature_text = (
                                fe_data.get("ohio_signature_text") or ""
                            )

                            # Broker indicators to check for in the actual signature/authorized person fields
                            broker_indicators = [
                                "on behalf of",
                                "for and on behalf of",
                                "utilities group",
                                "energy group",
                                "power group",
                                "broker",
                                "consultant",
                            ]

                            # Check if either field contains broker language
                            has_broker_in_auth_person = any(
                                indicator in authorized_person_text.lower()
                                for indicator in broker_indicators
                            )
                            has_broker_in_ohio_sig = any(
                                indicator in ohio_signature_text.lower()
                                for indicator in broker_indicators
                            )

                            if has_broker_in_auth_person:
                                firstenergy_validation_issues.append(
                                    "Document signed by broker/third-party, not by customer - broker signature detected in Authorized Person field"
                                )
                                self.logger.info(
                                    f"Added rejection for broker signature in Authorized Person field: '{authorized_person_text}'"
                                )

                            if has_broker_in_ohio_sig:
                                firstenergy_validation_issues.append(
                                    "Document signed by broker/third-party, not by customer - broker signature detected in Ohio authorization statement"
                                )
                                self.logger.info(
                                    f"Added rejection for broker signature in Ohio statement: '{ohio_signature_text}'"
                                )

                            self.logger.info(
                                f"Layer 3 complete: {len(firstenergy_validation_issues)} field validation issue(s) found"
                            )

                            # LAYER 3 CONTINUATION: Prominent prompt injection (like ComEd)
                            if firstenergy_validation_issues:
                                # Store for tracking
                                extraction_log[
                                    "firstenergy_code_level_validation_issues"
                                ] = firstenergy_validation_issues

                                # Add VERY PROMINENT context (same style as ComEd)
                                extracted_text += "\n\n" + "=" * 80 + "\n"
                                extracted_text += "CODE-LEVEL FIRSTENERGY COMPREHENSIVE VALIDATION RESULTS\n"
                                extracted_text += "=" * 80 + "\n"
                                extracted_text += f"FirstEnergy ({self.provided_udc}) - Ohio Utility\n"
                                extracted_text += "IMPORTANT: FirstEnergy LOAs have a specific form structure with required fields.\n"
                                extracted_text += "\n"
                                extracted_text += "The following REQUIRED fields/validations were checked:\n\n"

                                for i, issue in enumerate(
                                    firstenergy_validation_issues, 1
                                ):
                                    extracted_text += f"{i}. {issue}\n"

                                extracted_text += "\n**CRITICAL INSTRUCTION:**\n"
                                extracted_text += "These validation issues were detected by code-level checks with GPT-4o Vision.\n"
                                extracted_text += "You MUST include ALL of these issues in your rejectionReasons.\n"
                                extracted_text += "DO NOT skip or ignore any of these pre-validated issues.\n"
                                extracted_text += "=" * 80 + "\n\n"
                            else:
                                self.logger.info(
                                    "GPT-4o First Energy validation passed - all required fields present"
                                )

                                # Add success context
                                extracted_text += "\n\n" + "=" * 80 + "\n"
                                extracted_text += "CODE-LEVEL FIRSTENERGY COMPREHENSIVE VALIDATION RESULTS\n"
                                extracted_text += "=" * 80 + "\n"
                                extracted_text += (
                                    "✓ ALL REQUIRED FIRSTENERGY FIELDS PRESENT:\n"
                                )
                                if fe_data.get("customer_name_found"):
                                    extracted_text += f"  ✓ Customer Name: {fe_data.get('customer_name', 'Found')}\n"
                                if fe_data.get("customer_phone_found"):
                                    extracted_text += f"  ✓ Customer Phone: {fe_data.get('customer_phone', 'Found')}\n"
                                if fe_data.get("customer_address_found"):
                                    extracted_text += f"  ✓ Customer Address: {fe_data.get('customer_address', 'Found')[:50]}...\n"
                                if fe_data.get("authorized_person_title_found"):
                                    extracted_text += f"  ✓ Authorized Person/Title: {fe_data.get('authorized_person_title', 'Found')}\n"
                                if fe_data.get("account_numbers_found"):
                                    account_count = fe_data.get("account_count", 0)
                                    has_attachment = fe_data.get(
                                        "has_attachment_indicator", False
                                    )
                                    extracted_text += (
                                        f"  ✓ Account/SDI Numbers: {account_count} found"
                                        + (
                                            " + attachment indicated"
                                            if has_attachment
                                            else ""
                                        )
                                        + "\n"
                                    )
                                if fe_data.get("cres_name_found"):
                                    extracted_text += f"  ✓ CRES Name: {fe_data.get('cres_name', 'Found')}\n"
                                if fe_data.get("ohio_signature_found"):
                                    extracted_text += (
                                        "  ✓ Ohio Statement Signature: Present\n"
                                    )
                                if fe_data.get("ohio_date_found"):
                                    extracted_text += f"  ✓ Ohio Statement Date: {fe_data.get('ohio_signature_date', 'Found')}\n"
                                if fe_data.get("form_type_valid"):
                                    extracted_text += (
                                        "  ✓ Form Type: Valid FirstEnergy LOA format\n"
                                    )
                                if fe_data.get("ohio_phrase_utility_valid"):
                                    extracted_text += f"  ✓ Ohio Phrase Utility: Valid ({fe_data.get('ohio_phrase_utility', 'N/A')})\n"
                                extracted_text += "\n"
                                extracted_text += "FirstEnergy validation passed - document contains all required fields.\n"
                                extracted_text += "=" * 80 + "\n\n"
                        else:
                            # CRITICAL: GPT-4o failed - return ERROR status immediately
                            self.logger.error(
                                "GPT-4o FirstEnergy verification failed - returning ERROR status"
                            )
                            return {
                                "document_id": document_id,
                                "fileName": document_id,
                                "validation_status": "ERROR",
                                "status": "ERROR",
                                "rejectionReasons": [
                                    "GPT-4o vision verification failed - cannot validate FirstEnergy document without vision analysis"
                                ],
                                "all_rejection_reasons": [
                                    "GPT-4o vision verification failed"
                                ],
                                "expiration_date": "N/A",
                                "ocr_success": extraction_log["extraction_success"],
                                "extracted_text_length": len(extracted_text),
                                "processing_timestamp": datetime.now().isoformat(),
                                "error": "GPT-4o FirstEnergy field verification did not return success",
                                "gpt4o_failure": True,
                            }
                    except Exception as e:
                        # CRITICAL: GPT-4o exception - return ERROR status immediately
                        self.logger.error(
                            f"GPT-4o FirstEnergy field verification exception: {str(e)}"
                        )
                        return {
                            "document_id": document_id,
                            "fileName": document_id,
                            "validation_status": "ERROR",
                            "status": "ERROR",
                            "rejectionReasons": [
                                f"GPT-4o vision verification exception: {str(e)}"
                            ],
                            "all_rejection_reasons": [
                                f"GPT-4o vision verification failed: {str(e)}"
                            ],
                            "expiration_date": "N/A",
                            "ocr_success": extraction_log["extraction_success"],
                            "extracted_text_length": len(extracted_text),
                            "processing_timestamp": datetime.now().isoformat(),
                            "error": f"GPT-4o FirstEnergy field verification exception: {str(e)}",
                            "gpt4o_failure": True,
                            "exception_details": str(e),
                        }

            except Exception as e:
                self.logger.error(f"FirstEnergy GPT-4o validation error: {str(e)}")
                extracted_text += f"\n\nFIRSTENERGY GPT-4O VALIDATION ERROR: {str(e)}\n"

        # Fallback Scenario 6: NHEC Request Type Options Verification (NHEC UDC Only - 2 options)
        # CRITICAL: Always use GPT-4o for NHEC - regex detection is unreliable
        if (
            self.region == "New England"
            and self.provided_udc
            and "NHEC" in self.provided_udc.upper()
            and pdf_path
        ):
            # Always run GPT-4o for NHEC, regardless of whether regex detected anything
            # This is because regex pattern matching is unreliable for checkbox detection
            try:
                self.logger.info(
                    "NHEC document detected - Running GPT-4o request type options verification..."
                )
                extraction_log = self.gpt4o_verification_integration.verify_nhec_request_type_options_with_gpt4o(
                    pdf_path, extraction_log
                )
                if extraction_log.get("nhec_request_type_options", {}).get(
                    "gpt4o_verified"
                ):
                    selection_count = extraction_log["nhec_request_type_options"][
                        "selection_count"
                    ]
                    extracted_text += f"\n\nGPT-4O NHEC REQUEST TYPE OPTIONS VERIFICATION APPLIED: Verified {selection_count} option(s) selected."
                    self.logger.info(
                        f"GPT-4o NHEC verification complete: {selection_count} option(s) selected"
                    )
                else:
                    self.logger.warning(
                        "GPT-4o NHEC verification did not return verified results"
                    )
            except Exception as e:
                error_msg = (
                    f"GPT-4o NHEC request type options verification failed: {str(e)}"
                )
                self.logger.error(error_msg)
                extracted_text += (
                    f"\n\nGPT-4O NHEC REQUEST TYPE OPTIONS VERIFICATION ERROR: {str(e)}"
                )

        # Fallback Scenario 7: CMP/FGE Billing Options Verification (CMP/FGE UDC Only - 2 options)
        # CRITICAL: Always use GPT-4o for CMP/FGE billing options - regex detection is unreliable
        if (
            self.region == "New England"
            and self.provided_udc
            and (
                "CMP" in self.provided_udc.upper() or "FGE" in self.provided_udc.upper()
            )
            and pdf_path
        ):
            # Always run GPT-4o for CMP/FGE, regardless of whether regex detected anything
            # This is because regex pattern matching is unreliable for checkbox detection
            try:
                self.logger.info(
                    f"{self.provided_udc} document detected - Running GPT-4o billing options verification..."
                )
                extraction_log = self.gpt4o_verification_integration.verify_cmp_billing_options_with_gpt4o(
                    pdf_path, extraction_log
                )
                if extraction_log.get("cmp_billing_options", {}).get("gpt4o_verified"):
                    selection_count = extraction_log["cmp_billing_options"][
                        "selection_count"
                    ]
                    extracted_text += f"\n\nGPT-4O {self.provided_udc} BILLING OPTIONS VERIFICATION APPLIED: Verified {selection_count} option(s) selected."
                    self.logger.info(
                        f"GPT-4o {self.provided_udc} verification complete: {selection_count} option(s) selected"
                    )
                else:
                    self.logger.warning(
                        f"GPT-4o {self.provided_udc} verification did not return verified results"
                    )
            except Exception as e:
                error_msg = f"GPT-4o {self.provided_udc} billing options verification failed: {str(e)}"
                self.logger.error(error_msg)
                extracted_text += f"\n\nGPT-4O {self.provided_udc} BILLING OPTIONS VERIFICATION ERROR: {str(e)}"

        # Fallback Scenario 8: PSNH Subscription Options Verification (PSNH UDC Only - 3 options)
        # CRITICAL: Always use GPT-4o for PSNH - regex detection is unreliable
        if (
            self.region == "New England"
            and self.provided_udc
            and "PSNH" in self.provided_udc.upper()
            and pdf_path
        ):
            # Always run GPT-4o for PSNH, regardless of whether regex detected anything
            # This is because regex pattern matching is unreliable for checkbox detection
            try:
                self.logger.info(
                    "PSNH document detected - Running GPT-4o subscription options verification..."
                )
                extraction_log = self.gpt4o_verification_integration.verify_psnh_subscription_options_with_gpt4o(
                    pdf_path, extraction_log
                )
                if extraction_log.get("psnh_subscription_options", {}).get(
                    "gpt4o_verified"
                ):
                    selection_count = extraction_log["psnh_subscription_options"][
                        "selection_count"
                    ]
                    extracted_text += f"\n\nGPT-4O PSNH SUBSCRIPTION OPTIONS VERIFICATION APPLIED: Verified {selection_count} option(s) selected."
                    self.logger.info(
                        f"GPT-4o PSNH verification complete: {selection_count} option(s) selected"
                    )
                else:
                    self.logger.warning(
                        "GPT-4o PSNH verification did not return verified results"
                    )
            except Exception as e:
                error_msg = (
                    f"GPT-4o PSNH subscription options verification failed: {str(e)}"
                )
                self.logger.error(error_msg)
                extracted_text += (
                    f"\n\nGPT-4O PSNH SUBSCRIPTION OPTIONS VERIFICATION ERROR: {str(e)}"
                )

        # CRITICAL: Comprehensive GPT-4o Vision Call for BECO - Always Run for Every BECO Document
        # This extracts ALL critical data that Azure OCR commonly misses on BECO forms
        if self.provided_udc and "BECO" in self.provided_udc.upper() and pdf_path:
            try:
                self.logger.info(
                    "BECO document detected - Running comprehensive GPT-4o vision extraction for ALL fields..."
                )
                comprehensive_result = self.gpt4o_verification_integration.extract_comprehensive_data_with_gpt4o(
                    pdf_path
                )

                if comprehensive_result.get("success"):
                    data = comprehensive_result.get("data", {})

                    # 1. Merge Service Options
                    if "service_options" in data:
                        if "service_options" not in extraction_log:
                            extraction_log["service_options"] = {}
                        extraction_log["service_options"].update(
                            data["service_options"]
                        )
                        extraction_log["service_options"]["gpt4o_verified"] = True
                        self.logger.info(
                            f"GPT-4o extracted service options: {data['service_options']}"
                        )

                    # 2. Merge Signature Detection Results
                    if "signatures" in data:
                        sig_data = data["signatures"]

                        # Store comprehensive signature detection results
                        extraction_log["beco_signature_detection"] = {
                            "customer_signature_present": sig_data.get(
                                "customer_signature_present", False
                            ),
                            "customer_signature_text": sig_data.get(
                                "customer_signature_text"
                            ),
                            "requestor_signature_present": sig_data.get(
                                "requestor_signature_present", False
                            ),
                            "requestor_signature_text": sig_data.get(
                                "requestor_signature_text"
                            ),
                            "gpt4o_verified": True,
                        }

                        self.logger.info("GPT-4o signature detection:")
                        self.logger.info(
                            f"  - Customer signature present: {sig_data.get('customer_signature_present')}"
                        )
                        self.logger.info(
                            f"  - Customer signature text: '{sig_data.get('customer_signature_text')}'"
                        )
                        self.logger.info(
                            f"  - Customer signature reasoning: {data.get('reasoning', 'No reasoning provided')}"
                        )
                        self.logger.info(
                            f"  - Requestor signature present: {sig_data.get('requestor_signature_present')}"
                        )
                        self.logger.info(
                            f"  - Requestor signature text: '{sig_data.get('requestor_signature_text')}'"
                        )

                        # Extract customer signature date
                        if sig_data.get("customer_signature_date"):
                            extraction_log["customer_date_extracted_by_gpt4o"] = (
                                sig_data["customer_signature_date"]
                            )
                            extraction_log["customer_date_extraction_success"] = True
                            self.logger.info(
                                f"  - Customer date extracted: {sig_data['customer_signature_date']}"
                            )

                        # Extract requestor signature date
                        if sig_data.get("requestor_signature_date"):
                            extraction_log["requestor_date_extracted_by_gpt4o"] = (
                                sig_data["requestor_signature_date"]
                            )
                            extraction_log["requestor_date_extraction_success"] = True
                            self.logger.info(
                                f"  - Requestor date extracted: {sig_data['requestor_signature_date']}"
                            )

                    # 3. Merge Requestor/Billing Information (Key-Value Pairs)
                    if "requestor_billing_info" in data:
                        rb_data = data["requestor_billing_info"]

                        # Always use GPT-4o data for BECO (it's more accurate than Azure OCR)
                        kv_pairs = []
                        for key, value in rb_data.items():
                            if value:
                                kv_pairs.append(
                                    {
                                        "key": key,
                                        "value": value,
                                        "confidence": data.get("confidence", 99)
                                        / 100.0,
                                    }
                                )

                        if kv_pairs:
                            extraction_log["key_value_pairs"] = kv_pairs
                            extraction_log["requestor_billing_extracted_by_gpt4o"] = (
                                rb_data
                            )
                            extraction_log["requestor_billing_extraction_success"] = (
                                True
                            )
                            self.logger.info(
                                f"GPT-4o extracted {len(kv_pairs)} requestor/billing fields"
                            )

                    # 4. Merge Account Numbers
                    if "account_numbers" in data:
                        acc_data = data["account_numbers"]
                        extraction_log["beco_account_numbers"] = {
                            "has_account_numbers": acc_data.get(
                                "has_account_numbers", False
                            ),
                            "account_numbers_found": acc_data.get(
                                "account_numbers_found", []
                            ),
                            "account_count": acc_data.get("account_count", 0),
                            "gpt4o_verified": True,
                        }
                        self.logger.info(
                            f"GPT-4o account number detection: {acc_data.get('account_count', 0)} found"
                        )

                    extracted_text += "\n\n" + "=" * 80 + "\n"
                    extracted_text += (
                        "GPT-4O COMPREHENSIVE VISION EXTRACTION APPLIED FOR BECO\n"
                    )
                    extracted_text += "=" * 80 + "\n"
                    extracted_text += "Extracted:\n"
                    extracted_text += (
                        f"- Service Options: {data.get('service_options', {})}\n"
                    )
                    extracted_text += f"- Customer Signature: {'PRESENT' if data.get('signatures', {}).get('customer_signature_present') else 'MISSING'}\n"
                    extracted_text += f"- Customer Date: {data.get('signatures', {}).get('customer_signature_date', 'Not found')}\n"
                    extracted_text += f"- Requestor Signature: {'PRESENT' if data.get('signatures', {}).get('requestor_signature_present') else 'MISSING'}\n"
                    extracted_text += f"- Requestor Date: {data.get('signatures', {}).get('requestor_signature_date', 'Not found')}\n"
                    extracted_text += f"- Requestor/Billing Fields: {len(data.get('requestor_billing_info', {}))} fields\n"
                    extracted_text += "=" * 80 + "\n\n"

                else:
                    self.logger.warning(
                        "GPT-4o comprehensive extraction did not return success"
                    )
                    extracted_text += "\n\nGPT-4O COMPREHENSIVE EXTRACTION WARNING: Did not return success.\n"

            except Exception as e:
                error_msg = f"GPT-4o comprehensive extraction failed: {str(e)}"
                self.logger.error(error_msg)
                extracted_text += (
                    f"\n\nGPT-4O COMPREHENSIVE EXTRACTION ERROR FOR BECO: {str(e)}\n"
                )

        # Determine state based on provided parameters or utility detection
        # CRITICAL: Use validator region to set appropriate default state and UDC
        # IMPORTANT: Always prioritize the UDC passed as input parameter to the model
        provided_udc = self.provided_udc  # Use the UDC passed at initialization time
        # Note: We use self.region directly, no need for a separate provided_region variable

        # Set region-appropriate default state based on validator region
        # IMPORTANT: This sets the baseline that should be maintained unless explicitly overridden
        if self.region == "New England":
            detected_state = "MA"  # Default to Massachusetts for New England validators
        else:
            detected_state = "OH"  # Default to Ohio for Great Lakes validators

        # If UDC is provided, use it to determine state
        if provided_udc:
            udc_upper = provided_udc.upper()
            # Map known UDCs to states - includes both Great Lakes and New England
            # Updated to include case-insensitive variants and exact matching
            udc_to_state_mapping = {
                # Great Lakes Region UDCs - Multiple variants for robust matching
                "COMED": "IL",
                "CEI": "OH",  # FirstEnergy - Cleveland Electric Illuminating
                "TE": "OH",  # FirstEnergy - Toledo Edison
                "OE": "OH",  # FirstEnergy - Ohio Edison
                "DAYTON": "OH",
                "CILCO": "IL",
                "CINERGY": "OH",
                "CIPS": "IL",
                "IP": "IL",
                "CSPC": "OH",
                "OPC": "OH",
                "COMMONWEALTH EDISON": "IL",
                "COMMED": "IL",  # Common typo variant
                "AEP": "OH",
                "FIRSTENERGY": "OH",
                "FIRST ENERGY": "OH",
                "DUKE": "OH",
                "DUKE ENERGY": "OH",
                "AMEREN": "IL",
                "AMEREN ILLINOIS": "IL",
                "CONSUMERS": "MI",
                "CONSUMERS ENERGY": "MI",
                "DTE": "MI",
                "DTE ENERGY": "MI",
                "DETROIT EDISON": "MI",
                "CSPS": "OH",
                # New England Region UDCs
                "BHE": "ME",
                "BANGOR HYDRO ELECTRIC": "ME",
                "CMP": "ME",
                "CENTRAL MAINE POWER": "ME",
                "FGE": "MA",
                "FITCHBURG GAS & ELECTRIC": "MA",
                "NHEC": "NH",
                "NEW HAMPSHIRE ELECTRIC CO-OP": "NH",
                "GSECO": "NH",
                "GRANITE STATE ELECTRIC": "NH",
                "LIBERTY UTILITIES": "NH",
                "NANT": "MA",
                "NANTUCKET ELECTRIC": "MA",
                "NGRID": "MA",
                "MECO": "MA",
                "MASSACHUSETTS ELECTRIC": "MA",
                "NECO": "RI",
                "NARRAGANSETT ELECTRIC": "RI",
                "PPL": "RI",
                "BECO": "MA",
                "BOSTON EDISON": "MA",
                "CECO": "MA",
                "COMMONWEALTH ELECTRIC": "MA",
                "CELCO": "MA",
                "CAMBRIDGE ELECTRIC LIGHT": "MA",
                "CLP": "CT",
                "CONNECTICUT LIGHT & POWER": "CT",
                "PSNH": "NH",
                "PUBLIC SERVICE OF NEW HAMPSHIRE": "NH",
                "WMECO": "MA",
                "WESTERN MASSACHUSETTS ELECTRIC": "MA",
                "UI": "CT",
                "UNITED ILLUMINATING": "CT",
                "UES": "NH",
                "UNITIL ENERGY SYSTEMS": "NH",
                "CENTRAL ILLINOIS LIGHT COMPANY": "IL",
                "CENTRAL ILLINOIS PUBLIC SERVICE COMPANY": "IL",
                "ILLINOIS POWER": "IL",
            }

            # First try exact match
            if udc_upper in udc_to_state_mapping:
                detected_state = udc_to_state_mapping[udc_upper]
            else:
                # Then try substring matching for partial matches (except for multi-state utilities)
                for udc_key, state in udc_to_state_mapping.items():
                    if udc_key in udc_upper or udc_upper in udc_key:
                        detected_state = state
                        break

        # No OCR-based utility detection - UDC is always provided as input parameter

        # Extract signature date - Use GPT-4o for accurate customer date extraction
        # CRITICAL: Always use CUSTOMER signature date, not supplier/broker date
        signature_dates = []
        customer_signature_date_from_gpt4o = None

        # Try GPT-4o extraction first if pdf_path is available (most reliable method)
        # SKIP if comprehensive fallback already ran for BECO and extracted the date
        if pdf_path and not extraction_log.get("customer_date_extracted_by_gpt4o"):
            try:
                self.logger.info("Using GPT-4o to extract customer signature date...")
                gpt4o_result = self.gpt4o_verification_integration.extract_customer_signature_date_with_gpt4o(
                    pdf_path
                )

                if gpt4o_result.get("success") and gpt4o_result.get(
                    "customer_signature_date"
                ):
                    customer_signature_date_from_gpt4o = gpt4o_result[
                        "customer_signature_date"
                    ]
                    signature_dates.append(customer_signature_date_from_gpt4o)
                    self.logger.info(
                        f"GPT-4o extracted customer signature date: {customer_signature_date_from_gpt4o}"
                    )

                    # Log the full GPT-4o response for debugging
                    self.logger.info(
                        f"Full GPT-4o response for customer signature date: {json.dumps(gpt4o_result, indent=2)}"
                    )

                    # CRITICAL: Add extracted date in VERY PROMINENT format that GPT-4o cannot miss
                    extracted_text += f"\n\n{'='*80}\n"
                    extracted_text += (
                        "CRITICAL VALIDATION NOTE - CUSTOMER SIGNATURE DATE EXISTS\n"
                    )
                    extracted_text += f"{'='*80}\n"
                    extracted_text += (
                        "GPT-4O CUSTOMER SIGNATURE DATE EXTRACTION RESULT:\n"
                    )
                    extracted_text += f"- Customer Signature Date: {customer_signature_date_from_gpt4o}\n"
                    extracted_text += f"- Location: {gpt4o_result.get('location_description', 'Customer Information section')}\n"
                    extracted_text += (
                        f"- Confidence: {gpt4o_result.get('confidence', 100)}%\n"
                    )
                    extracted_text += "- STATUS: DATE IS PRESENT (NOT MISSING)\n"
                    extracted_text += "\nIMPORTANT: This date was successfully extracted from the document.\n"
                    extracted_text += "DO NOT report 'Customer signature date is missing' in rejection reasons.\n"
                    extracted_text += f"{'='*80}\n\n"

                    # CRITICAL FIX: Store the successfully extracted date for validation bypass
                    extraction_log["customer_date_extracted_by_gpt4o"] = (
                        customer_signature_date_from_gpt4o
                    )
                    extraction_log["customer_date_extraction_success"] = True
                else:
                    self.logger.warning(
                        "GPT-4o customer signature date extraction failed or returned no date"
                    )
                    extraction_log["customer_date_extraction_success"] = False
            except Exception as e:
                self.logger.error(
                    f"GPT-4o customer signature date extraction error: {str(e)}"
                )
                extraction_log["customer_date_extraction_success"] = False

        # Fallback to regex if GPT-4o didn't find a date
        if not signature_dates:
            self.logger.info(
                "Falling back to regex for customer signature date extraction..."
            )
            # Customer-specific signature date patterns
            # CRITICAL: Added specific pattern for "Date signed by customer" and digital signature formats
            customer_signature_patterns = [
                # Specific pattern for "Date signed by customer" phrase (with optional spaces)
                r"Date\s+signed\s+by\s+customer[:\s]*(\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{2,4})",
                # Digital signature with full month name: "March 19, 2025 | 2:09 PM PDT"
                r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})\s*\|",
                # Digital signature date pattern: "Date: 2024.05.03" or "Date: 2024.05.03 10:40:36-04'00'"
                r"Date:\s*(\d{4}\.\d{2}\.\d{2})",
                # Patterns with optional spaces around separators
                r"Customer\s+Information.*?Date[:\s]*(\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{2,4})",
                r"To be completed by the Customer.*?Date[:\s]*(\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{2,4})",
                r"[*\s]*Customer Signature.*?[*\s]*Date[:\s]*(\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{2,4})",
                r"Customer.*?Signature.*?(\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{2,4})",
            ]

            # Try customer-specific patterns
            for pattern in customer_signature_patterns:
                dates = re.findall(pattern, extracted_text, re.IGNORECASE | re.DOTALL)
                if dates:
                    # Clean up any spaces in the captured dates (e.g., "6/ 3/ 2025" -> "6/3/2025")
                    cleaned_dates = [re.sub(r"\s+", "", date) for date in dates]
                    signature_dates.extend(cleaned_dates)
                    break

            # If still no date, try more general patterns
            if not signature_dates:
                general_patterns = [
                    r"signature.*?(\d{1,2}[-/]\d{1,2}[-/]\d{4})",
                    r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})",
                ]

                for pattern in general_patterns:
                    dates = re.findall(pattern, extracted_text, re.IGNORECASE)
                    if dates:
                        signature_dates.extend(dates[:1])  # Only take first match
                        break

        signature_validity_result = None
        loa_expiration_result = None
        if signature_dates:
            # Use the first date found (usually the signature date)
            signature_date_str = signature_dates[0]
            signature_validity_result = self.calculate_signature_validity(
                signature_date_str, detected_state
            )

            # Calculate LOA expiration date
            # Use the provided UDC for expiration rules (more reliable than OCR detection)
            utility_for_expiration = provided_udc if provided_udc else None

            loa_expiration_result = self.calculate_loa_expiration_date(
                signature_date_str,
                detected_state,
                utility_for_expiration,
                extracted_text,
            )

        # Smart Fallback for BECO for missing Key-Value Pairs (MUST RUN BEFORE LAYOUT CONTEXT IS BUILT)
        requestor_billing_data_from_gpt4o = {}

        if (
            self.provided_udc
            and "BECO" in self.provided_udc.upper()
            and pdf_path
            and not extraction_log.get("key_value_pairs")
        ):
            try:
                self.logger.info(
                    "BECO document with missing key-value pairs detected - Running comprehensive GPT-4o fallback..."
                )
                comprehensive_result = self.gpt4o_verification_integration.extract_comprehensive_data_with_gpt4o(
                    pdf_path
                )
                if comprehensive_result.get("success"):
                    data = comprehensive_result.get("data", {})

                    # Intelligently merge the results
                    if "requestor_billing_info" in data:
                        kv_pairs = []
                        requestor_billing_data_from_gpt4o = data[
                            "requestor_billing_info"
                        ]

                        for key, value in data["requestor_billing_info"].items():
                            if value:
                                kv_pairs.append(
                                    {
                                        "key": key,
                                        "value": value,
                                        "confidence": data.get("confidence", 99)
                                        / 100.0,
                                    }
                                )

                        extraction_log["key_value_pairs"] = kv_pairs
                        key_value_pairs = kv_pairs  # Update local variable

                        # Store for later validation bypass
                        extraction_log["requestor_billing_extracted_by_gpt4o"] = (
                            requestor_billing_data_from_gpt4o
                        )
                        extraction_log["requestor_billing_extraction_success"] = True

                        self.logger.info(
                            f"GPT-4o fallback successfully extracted {len(kv_pairs)} key-value pairs."
                        )

                    extracted_text += "\n\nGPT-4O COMPREHENSIVE FALLBACK APPLIED FOR BECO KEY-VALUE PAIRS.\n"
            except Exception as e:
                extracted_text += (
                    f"\n\nGPT-4O COMPREHENSIVE FALLBACK ERROR FOR BECO: {str(e)}"
                )

        # Process with GPT-4o using advanced layout information
        # Create detailed context with layout analysis results
        layout_context = f"""
            ADVANCED LAYOUT ANALYSIS RESULTS:

            SELECTION MARKS DETECTED: {len(selection_marks)} marks found
            """

        for i, mark in enumerate(selection_marks, 1):
            layout_context += f"Mark {i}: State={mark.get('state', 'unknown')}, Confidence={mark.get('confidence', 'N/A')}, Content={mark.get('content', 'N/A')}\n"

        layout_context += (
            f"\nKEY-VALUE PAIRS DETECTED: {len(key_value_pairs)} pairs found\n"
        )
        for pair in key_value_pairs:
            layout_context += f"Field: {pair.get('key', 'N/A')} = {pair.get('value', 'N/A')} (Confidence: {pair.get('confidence', 'N/A')})\n"

        # Add information about potential handwritten initials
        layout_context += f"\nPOTENTIAL HANDWRITTEN INITIALS DETECTED: {len(potential_initials)} found\n"
        for i, initial in enumerate(potential_initials, 1):
            layout_context += f"Initial {i}: Text='{initial.get('text', '')}', Context='{initial.get('context', '')}'\n"

        # UDC information from input parameter (not detected from OCR)
        if self.provided_udc:
            layout_context += f"\nPROVIDED UDC: {self.provided_udc}\n"

        # Add FirstEnergy interval granularity information if applicable (for ALL regions)
        if is_firstenergy_udc and extraction_log.get(
            "firstenergy_interval_granularity"
        ):
            fe_granularity = extraction_log["firstenergy_interval_granularity"]
            layout_context += "\nFIRSTENERGY INTERVAL DATA GRANULARITY DETECTION:\n"
            layout_context += "- GPT-4o Vision Analysis Applied: Yes\n"
            layout_context += (
                f"- Granularity Text Found: {fe_granularity.get('text_found', False)}\n"
            )

            if fe_granularity.get("text_found"):
                layout_context += f"- Extracted Text: '{fe_granularity.get('extracted_text', 'N/A')}'\n"
                layout_context += (
                    f"- Location: {fe_granularity.get('location', 'N/A')}\n"
                )
                layout_context += (
                    f"- Confidence: {fe_granularity.get('confidence', 0)}%\n"
                )
                layout_context += "\n**CRITICAL**: Interval data granularity IS SPECIFIED in this FirstEnergy document.\n"
                layout_context += f"The text '{fe_granularity.get('extracted_text')}' clearly indicates the data types being requested.\n"
                layout_context += "DO NOT reject for 'Interval data granularity is not specified' - it IS specified.\n"
            else:
                layout_context += (
                    "- Status: No interval granularity specifications found\n"
                )
                layout_context += (
                    "- This MAY indicate missing granularity information\n"
                )

            layout_context += (
                f"- GPT-4o Verified: {fe_granularity.get('gpt4o_verified', False)}\n"
            )

        # Add information about service options for New England LOAs
        if self.region == "New England":
            service_options = extraction_log.get("service_options", {})
            layout_context += "\nNEW ENGLAND SERVICE OPTIONS DETECTION:\n"
            layout_context += (
                f"- Options Detected: {service_options.get('detected', False)}\n"
            )
            layout_context += f"- One Time Request Selected: {service_options.get('one_time_selected', False)}\n"
            layout_context += f"- Annual Subscription Selected: {service_options.get('annual_subscription_selected', False)}\n"
            layout_context += (
                f"- Selection Count: {service_options.get('selection_count', 0)}\n"
            )
            layout_context += (
                f"- GPT-4o Verified: {service_options.get('gpt4o_verified', False)}\n"
            )

            # Add MECO subscription options information if applicable
            if self.provided_udc and "MECO" in self.provided_udc.upper():
                meco_options = extraction_log.get("meco_subscription_options", {})
                layout_context += "\nMECO SUBSCRIPTION OPTIONS DETECTION:\n"
                layout_context += (
                    f"- Options Detected: {meco_options.get('detected', False)}\n"
                )
                layout_context += f"- Two Weeks Selected: {meco_options.get('two_weeks_selected', False)}\n"
                layout_context += f"- One Year Selected: {meco_options.get('one_year_selected', False)}\n"
                layout_context += f"- Auto-Renewing Selected: {meco_options.get('auto_renewing_selected', False)}\n"
                layout_context += (
                    f"- Selection Count: {meco_options.get('selection_count', 0)}\n"
                )
                layout_context += (
                    f"- GPT-4o Verified: {meco_options.get('gpt4o_verified', False)}\n"
                )
                layout_context += "\nIMPORTANT: MECO LOAs require exactly ONE subscription option to be selected.\n"
                layout_context += "- If selection_count = 0: REJECT (No subscription option selected)\n"
                layout_context += "- If selection_count > 1: REJECT (Multiple subscription options selected)\n"
                layout_context += "- If selection_count = 1: ACCEPT (Correct - exactly one option selected)\n"

            # Add NECO subscription options information if applicable
            if self.provided_udc and "NECO" in self.provided_udc.upper():
                neco_options = extraction_log.get("neco_subscription_options", {})
                layout_context += "\nNECO SUBSCRIPTION OPTIONS DETECTION:\n"
                layout_context += (
                    f"- Options Detected: {neco_options.get('detected', False)}\n"
                )
                layout_context += f"- Two Weeks Selected: {neco_options.get('two_weeks_selected', False)}\n"
                layout_context += f"- One Year Selected: {neco_options.get('one_year_selected', False)}\n"
                layout_context += (
                    f"- Selection Count: {neco_options.get('selection_count', 0)}\n"
                )
                layout_context += (
                    f"- GPT-4o Verified: {neco_options.get('gpt4o_verified', False)}\n"
                )
                layout_context += "\nIMPORTANT: NECO LOAs require exactly ONE subscription option to be selected.\n"
                layout_context += "- If selection_count = 0: REJECT (No subscription option selected)\n"
                layout_context += "- If selection_count > 1: REJECT (Multiple subscription options selected)\n"
                layout_context += "- If selection_count = 1: ACCEPT (Correct - exactly one option selected)\n"

            # Add NHEC request type options information if applicable
            if self.provided_udc and "NHEC" in self.provided_udc.upper():
                nhec_options = extraction_log.get("nhec_request_type_options", {})
                layout_context += "\nNHEC REQUEST TYPE OPTIONS DETECTION:\n"
                layout_context += (
                    f"- Options Detected: {nhec_options.get('detected', False)}\n"
                )
                layout_context += f"- Ad-hoc Request Selected: {nhec_options.get('adhoc_selected', False)}\n"
                layout_context += f"- Subscription Request Selected: {nhec_options.get('subscription_selected', False)}\n"
                layout_context += (
                    f"- Selection Count: {nhec_options.get('selection_count', 0)}\n"
                )
                layout_context += (
                    f"- GPT-4o Verified: {nhec_options.get('gpt4o_verified', False)}\n"
                )
                layout_context += "\nIMPORTANT: NHEC LOAs require exactly ONE request type option to be selected.\n"
                layout_context += "- If selection_count = 0: REJECT (No request type option selected)\n"
                layout_context += "- If selection_count > 1: REJECT (Multiple request type options selected)\n"
                layout_context += "- If selection_count = 1: ACCEPT (Correct - exactly one option selected)\n"

            # Add CMP billing options information if applicable
            if self.provided_udc and "CMP" in self.provided_udc.upper():
                cmp_options = extraction_log.get("cmp_billing_options", {})
                layout_context += "\nCMP BILLING OPTIONS DETECTION:\n"
                layout_context += f"- Billing Section Exists: {cmp_options.get('billing_section_exists', True)}\n"
                layout_context += (
                    f"- Options Detected: {cmp_options.get('detected', False)}\n"
                )
                layout_context += f"- Invoice Customer Selected: {cmp_options.get('invoice_customer_selected', False)}\n"
                layout_context += f"- Invoice Supplier/Broker Selected: {cmp_options.get('invoice_supplier_selected', False)}\n"
                layout_context += (
                    f"- Selection Count: {cmp_options.get('selection_count', 0)}\n"
                )
                layout_context += (
                    f"- GPT-4o Verified: {cmp_options.get('gpt4o_verified', False)}\n"
                )
                layout_context += "\nIMPORTANT: CMP billing options validation rules:\n"
                layout_context += "- If billing_section_exists=False: VALID - DO NOT reject (some older CMP LOAs don't have this section)\n"
                layout_context += "- If billing_section_exists=True AND selection_count=0: REJECT (No billing option selected)\n"
                layout_context += "- If billing_section_exists=True AND selection_count=1: ACCEPT (Correct - exactly one option selected)\n"
                layout_context += "- If billing_section_exists=True AND selection_count=2: REJECT (Both billing options selected)\n"

            # Add detailed verification information if available
            if service_options.get("gpt4o_verified"):
                layout_context += (
                    "\nIMPORTANT - GPT-4o SERVICE OPTIONS VERIFICATION RESULTS:\n"
                )
                layout_context += "The GPT-4o vision model has verified the service options selection with high confidence.\n"
                if service_options.get("one_time_selected"):
                    layout_context += (
                        "VERIFIED RESULT: 'One Time Request' is SELECTED.\n"
                    )
                if service_options.get("annual_subscription_selected"):
                    layout_context += (
                        "VERIFIED RESULT: 'Annual Subscription' is SELECTED.\n"
                    )
                layout_context += f"Total options selected: {service_options.get('selection_count', 0)}\n"
                layout_context += f"Verification details: {str(service_options.get('gpt4o_verification_details', {}))}\n"
                layout_context += "\nIMPORTANT: For New England LOAs, exactly ONE service option must be selected. When GPT-4o has verified that exactly one option is selected, this requirement is SATISFIED and should not cause rejection.\n"

        # Load system prompt from markdown file with state-specific rules
        system_prompt = self._load_system_prompt(detected_state)

        # Apply state-specific modifications to the prompt
        if detected_state == "IL":
            # Modify prompt for Illinois - remove Ohio-specific requirements
            system_prompt = system_prompt.replace(
                "with special focus on Ohio LOAs", "with special focus on Illinois LOAs"
            )
            # Remove Ohio-specific statement requirement for Illinois
            ohio_statement_section = """8. **Ohio-Specific Requirements**: UNIVERSAL UTILITY NAME VALIDATION - ENHANCED CONSISTENCY CHECK
   - Must include this statement structure prominently before signature. If it does not include it the LOA will be rejected:
   "I realize that under the rules and regulations of the public utilities commission of Ohio, I may refuse to allow [UTILITY NAME] to release the information set forth above. By my signature, I freely give [UTILITY NAME] permission to release the information designated above."
   E.g: This is a valid statement "I realize that under the rules and regulations of the Public Utilities Commission of Ohio, I may refuse to allow AEP Ohio to release the information set forth above. By my signature, I freely give AEP Ohio permission to release the information designated above."
   American Electric Power operates for Ohio, so LOAs with this utility which don't have the Ohio specific statement will be rejected.
   - ENHANCED CONSISTENCY VALIDATION: Accept utility names as consistent if they refer to the same utility company, even with minor variations:
     * "AEP Ohio" and "AEP Ohio" = CONSISTENT (exact match)
     * "AEP Ohio" and "AEP" = CONSISTENT (abbreviated form)
     * "Toledo Edison" and "Toledo Edison Company" = CONSISTENT (with/without "Company")
     * "Ohio Edison" and "OE" = CONSISTENT (abbreviated form)
     * "CNE" and "Constellation" = CONSISTENT (CNE is Constellation)
     * "Duke Energy" and "Cinergy" = CONSISTENT (Cinergy is part of Duke Energy)
     * Only REJECT if clearly different utilities are referenced (e.g., "AEP Ohio" vs "Toledo Edison")
   - CRITICAL UPDATE: When the Ohio-specific statement mentions "CNE", "Duke Energy", or "Cinergy" as the utility name, this is VALID and should be ACCEPTED. These are valid utility name references for Ohio LOAs.
   - IMPORTANT: This change was made because documents like "Aimbridge Hospitality_AEP_LOA" correctly use "AEP Ohio" in both places but were incorrectly flagged as inconsistent
   - IMPORTANT: The [UTILITY NAME] can be ANY utility company mentioned in the document
   - DO NOT require a specific utility name - accept ANY utility name mentioned in the document
   - The statement should reference the UTILITY (like "Toledo Edison") not the CRES provider
   - This is CORRECT because the utility releases the data to the CRES provider
   - Must specify identity of data recipients
   - Must specify type and granularity of data being collected
   - If the document mentions "For multiple account/SDI numbers, please attach a spreadsheet..." but doesn't indicate if attachments are included, REJECT
   - This is a GENERAL requirement that applies to the entire document, not just specific sections"""

            illinois_statement_section = """8. **Illinois-Specific Requirements**: ComEd/Illinois Utility Validation
   - Illinois LOAs do not require Ohio-specific utility statements
   - Focus on proper interval data authorization and EUI requirements
   - Third-party broker authorization is ALLOWED in Illinois
   - Utility name consistency validation applies but without Ohio statement requirement"""

            system_prompt = system_prompt.replace(
                ohio_statement_section, illinois_statement_section
            )

        # Check for email addresses and validate domains
        email_pattern = r"Email:\s*([^\s]+@[^\s]+)"
        email_matches = re.findall(email_pattern, extracted_text, re.IGNORECASE)

        # Enhanced email pattern to find more email addresses in the document
        enhanced_email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        enhanced_email_matches = re.findall(
            enhanced_email_pattern, extracted_text, re.IGNORECASE
        )

        # Combine and deduplicate email matches
        all_email_matches = list(set(email_matches + enhanced_email_matches))

        # CRITICAL: Different regions have different broker validation rules
        email_validation_context = ""
        if all_email_matches:
            # Check region-specific email validation rules
            is_illinois = detected_state == "IL"
            is_new_england = self.region == "New England"

            if is_new_england:
                # CRITICAL: Only check emails in Customer Information section, not page 2 or other sections
                # Extract the Customer Information section (typically at bottom of document)
                customer_section_patterns = [
                    r"Customer\s+Information.*?(?=\n\n\n|\Z)",
                    r"To\s+be\s+completed\s+by\s+the?\s+Customer.*?(?=\n\n\n|\Z)",
                    r"Customer\s+Authorization.*?(?=\n\n\n|\Z)",
                    r"Customer\s+Signature.*?(?=(?:Supplier|CRES|Third\s+Party|\n\n\n)|\Z)",
                ]

                customer_section_text = ""
                for pattern in customer_section_patterns:
                    matches = re.findall(
                        pattern, extracted_text, re.IGNORECASE | re.DOTALL
                    )
                    if matches:
                        customer_section_text = matches[0]
                        break

                # Now find emails ONLY in the customer section (if found)
                customer_section_emails = []
                if customer_section_text:
                    for email in all_email_matches:
                        if email.lower() in customer_section_text.lower():
                            customer_section_emails.append(email)
                else:
                    # If no customer section found, skip broker email validation entirely
                    # (don't want to falsely reject based on emails in other sections)
                    customer_section_emails = []

                # Simple New England broker detection - only check emails from customer section
                broker_domains = [
                    "getchoice.com",
                    "energycx.com",
                    "ezenergyservices.com",
                    "berryglob.com",
                    "energylink.com",
                    "energyservicesgroup.net",
                    "utilityaccount.com",
                    "energywatch.com",
                    "totalchoiceusa.com",
                    "energyintel.com",
                    "energyprocurement.com",
                    "powersource.com",
                    "energyadvisors.com",
                    "utilityservices.com",
                    "energyconsultants.com",
                ]

                # Energy-related domain keywords that suggest a broker
                energy_keywords = [
                    "energy",
                    "power",
                    "util",
                    "electric",
                    "broker",
                    "consult",
                ]

                broker_emails = []

                email_validation_context = (
                    f"NEW ENGLAND BROKER EMAIL DETECTION:\n"
                    f"Found email addresses in document: {all_email_matches}\n"
                    f"\n"
                    f"EMAIL CLASSIFICATION:\n"
                )

                for email in all_email_matches:
                    if "@" not in email:
                        continue

                    domain = email.split("@")[1].lower()

                    # Skip Constellation domains and affiliated/partner domains
                    if any(
                        constellation_domain in domain
                        for constellation_domain in [
                            "constellation",
                            "retailoperations",
                            "felpower",
                            "sesenergy",
                        ]
                    ):
                        email_validation_context += f"- {email} → CRES Provider (Constellation/Partner domain) - VALID\n"
                        continue

                    # Skip utility domains and submission instruction emails (often in instructions)
                    utility_domains = [
                        "eversource.com",
                        "cmpco.com",
                        "neco.com",
                        "rienergy.com",  # NECO utility domain
                        "balancedrockenergy.com",  # NECO utility-related domain
                        "unitil.com",
                        "nationalgrid.com",
                        "us.ngrid.com",
                        "ngrid.com",
                    ]

                    # Also skip specific submission instruction emails
                    submission_emails = [
                        "intervaldatarequests@rienergy.com",
                        "epoadmin@eversource.com",
                    ]

                    if any(
                        utility_domain in domain for utility_domain in utility_domains
                    ):
                        email_validation_context += (
                            f"- {email} → UTILITY DOMAIN - IGNORED\n"
                        )
                        continue

                    if email.lower() in submission_emails:
                        email_validation_context += (
                            f"- {email} → SUBMISSION INSTRUCTION EMAIL - IGNORED\n"
                        )
                        continue

                    # Check for broker domains
                    is_broker_domain = any(
                        broker_domain in domain for broker_domain in broker_domains
                    )
                    has_energy_keyword = any(
                        keyword in domain for keyword in energy_keywords
                    )

                    if is_broker_domain or has_energy_keyword:
                        broker_emails.append(email)
                        email_validation_context += (
                            f"- {email} → BROKER DOMAIN - INVALID for New England\n"
                        )

                # Simple rule: Any broker emails = reject
                if broker_emails:
                    email_validation_context += f"\nANALYSIS: Found {len(broker_emails)} broker email domains - REJECT for New England Region\n"
                    email_validation_context += "REASON: New England LOAs must be signed by the customer, not by a broker.\n"
                    email_validation_context += (
                        f"BROKER EMAILS DETECTED: {', '.join(broker_emails)}\n"
                    )
                else:
                    email_validation_context += (
                        "\nANALYSIS: No broker email domains found - PASS\n"
                    )

            elif is_illinois:
                email_validation_context = (
                    f"EMAIL DOMAIN VALIDATION ANALYSIS (ILLINOIS - Third-party brokers ALLOWED):\n"
                    f"Found email addresses in document: {email_matches}\n"
                    f"\n"
                    f"ILLINOIS EMAIL VALIDATION RULES:\n"
                    f"- Constellation domains: @constellation.com, @constellationenergy.com, @retailoperations.com (CRES provider)\n"
                    f"- Third-party broker domains: ANY domain is acceptable (e.g., @berryglob.com, @energycx.com, @energylink.com)\n"
                    f"- Illinois LOAs allow authorized agents/brokers to sign on behalf of customer\n"
                    f"- If document has BOTH Constellation email AND another email, the other email is likely the authorized agent\n"
                    f"- IMPORTANT: Do NOT reject for non-Constellation domains in Illinois LOAs\n"
                    f"\n"
                    f"EMAIL CLASSIFICATION:\n"
                )

                constellation_emails = []
                broker_emails = []

                for email in email_matches:
                    domain = email.split("@")[1] if "@" in email else "unknown"
                    is_constellation_domain = any(
                        constellation_domain in domain.lower()
                        for constellation_domain in [
                            "constellation",
                            "retailoperations",
                        ]
                    )

                    if is_constellation_domain:
                        constellation_emails.append(email)
                        email_validation_context += (
                            f"- {email} → CRES Provider (Constellation domain)\n"
                        )
                    else:
                        broker_emails.append(email)
                        email_validation_context += f"- {email} → Authorized Agent/Broker (ALLOWED in Illinois)\n"

                if constellation_emails and broker_emails:
                    email_validation_context += f"\nINTERPRETATION: Document shows Constellation as CRES provider and {broker_emails[0].split('@')[1]} as authorized agent - VALID for Illinois\n"
                elif broker_emails and not constellation_emails:
                    email_validation_context += "\nINTERPRETATION: Third-party broker/agent email present - VALID for Illinois LOAs\n"

            else:
                # Non-Illinois states - original validation logic
                email_validation_context = (
                    f"EMAIL DOMAIN VALIDATION ANALYSIS:\n"
                    f"Found email addresses in document: {email_matches}\n"
                    f"\n"
                    f"DOMAIN VALIDATION RULES:\n"
                    f"- ACCEPT: @constellation.com, @constellationenergy.com, @retailoperations.com\n"
                    f"- REJECT: @exelon.com, @exeloncorp.com, @strategic.com, @integrys.com, @pepco.com\n"
                    f"\n"
                    f"VALIDATION RESULT:\n"
                )
                for email in email_matches:
                    domain = email.split("@")[1] if "@" in email else "unknown"
                    is_constellation_domain = any(
                        constellation_domain in domain.lower()
                        for constellation_domain in [
                            "constellation",
                            "retailoperations",
                        ]
                    )
                    email_validation_context += f"- Email: {email} → Domain: @{domain} → {'ACCEPT (Constellation domain)' if is_constellation_domain else 'REJECT (Non-Constellation domain)'}\n"

        # Check for broker signatures (excluding audit trail sections)
        broker_signature_patterns = [
            r"on behalf of",
            r"for and on behalf of",
            # r'as agent for', New England Uses agent for non broker representative
            r"authorized agent",
            r"energy consultant",
            r"consultant",
            r"broker",
            r"utilities group",
            r"energy group",
            r"power group",
        ]

        # Check for New England specific authorized agent terms that are valid (not broker signatures)
        ne_agent_terms = [
            r"agent for customer",
            r"agent for the customer",
            r"customer\'s agent",
            r"customer representative",
            r"authorized representative",
            r"duly authorized",
            r"authorized to execute",
        ]

        # Remove audit trail sections from text before checking for broker patterns
        # Audit trail sections typically contain metadata about document processing
        audit_trail_patterns = [
            r"Audit trail.*?(?=\n\n|\Z)",
            r"Document History.*?(?=\n\n|\Z)",
            r"Sent for signature.*?(?=\n|\Z)",
            r"Viewed by.*?(?=\n|\Z)",
            r"Signed by.*?(?=\n|\Z)",
            r"The document has been completed.*?(?=\n|\Z)",
            r"Powered by.*?(?=\n|\Z)",
            r"Dropbox Sign.*?(?=\n|\Z)",
            r"from\s+[^\s]+@[^\s]+.*?(?=\n|\Z)",
        ]

        # Create a copy of extracted text without audit trail sections
        text_without_audit_trail = extracted_text
        for pattern in audit_trail_patterns:
            text_without_audit_trail = re.sub(
                pattern, "", text_without_audit_trail, flags=re.IGNORECASE | re.DOTALL
            )

        broker_signature_found = any(
            re.search(pattern, text_without_audit_trail, re.IGNORECASE)
            for pattern in broker_signature_patterns
        )

        # CRITICAL: Check validation issues AFTER GPT-4o fallback scenarios have completed
        # This ensures we use the updated extraction_log with GPT-4o results

        # Add interval data validation note if interval_needed is False
        if not self.interval_needed:
            extracted_text += "\n\n[SYSTEM NOTE: INTERVAL DATA VALIDATION IS DISABLED. ANY MISSING INTERVAL DATA SPECIFICATIONS SHOULD NOT CAUSE REJECTION.]\n\n"

        # Re-get the updated values after GPT-4o fallback
        updated_selection_marks = extraction_log.get("selection_marks", [])
        updated_initial_boxes = extraction_log.get("initial_boxes", [])
        updated_potential_initials = extraction_log.get("potential_initials", [])

        # Check if this is a FirstEnergy UDC - needed for conditional validation
        is_firstenergy_udc = self.provided_udc and self.provided_udc.upper() in [
            "CEI",
            "OE",
            "TE",
        ]

        # CRITICAL: For FirstEnergy documents, use ONLY GPT-4o comprehensive validation results
        # Do NOT use the old Azure OCR-based initial box detection
        # FirstEnergy validation is handled entirely in Layer 3 (lines 2435-2496)

        validation_issues = []

        # Only apply old-style initial box validation for NON-FirstEnergy Ohio documents
        # FirstEnergy documents have their own comprehensive GPT-4o validation
        if detected_state.upper() == "OH" and not is_firstenergy_udc:
            # Check if no initial boxes exist at all (AFTER GPT-4o fallback)
            filled_initial_boxes = [
                box for box in updated_initial_boxes if box.get("is_filled", False)
            ]

            # CRITICAL FIX: Only consider "no initial boxes" as a problem if NO initials are found anywhere
            # If potential_initials are found (even if they spilled outside the box), don't reject for "no initial boxes"
            has_any_initials = len(updated_potential_initials) > 0
            len(filled_initial_boxes) == 0 and not has_any_initials

            # Re-check for X marks with updated data - FIXED: More precise X mark detection with confidence threshold
            # Only consider it an X mark if it's clearly just "X" and meets confidence threshold
            x_marks_found = []

            # Pre-check: Look for X marks followed by signature/date or near Signature/Date labels in extracted text
            # If we find "X :selected:" followed by Signature/Date, skip ALL X mark validation
            # This indicates the form uses X marks as valid selection indicators for signature/date fields
            signature_date_x_patterns = [
                r"X\s+:selected:[^\n]*(?:Signature|Date)",  # X followed by signature/date on same line
                r"X\s+:selected:[^\n]*\n\s*(?:Signature|Date)",  # X followed by signature/date on next line
            ]
            has_signature_date_x_marks = any(
                re.search(pattern, extracted_text, re.IGNORECASE | re.MULTILINE)
                for pattern in signature_date_x_patterns
            )

            # If X marks are used for signature/date fields, skip X mark validation entirely
            # These forms may legitimately use X marks throughout
            if not has_signature_date_x_marks:
                for initial in updated_potential_initials:
                    text = initial.get("text", "").strip().upper()
                    context = initial.get("context", "").lower()
                    confidence = initial.get(
                        "confidence", 1.0
                    )  # Default to 1.0 if no confidence provided

                    # Check if context indicates this is a signature/date field marker (not an initial box)
                    is_signature_or_date_field = (
                        "signature" in context
                        or "date" in context
                        or ":selected:" in context
                    )

                    # Check if context indicates this is in an initial box area
                    is_initial_box_area = (
                        "initial" in context.lower() or "box" in context.lower()
                    )

                    # Only flag as X mark if:
                    # 1. Text is exactly "X" (not part of a larger word)
                    # 2. Not in context that suggests it's part of a word (like "EXAMPLE", "EXACT", etc.)
                    # 3. Context suggests it's actually in an initial box area (not signature/date field)
                    # 4. Confidence meets or exceeds the threshold (95% by default)
                    if (
                        text == "X"
                        and "example" not in context
                        and "exact" not in context
                        and "text" not in context
                        and "express" not in context
                        and "exit" not in context
                        and not is_signature_or_date_field  # Exclude signature/date field markers
                        and is_initial_box_area  # Must be in initial box area
                        and confidence >= self.x_mark_confidence_threshold
                    ):  # Only flag high-confidence X marks
                        x_marks_found.append(initial)

            # Count selected and unselected marks with updated data
            selected_marks = [
                mark
                for mark in updated_selection_marks
                if mark.get("state") == "selected"
            ]
            unselected_marks = [
                mark
                for mark in updated_selection_marks
                if mark.get("state") == "unselected"
            ]

            # For non-FirstEnergy Ohio LOAs: Check initial box/initial requirements
            # If initial boxes exist, they must be filled
            # If NO initial boxes exist, letter initials must be present
            if len(updated_initial_boxes) > 0:
                # Initial boxes exist - check if they're empty/unselected
                if unselected_marks and not has_any_initials:
                    validation_issues.append(
                        f"Found {len(unselected_marks)} empty/unselected initial boxes with no initials detected"
                    )
            else:
                # No initial boxes exist - check if letter initials are present
                if not has_any_initials:
                    validation_issues.append(
                        "No initial boxes or letter initials found in the document"
                    )

            # For non-FirstEnergy Ohio LOAs: Flag X marks as validation issues
            if x_marks_found:
                validation_issues.append(
                    f"Found {len(x_marks_found)} 'X' marks in initial boxes. Ohio LOAs require LETTER INITIALS (not X marks)"
                )

        # CRITICAL: Build initial override context for Ohio documents BEFORE validation context
        # This ensures GPT knows about detected initials even if no code-level issues found
        initial_override_context = ""
        if detected_state.upper() == "OH":
            # Build list of detected initials for display
            initials_list = [
                f"{init.get('text', 'Unknown')}"
                for init in updated_potential_initials
                if init.get("text", "").strip()
            ]
            initials_display = (
                ", ".join(initials_list) if initials_list else "None detected"
            )

            # Check if valid letter initials were detected
            has_valid_initials = len(initials_list) > 0 and not any(
                "X" == init.get("text", "").strip().upper()
                for init in updated_potential_initials
            )

            # Build initial override message if valid initials detected
            if has_valid_initials:
                initial_override_context = (
                    f"\n{'#'*120}\n"
                    f"### MANDATORY OVERRIDE - VALID LETTER INITIALS DETECTED ###\n"
                    f"{'#'*120}\n"
                    f"**CRITICAL INSTRUCTION:**\n"
                    f"Code-level analysis has detected VALID LETTER INITIALS: {initials_display}\n"
                    f"These are LETTER initials (not X marks), which are ACCEPTABLE for Ohio LOAs.\n"
                    f"**YOU MUST NOT REJECT for 'Initial boxes not filled' or 'Missing initials'.**\n"
                    f"Valid letter initials have been confirmed by code-level detection.\n"
                    f"{'#'*120}\n\n"
                )

        # CRITICAL: Construct initial_validation_context AFTER GPT-4o fallback results are applied
        # Add initial_override_context at the beginning if it exists
        # Now construct the context with the corrected validation issues
        if validation_issues:
            # Build list of detected initials for display
            initials_list = [
                f"{init.get('text', 'Unknown')}"
                for init in updated_potential_initials
                if init.get("text", "").strip()
            ]
            initials_display = (
                ", ".join(initials_list) if initials_list else "None detected"
            )

            # If this is an Ohio document, include the Ohio-specific initial box validation rules
            if detected_state.upper() == "OH":
                initial_validation_context = (
                    f"{initial_override_context}"  # Add override at the beginning
                    f"INITIAL BOX AND SIGNATURE VALIDATION ANALYSIS:\n"
                    f"Selection Marks Analysis (AFTER GPT-4o fallback):\n"
                    f"- Total selection marks: {len(updated_selection_marks)}\n"
                    f"- Selected marks: {len(selected_marks)}\n"
                    f"- Unselected marks: {len(unselected_marks)}\n"
                    f"- X marks detected: {len(x_marks_found)}\n"
                    f"- Initial boxes detected: {len(updated_initial_boxes)}\n"
                    f"- Filled initial boxes: {len(filled_initial_boxes)}\n"
                    f"- Detected initials: {initials_display}\n"
                    f"\n"
                    f"Broker Signature Analysis:\n"
                    f"- Broker signature patterns found: {broker_signature_found}\n"
                    f"\n"
                    f"VALIDATION ISSUES FOUND:\n"
                    f"{chr(10).join(f'- {issue}' for issue in validation_issues)}\n"
                    f"\n"
                    f"OHIO-SPECIFIC VALIDATION RULES:\n"
                    f'1. "X" marks are NOT valid initials and should be REJECTED (OHIO REQUIREMENT)\n'
                    f'2. Valid initials must be letters (A-Z), not "X" marks or check marks (OHIO REQUIREMENT)\n'
                    f"3. If initial boxes exist, they must be filled (OHIO REQUIREMENT)\n"
                    f"4. If no initial boxes exist, letter initials must be present somewhere in the document (OHIO REQUIREMENT)\n"
                    f"5. Document must be signed by customer, not by broker/third-party\n"
                    f"\n"
                    f"CRITICAL: These initial box requirements are ONLY applicable to Ohio LOAs, not New England or Illinois LOAs.\n"
                    f'If ANY of these issues exist, Required Checkboxes/Initials status = "FAIL" and Broker Authorization status = "FAIL"\n'
                )
            else:
                # For non-Ohio documents, only include broker signature analysis if needed
                initial_validation_context = (
                    f"BROKER SIGNATURE VALIDATION ANALYSIS:\n"
                    f"- Broker signature patterns found: {broker_signature_found}\n"
                    f"\n"
                    f"VALIDATION ISSUES FOUND:\n"
                    f"{chr(10).join(f'- {issue}' for issue in validation_issues)}\n"
                    f"\n"
                    f"VALIDATION RULES:\n"
                    f"1. Document must be signed by customer, not by broker/third-party\n"
                    f"\n"
                    f"NOTE: Initial box requirements (must exist, can't be X marks, etc.) only apply to Ohio LOAs.\n"
                    f"New England LOAs are NOT subject to these requirements.\n"
                )
        else:
            # No validation issues found, but still need to add override context if it exists
            initial_validation_context = initial_override_context

        if extraction_log.get("beco_signature_detection", {}).get("gpt4o_verified"):
            sig_details = extraction_log["beco_signature_detection"]
            if sig_details.get("customer_signature_present"):
                override_text = "\n\n" + "#" * 120 + "\n"
                override_text += (
                    "### MANDATORY OVERRIDE - CUSTOMER SIGNATURE IS PRESENT ###\n"
                )
                override_text += "#" * 120 + "\n\n"
                override_text += "**CRITICAL INSTRUCTION:**\n"
                override_text += "The GPT-4o Vision model has confirmed a customer signature is present.\n"
                override_text += f"   - Signature Text Found: '{sig_details.get('customer_signature_text', 'N/A')}'\n"
                override_text += "   - STATUS: SIGNATURE IS PRESENT\n\n"
                override_text += (
                    "**YOU MUST NOT REJECT FOR 'Customer Signature: Missing'.**\n"
                )
                override_text += (
                    "This has been verified by a separate vision analysis step.\n\n"
                )
                override_text += "#" * 120 + "\n\n"
                initial_validation_context = override_text + initial_validation_context

        # Add COMED signature verification override
        if extraction_log.get("comed_signature_detection", {}).get("gpt4o_verified"):
            sig_details = extraction_log["comed_signature_detection"]
            override_text = "\n\n" + "#" * 120 + "\n"
            override_text += (
                "### MANDATORY OVERRIDE - COMED SIGNATURE VERIFICATION COMPLETE ###\n"
            )
            override_text += "#" * 120 + "\n\n"
            override_text += "**CRITICAL INSTRUCTION:**\n"
            override_text += (
                "GPT-4o Vision has analyzed the COMED document signature field:\n"
            )
            override_text += f"   - Customer Signature Present: {sig_details.get('customer_signature_present')}\n"
            override_text += f"   - Signature Text: '{sig_details.get('customer_signature_text', 'None')}'\n\n"

            if sig_details.get("customer_signature_present"):
                override_text += "   - RESULT: SIGNATURE IS PRESENT ✓\n\n"
                override_text += "**YOU MUST NOT REJECT for 'COMED: Customer signature is missing'**\n"
                override_text += "Signature has been verified by GPT-4o Vision.\n\n"
            else:
                override_text += "   - RESULT: SIGNATURE IS MISSING ✗\n\n"
                override_text += (
                    "**YOU MUST REJECT for 'COMED: Customer signature is missing'**\n"
                )
                override_text += (
                    "GPT-4o Vision confirmed no actual signature is present.\n"
                )
                override_text += (
                    "Only field label detected, not an actual signature.\n\n"
                )

            override_text += "#" * 120 + "\n\n"
            initial_validation_context = override_text + initial_validation_context

        # Only check for broker signature if not an authorized person
        authorized_person_patterns = [
            r"authorized person",
            r"authorized representative",
            r"authorized signatory",
            r"Authorized Person/Title:",
        ]

        authorized_person_found = any(
            re.search(pattern, extracted_text, re.IGNORECASE)
            for pattern in authorized_person_patterns
        )

        # Check if any New England agent terms are found (these are legitimate authorization, not broker)
        if self.region == "New England":
            any(
                re.search(pattern, text_without_audit_trail, re.IGNORECASE)
                for pattern in ne_agent_terms
            )

        # If someone is listed as "Authorized Person", they are NOT a broker - they are customer's representative
        # CRITICAL: Illinois allows third-party broker authorization - do not add validation issue for IL
        # For New England, do NOT use pattern-based broker detection at all - rely solely on prompt/email detection
        # This is CRITICAL for proper New England broker detection which is based on email domains
        if (
            broker_signature_found
            and not authorized_person_found
            and detected_state != "IL"
            and self.region != "New England"
        ):  # Completely exclude New England region from pattern-based detection
            validation_issues.append(
                "Document signed by broker/third-party, not customer"
            )

        # Override broker detection if authorized person is clearly identified
        if authorized_person_found:
            broker_signature_found = (
                False  # Clear broker flag if authorized person is identified
            )

        if validation_issues:
            # If this is an Ohio document, include the Ohio-specific initial box validation rules
            if detected_state.upper() == "OH":
                initial_validation_context = (
                    f"INITIAL BOX AND SIGNATURE VALIDATION ANALYSIS:\n"
                    f"Selection Marks Analysis:\n"
                    f"- Total selection marks: {len(selection_marks)}\n"
                    f"- Selected marks: {len(selected_marks)}\n"
                    f"- Unselected marks: {len(unselected_marks)}\n"
                    f"- X marks detected: {len(x_marks_found)}\n"
                    f"\n"
                    f"Broker Signature Analysis:\n"
                    f"- Broker signature patterns found: {broker_signature_found}\n"
                    f"\n"
                    f"VALIDATION ISSUES FOUND:\n"
                    f"{chr(10).join(f'- {issue}' for issue in validation_issues)}\n"
                    f"\n"
                    f"OHIO-SPECIFIC VALIDATION RULES:\n"
                    f'1. "X" marks are NOT valid initials and should be REJECTED (OHIO REQUIREMENT)\n'
                    f'2. Valid initials must be letters (A-Z), not "X" marks or check marks (OHIO REQUIREMENT)\n'
                    f"3. If initial boxes exist, they must be filled (OHIO REQUIREMENT)\n"
                    f"4. If no initial boxes exist, letter initials must be present somewhere in the document (OHIO REQUIREMENT)\n"
                    f"5. Document must be signed by customer, not by broker/third-party\n"
                    f"\n"
                    f"CRITICAL: These initial box requirements are ONLY applicable to Ohio LOAs, not New England or Illinois LOAs.\n"
                    f'If ANY of these issues exist, Required Checkboxes/Initials status = "FAIL" and Broker Authorization status = "FAIL"\n'
                )
            else:
                # For non-Ohio documents, only include broker signature analysis if needed
                initial_validation_context = (
                    f"BROKER SIGNATURE VALIDATION ANALYSIS:\n"
                    f"- Broker signature patterns found: {broker_signature_found}\n"
                    f"\n"
                    f"VALIDATION ISSUES FOUND:\n"
                    f"{chr(10).join(f'- {issue}' for issue in validation_issues)}\n"
                    f"\n"
                    f"VALIDATION RULES:\n"
                    f"1. Document must be signed by customer, not by broker/third-party\n"
                    f"\n"
                    f"NOTE: Initial box requirements (must exist, can't be X marks, etc.) only apply to Ohio LOAs.\n"
                    f"New England LOAs are NOT subject to these requirements.\n"
                )

        # Add signature validity and expiration calculation to the context
        signature_validity_context = ""
        if signature_validity_result:
            signature_validity_context = (
                f"SIGNATURE DATE VALIDATION CALCULATION RESULT:\n"
                f"{signature_validity_result['calculation_details']}\n"
                f"VALIDATION RESULT:\n"
                f"- Is Valid: {'YES' if signature_validity_result['is_valid'] else 'NO'}\n"
                f"- Reason: {signature_validity_result['reason']}\n"
                f"- Signature Date: {signature_validity_result.get('signature_date', 'N/A')}\n"
                f"- Today's Date: {signature_validity_result.get('today_date', 'N/A')}\n"
                f"- Days Old: {signature_validity_result.get('days_old', 'N/A')}\n"
                f"- Months Old: {signature_validity_result.get('months_old', 'N/A')}\n"
                f"- Years Old: {signature_validity_result.get('years_old', 'N/A')}\n"
                f"- State Limit: {signature_validity_result.get('state_limit', 'N/A')} months\n"
                f"\n"
                f"CRITICAL: Use this calculation result for Customer Signature Requirements validation.\n"
                f'If Is Valid = NO, then Customer Signature Requirements status = "FAIL"\n'
                f'If Is Valid = YES, then Customer Signature Requirements status = "PASS"\n'
            )

        # Add expiration information to context
        if loa_expiration_result:
            expiration_context = (
                f"\n"
                f"LOA EXPIRATION CALCULATION RESULT:\n"
                f"{loa_expiration_result.get('calculation_details', 'No details available')}\n"
                f"EXPIRATION STATUS:\n"
                f"- Expiration Date: {loa_expiration_result.get('expiration_date_formatted', 'N/A')}\n"
                f"- Days Until Expiration: {loa_expiration_result.get('days_until_expiration', 'N/A')}\n"
                f"- Months Until Expiration: {loa_expiration_result.get('months_until_expiration', 'N/A')}\n"
                f"- Is Expired: {'YES' if loa_expiration_result.get('is_expired', False) else 'NO'}\n"
                f"- Status: {'EXPIRED' if loa_expiration_result.get('is_expired', False) else 'ACTIVE'}\n"
                f"\n"
                f"CRITICAL: If Is Expired = YES or Status = EXPIRED, the LOA MUST BE REJECTED.\n"
                f"Rejection reason: LOA expired on {loa_expiration_result.get('expiration_date_formatted', 'N/A')}\n"
            )
            signature_validity_context += expiration_context

        # CRITICAL FIX: If customer date was successfully extracted, add MANDATORY validation bypass
        # This MUST override ALL other validation logic - the date EXISTS and was extracted
        skip_customer_date_missing_rejection = False
        if extraction_log.get(
            "customer_date_extraction_success"
        ) and extraction_log.get("customer_date_extracted_by_gpt4o"):
            extracted_customer_date = extraction_log["customer_date_extracted_by_gpt4o"]
            skip_customer_date_missing_rejection = True

            # ULTRA-STRONG BYPASS INSTRUCTION - IMPOSSIBLE TO IGNORE
            signature_validity_context += f"\n\n{'#' * 120}\n"
            signature_validity_context += f"{'#' * 120}\n"
            signature_validity_context += (
                "###  MANDATORY OVERRIDE - CUSTOMER SIGNATURE DATE IS PRESENT  ###\n"
            )
            signature_validity_context += f"{'#' * 120}\n"
            signature_validity_context += f"{'#' * 120}\n\n"

            signature_validity_context += "**CRITICAL INSTRUCTION - THIS OVERRIDES ALL OTHER VALIDATION RULES:**\n\n"
            signature_validity_context += "The customer signature date HAS BEEN SUCCESSFULLY EXTRACTED by GPT-4o Vision:\n"
            signature_validity_context += (
                f"   ► EXTRACTED DATE: {extracted_customer_date}\n"
            )
            signature_validity_context += "   ► CONFIDENCE: 100%\n"
            signature_validity_context += f"   ► USED FOR EXPIRATION: {loa_expiration_result.get('expiration_date_formatted', 'N/A') if loa_expiration_result else 'N/A'}\n"
            signature_validity_context += "   ► STATUS: DATE EXISTS AND IS VALID\n\n"

            signature_validity_context += (
                "**YOU EXTRACTED THIS DATE YOURSELF IN THE PREVIOUS STEP**\n\n"
            )

            signature_validity_context += "**ABSOLUTE REQUIREMENT - NO EXCEPTIONS:**\n"
            signature_validity_context += "DO NOT - under any circumstances - include ANY of these rejection reasons:\n"
            signature_validity_context += "   ✗ 'Customer signature date is missing'\n"
            signature_validity_context += "   ✗ 'Customer date is missing'\n"
            signature_validity_context += "   ✗ 'Missing customer signature date'\n"
            signature_validity_context += "   ✗ 'No customer date found'\n"
            signature_validity_context += "   ✗ 'Customer signature missing'\n"
            signature_validity_context += "   ✗ 'Customer date missing'\n"
            signature_validity_context += (
                "   ✗ Any variation stating the customer date is missing\n"
            )
            signature_validity_context += "   ✗ Any phrase containing both 'customer' AND 'missing' AND 'date'\n\n"

            signature_validity_context += "**WHY THIS OVERRIDE EXISTS:**\n"
            signature_validity_context += f"You already extracted the date '{extracted_customer_date}' successfully.\n"
            signature_validity_context += (
                "The date was used to calculate an expiration date.\n"
            )
            signature_validity_context += (
                "Therefore, by definition, the date is NOT missing.\n"
            )
            signature_validity_context += "It is logically impossible for a date to be both extracted AND missing.\n\n"

            signature_validity_context += "**IF THE CUSTOMER SIGNATURE IS MISSING:**\n"
            signature_validity_context += "You MAY reject for 'Customer signature is missing' (the signature itself)\n"
            signature_validity_context += (
                "But you MUST NOT reject for the date being missing\n"
            )
            signature_validity_context += "These are two separate things:\n"
            signature_validity_context += (
                "   - Signature can be missing (physical signature/name)\n"
            )
            signature_validity_context += f"   - Date can exist (the date field is filled: {extracted_customer_date})\n\n"

            signature_validity_context += f"{'#' * 120}\n"
            signature_validity_context += f"END OF MANDATORY OVERRIDE - CUSTOMER DATE EXISTS: {extracted_customer_date}\n"
            signature_validity_context += f"{'#' * 120}\n\n"

        # CRITICAL FIX: Extract requestor/billing signature date using GPT-4o (SAME PATTERN AS CUSTOMER DATE)
        # SKIP if comprehensive fallback already ran for BECO and extracted the date
        requestor_signature_date_from_gpt4o = None
        if pdf_path and not extraction_log.get("requestor_date_extracted_by_gpt4o"):
            try:
                self.logger.info("Using GPT-4o to extract requestor signature date...")
                gpt4o_requestor_result = self.gpt4o_verification_integration.extract_requestor_signature_date_with_gpt4o(
                    pdf_path
                )

                if gpt4o_requestor_result.get("success") and gpt4o_requestor_result.get(
                    "requestor_signature_date"
                ):
                    requestor_signature_date_from_gpt4o = gpt4o_requestor_result[
                        "requestor_signature_date"
                    ]
                    self.logger.info(
                        f"GPT-4o extracted requestor signature date: {requestor_signature_date_from_gpt4o}"
                    )

                    # Store the successfully extracted date for validation bypass
                    extraction_log["requestor_date_extracted_by_gpt4o"] = (
                        requestor_signature_date_from_gpt4o
                    )
                    extraction_log["requestor_date_extraction_success"] = True

                    # Add MANDATORY validation bypass instruction for requestor date
                    signature_validity_context += f"\n\n{'*' * 100}\n"
                    signature_validity_context += "MANDATORY VALIDATION BYPASS - REQUESTOR/BILLING DATE SUCCESSFULLY EXTRACTED\n"
                    signature_validity_context += f"{'*' * 100}\n"
                    signature_validity_context += f"GPT-4o Vision Model extracted requestor/billing date: {requestor_signature_date_from_gpt4o}\n"
                    signature_validity_context += f"- Location: {gpt4o_requestor_result.get('location_description', 'Requestor & Billing Information section')}\n"
                    signature_validity_context += f"- Confidence: {gpt4o_requestor_result.get('confidence', 100)}%\n"
                    signature_validity_context += (
                        "- STATUS: DATE IS PRESENT (NOT MISSING)\n"
                    )
                    signature_validity_context += "\nMANDATORY INSTRUCTION:\n"
                    signature_validity_context += "Since this date was SUCCESSFULLY EXTRACTED from the document,\n"
                    signature_validity_context += "you MUST NOT include 'Requestor/Billing signature date is missing' in rejection reasons.\n"
                    signature_validity_context += (
                        "The date IS PRESENT and has been VERIFIED.\n"
                    )
                    signature_validity_context += f"{'*' * 100}\n\n"
                else:
                    self.logger.warning(
                        "GPT-4o requestor signature date extraction failed or returned no date"
                    )
                    extraction_log["requestor_date_extraction_success"] = False
            except Exception as e:
                self.logger.error(
                    f"GPT-4o requestor signature date extraction error: {str(e)}"
                )
                extraction_log["requestor_date_extraction_success"] = False

        # Build requestor/billing validation override context if data was extracted by GPT-4o
        requestor_billing_override_context = ""
        if extraction_log.get(
            "requestor_billing_extraction_success"
        ) and extraction_log.get("requestor_billing_extracted_by_gpt4o"):
            rb_data = extraction_log["requestor_billing_extracted_by_gpt4o"]

            requestor_billing_override_context = "\n\n" + "#" * 120 + "\n"
            requestor_billing_override_context += "### MANDATORY OVERRIDE - REQUESTOR/BILLING INFORMATION IS PRESENT ###\n"
            requestor_billing_override_context += "#" * 120 + "\n\n"
            requestor_billing_override_context += "**CRITICAL INSTRUCTION:**\n"
            requestor_billing_override_context += "The GPT-4o Vision model has successfully extracted ALL requestor/billing fields.\n\n"
            requestor_billing_override_context += (
                "**EXTRACTED REQUESTOR/BILLING FIELDS:**\n"
            )

            for field, value in rb_data.items():
                if value:
                    requestor_billing_override_context += f"   ✓ {field}: {value}\n"

            requestor_billing_override_context += (
                "\n**ABSOLUTE REQUIREMENT - NO EXCEPTIONS:**\n"
            )
            requestor_billing_override_context += (
                "DO NOT reject for ANY of these reasons:\n"
            )

            for field in rb_data.keys():
                if rb_data[field]:
                    requestor_billing_override_context += f"   ✗ '{field} is missing'\n"
                    requestor_billing_override_context += (
                        f"   ✗ '{field} field is missing'\n"
                    )

            requestor_billing_override_context += (
                "   ✗ Any variation stating requestor/billing fields are missing\n\n"
            )

            requestor_billing_override_context += "**WHY THIS OVERRIDE EXISTS:**\n"
            requestor_billing_override_context += "These fields were extracted in a previous GPT-4o Vision analysis step.\n"
            requestor_billing_override_context += (
                "Therefore, by definition, they are NOT missing.\n"
            )
            requestor_billing_override_context += "It is logically impossible for fields to be both extracted AND missing.\n\n"

            requestor_billing_override_context += "#" * 120 + "\n"
            requestor_billing_override_context += (
                "END OF MANDATORY OVERRIDE - REQUESTOR/BILLING FIELDS ARE PRESENT\n"
            )
            requestor_billing_override_context += "#" * 120 + "\n\n"

        # Load user prompt from markdown file
        user_prompt = self._load_user_prompt(
            document_id=document_id,
            extracted_text_length=len(extracted_text),
            extraction_success=extraction_log["extraction_success"],
            current_date=datetime.now().strftime("%m/%d/%Y"),
            email_validation_context=email_validation_context,
            initial_validation_context=initial_validation_context,
            extracted_text=extracted_text,
            layout_context=layout_context,
            signature_validity_context=signature_validity_context
            + requestor_billing_override_context,
            potential_initials_count=len(potential_initials),
            detected_state=detected_state,
            selection_marks_count=len(selection_marks),
            key_value_pairs_count=len(key_value_pairs),
            provided_udc=self.provided_udc if self.provided_udc else "Not provided",
        )

        # CRITICAL: Add final validation override at the VERY END of the prompt
        # This ensures it's the LAST thing GPT-4o sees before making its decision
        final_override = ""

        # Add ALL FirstEnergy validation issues to final override
        # GPT-4o ignores issues placed in the middle of the prompt - they MUST be at the end
        if extraction_log.get("firstenergy_code_level_validation_issues"):
            fe_issues = extraction_log["firstenergy_code_level_validation_issues"]

            if fe_issues:
                final_override += "\n\n" + "=" * 120 + "\n"
                final_override += (
                    "FINAL VALIDATION OVERRIDE - FIRSTENERGY ISSUES - READ THIS LAST\n"
                )
                final_override += "=" * 120 + "\n\n"
                final_override += "**MANDATORY: The following FirstEnergy validation issues MUST be included in rejectionReasons:**\n\n"
                for i, issue in enumerate(fe_issues, 1):
                    final_override += f"{i}. {issue}\n"
                final_override += (
                    "\n**YOU MUST INCLUDE ALL OF THESE ISSUES IN YOUR RESPONSE**\n"
                )
                final_override += "=" * 120 + "\n\n"

        # Add ALL AEP validation issues to final override
        # GPT-4o ignores issues placed in the middle of the prompt - they MUST be at the end
        if extraction_log.get("aep_code_level_validation_issues"):
            aep_issues = extraction_log["aep_code_level_validation_issues"]

            if aep_issues:
                final_override += "\n\n" + "=" * 120 + "\n"
                final_override += (
                    "FINAL VALIDATION OVERRIDE - AEP ISSUES - READ THIS LAST\n"
                )
                final_override += "=" * 120 + "\n\n"
                final_override += "**MANDATORY: The following AEP validation issues MUST be included in rejectionReasons:**\n\n"
                for i, issue in enumerate(aep_issues, 1):
                    final_override += f"{i}. {issue}\n"
                final_override += (
                    "\n**YOU MUST INCLUDE ALL OF THESE ISSUES IN YOUR RESPONSE**\n"
                )
                final_override += "=" * 120 + "\n\n"

        if extraction_log.get("customer_date_extraction_success"):
            final_override += "\n\n" + "=" * 120 + "\n"
            final_override += (
                "FINAL VALIDATION OVERRIDE - READ THIS LAST BEFORE RESPONDING\n"
            )
            final_override += "=" * 120 + "\n\n"
            final_override += f"CUSTOMER SIGNATURE DATE: {extraction_log.get('customer_date_extracted_by_gpt4o')}\n"
            final_override += "STATUS: ✓ PRESENT (Successfully extracted)\n\n"
            final_override += "DO NOT include 'Customer signature date is missing' in rejectionReasons\n"
            final_override += "=" * 120 + "\n\n"

        if extraction_log.get("requestor_date_extraction_success"):
            final_override += "\n\n" + "=" * 120 + "\n"
            final_override += (
                "FINAL VALIDATION OVERRIDE - READ THIS LAST BEFORE RESPONDING\n"
            )
            final_override += "=" * 120 + "\n\n"
            final_override += f"REQUESTOR/BILLING SIGNATURE DATE: {extraction_log.get('requestor_date_extracted_by_gpt4o')}\n"
            final_override += "STATUS: ✓ PRESENT (Successfully extracted)\n\n"
            final_override += "DO NOT include 'Requestor/Billing signature date is missing' in rejectionReasons\n"
            final_override += "=" * 120 + "\n\n"

        if final_override:
            user_prompt += final_override

        # Remove audit trail sections from extracted text before sending to GPT
        audit_trail_patterns = [
            r"Audit trail.*?(?=\n\n|\Z)",
            r"Document History.*?(?=\n\n|\Z)",
            r"Sent for signature.*?(?=\n|\Z)",
            r"Viewed by.*?(?=\n|\Z)",
            r"Signed by.*?(?=\n|\Z)",
            r"The document has been completed.*?(?=\n|\Z)",
            r"Powered by.*?(?=\n|\Z)",
            r"Dropbox Sign.*?(?=\n|\Z)",
            r"from\s+[^\s]+@[^\s]+.*?(?=\n|\Z)",
        ]

        # Create cleaned text without audit trail sections
        cleaned_extracted_text = extracted_text
        for pattern in audit_trail_patterns:
            cleaned_extracted_text = re.sub(
                pattern, "", cleaned_extracted_text, flags=re.IGNORECASE | re.DOTALL
            )

        # Update the user prompt to use cleaned text
        user_prompt = user_prompt.replace(
            f"FULL DOCUMENT TEXT:\n{extracted_text}",
            f"FULL DOCUMENT TEXT:\n{cleaned_extracted_text}",
        )

        # Apply enhanced selection validation if available
        if hasattr(self, "enhanced_selection_validator"):
            try:
                # Use enhanced selection validation for better checkbox analysis
                enhanced_validation_results = (
                    self.enhanced_selection_validator.validate_selection_marks(
                        selection_marks, extracted_text
                    )
                )
                # Add enhanced validation results to layout context
                layout_context += "\nENHANCED SELECTION VALIDATION:\n"
                layout_context += f"X-marks detected: {enhanced_validation_results.get('x_marks_found', 0)}\n"
                layout_context += f"Valid initials detected: {enhanced_validation_results.get('valid_initials_found', 0)}\n"
                layout_context += f"Empty boxes detected: {enhanced_validation_results.get('empty_boxes_found', 0)}\n"
            except Exception as e:
                # Log error but continue with validation
                layout_context += f"\nENHANCED SELECTION VALIDATION ERROR: {str(e)}\n"

        # Get GPT-4o analysis using the provided OpenAI service with fallback mechanisms
        gpt_response = self._get_gpt4o_analysis_with_fallback(
            system_prompt, user_prompt
        )

        # If GPT-4o OCR fallback was used and verification is needed, apply verification
        if (
            hasattr(self, "gpt4o_verification_integration")
            and len(potential_initials) == 0
        ):
            try:
                # Use GPT-4o verification for critical checkboxes when initial detection fails
                verification_results = (
                    self.gpt4o_verification_integration.verify_critical_checkboxes(
                        extracted_text, selection_marks, document_id
                    )
                )
                if verification_results.get("critical_checkboxes_verified"):
                    layout_context += (
                        "\nGPT-4O VERIFICATION APPLIED: Critical checkboxes verified\n"
                    )
            except Exception as e:
                # Log error but continue
                layout_context += f"\nGPT-4O VERIFICATION ERROR: {str(e)}\n"

        # Note: Debug logging removed for production - available in experimental version

        try:
            # Extract JSON from response
            if "```json" in gpt_response:
                json_start = gpt_response.find("```json") + 7
                json_end = gpt_response.find("```", json_start)
                json_text = gpt_response[json_start:json_end].strip()
            else:
                json_text = gpt_response.strip()

            # Parse the simplified GPT response
            gpt_validation_result = json.loads(json_text)

            # CRITICAL SAFETY CHECK: If customer date was successfully extracted, remove false "customer date missing" rejections
            if skip_customer_date_missing_rejection:
                rejection_reasons = gpt_validation_result.get("rejectionReasons", [])

                # Filter out any rejection reasons containing "customer" AND "date" AND "missing"
                filtered_reasons = []
                for reason in rejection_reasons:
                    reason_lower = reason.lower()
                    # Check if this is a false "customer date missing" rejection
                    has_customer = "customer" in reason_lower
                    has_date = "date" in reason_lower
                    has_missing = "missing" in reason_lower

                    # Also check for "no date" patterns
                    has_no_date = (
                        "no date" in reason_lower or "no customer date" in reason_lower
                    )

                    if (has_customer and has_date and has_missing) or has_no_date:
                        # Skip this false rejection
                        self.logger.info(
                            f"FILTERED OUT false customer date rejection: {reason}"
                        )
                        continue

                    filtered_reasons.append(reason)

                # Update the validation result with filtered reasons
                gpt_validation_result["rejectionReasons"] = filtered_reasons

                # If all rejections were removed and status was REJECT, change to ACCEPT
                if len(filtered_reasons) == 0 and len(rejection_reasons) > 0:
                    gpt_validation_result["status"] = "ACCEPT"
                    self.logger.info(
                        "All rejection reasons were false customer date rejections - status changed to ACCEPT"
                    )

            # CRITICAL: Combine integrity rejection reasons with GPT-4o rejection reasons
            # Integrity issues must ALWAYS result in rejection
            integrity_rejection_reasons = extraction_log.get(
                "integrity_rejection_reasons", []
            )
            if integrity_rejection_reasons:
                self.logger.info(
                    f"Adding {len(integrity_rejection_reasons)} integrity rejection reason(s) to validation result"
                )

                # Get current rejection reasons from GPT
                current_rejections = gpt_validation_result.get("rejectionReasons", [])

                # Combine integrity reasons first (they're most critical), then GPT reasons
                combined_rejections = integrity_rejection_reasons + current_rejections

                # Update the validation result with combined reasons
                gpt_validation_result["rejectionReasons"] = combined_rejections

                # Force status to REJECT if integrity issues exist
                gpt_validation_result["status"] = "REJECT"
                self.logger.info(
                    f"Status forced to REJECT due to integrity issues. Total rejection reasons: {len(combined_rejections)}"
                )

            # =========================================================================
            # ACCOUNT NAME COMPARISON (Great Lakes Region Only)
            # =========================================================================
            # Perform account name comparison if parameter provided
            account_name_rejections = []

            if self.region == "Great Lakes" and self.account_name:
                self.logger.info(
                    "Performing account name comparison for Great Lakes Region..."
                )

                # Extract customer name from LOA if account_name parameter provided
                if pdf_path:
                    try:
                        self.logger.info(
                            f"Comparing account name: Salesforce='{self.account_name}'"
                        )
                        customer_name_result = self.gpt4o_verification_integration.extract_customer_name_from_great_lakes_loa(
                            pdf_path=pdf_path, udc=self.provided_udc
                        )

                        if customer_name_result.get(
                            "success"
                        ) and customer_name_result.get("customer_name"):
                            loa_customer_name = customer_name_result["customer_name"]
                            self.logger.info(
                                f"  LOA Customer Name: '{loa_customer_name}'"
                            )

                            # Compare names
                            name_comparison = self.gpt4o_verification_integration.compare_account_names(
                                salesforce_account_name=self.account_name,
                                loa_customer_name=loa_customer_name,
                            )

                            if not name_comparison["match"]:
                                self.logger.warning("  Account name mismatch detected!")
                                account_name_rejections.append(
                                    name_comparison.get("reason")
                                )
                            else:
                                self.logger.info(
                                    f"  ✓ Account names match ({name_comparison.get('match_type', 'exact')})"
                                )
                        else:
                            self.logger.warning(
                                "  Failed to extract customer name from LOA"
                            )
                            account_name_rejections.append(
                                "Failed to extract customer name from LOA for comparison"
                            )
                    except Exception as e:
                        self.logger.error(f"Account name comparison error: {str(e)}")
                        account_name_rejections.append(
                            f"Account name comparison failed: {str(e)}"
                        )

            # Add account name comparison rejections to the validation result
            if account_name_rejections:
                self.logger.warning(
                    f"Account name comparison found {len(account_name_rejections)} issue(s) - adding to rejection reasons"
                )

                # Combine with existing rejection reasons
                existing_rejections = gpt_validation_result.get("rejectionReasons", [])
                combined_rejections = existing_rejections + account_name_rejections
                gpt_validation_result["rejectionReasons"] = combined_rejections

                # Force status to REJECT
                gpt_validation_result["status"] = "REJECT"
                self.logger.info(
                    f"Status changed to REJECT due to account name mismatch. Total rejection reasons: {len(combined_rejections)}"
                )

            # =========================================================================
            # ACCOUNT NUMBER COMPARISON (Great Lakes Region Only)
            # =========================================================================
            # Perform account number comparison if parameter provided
            account_number_rejections = []

            if self.region == "Great Lakes" and self.service_location_ldc:
                self.logger.info(
                    "Performing account number comparison for Great Lakes Region..."
                )

                # Extract account numbers from LOA if service_location_ldc parameter provided
                if pdf_path:
                    try:
                        self.logger.info(
                            f"Comparing account numbers: Salesforce='{self.service_location_ldc}'"
                        )
                        account_numbers_result = self.gpt4o_verification_integration.extract_account_numbers_from_great_lakes_loa(
                            pdf_path=pdf_path,
                            udc=self.provided_udc,
                            extraction_log=extraction_log,
                        )

                        if account_numbers_result.get(
                            "success"
                        ) and account_numbers_result.get("account_numbers"):
                            loa_account_numbers = account_numbers_result[
                                "account_numbers"
                            ]
                            extraction_method = account_numbers_result.get(
                                "method", "unknown"
                            )
                            self.logger.info(
                                f"  LOA Account Numbers ({len(loa_account_numbers)}): {loa_account_numbers}"
                            )
                            self.logger.info(
                                f"  Extraction Method: {extraction_method}"
                            )

                            # Compare account numbers with EXACT matching (no tolerance)
                            account_comparison = self.gpt4o_verification_integration.compare_account_numbers_exact(
                                salesforce_accounts=self.service_location_ldc,
                                loa_accounts=loa_account_numbers,
                            )

                            if not account_comparison["match"]:
                                self.logger.warning(
                                    "  Account number mismatch detected!"
                                )
                                account_number_rejections.append(
                                    account_comparison.get("reason")
                                )
                            else:
                                self.logger.info(
                                    "  ✓ All account numbers matched exactly!"
                                )
                                # Log matching details
                                for match in account_comparison.get(
                                    "matched_accounts", []
                                ):
                                    self.logger.info(
                                        f"    - SF: {match['salesforce']} = LOA: {match['loa']}"
                                    )
                        else:
                            self.logger.warning(
                                "  Failed to extract account numbers from LOA"
                            )
                            account_number_rejections.append(
                                "Failed to extract account numbers from LOA for comparison"
                            )
                    except Exception as e:
                        self.logger.error(f"Account number comparison error: {str(e)}")
                        account_number_rejections.append(
                            f"Account number comparison failed: {str(e)}"
                        )

            # Add account number comparison rejections to the validation result
            if account_number_rejections:
                self.logger.warning(
                    f"Account number comparison found {len(account_number_rejections)} issue(s) - adding to rejection reasons"
                )

                # Combine with existing rejection reasons
                existing_rejections = gpt_validation_result.get("rejectionReasons", [])
                combined_rejections = existing_rejections + account_number_rejections
                gpt_validation_result["rejectionReasons"] = combined_rejections

                # Force status to REJECT
                gpt_validation_result["status"] = "REJECT"
                self.logger.info(
                    f"Status changed to REJECT due to account number mismatch. Total rejection reasons: {len(combined_rejections)}"
                )

            # Build comprehensive validation result with simplified production format + internal testing details
            validation_result = {
                # Production model output format (simplified)
                "fileName": gpt_validation_result.get("fileName", document_id),
                "status": gpt_validation_result.get("status", "REJECT"),
                "rejectionReasons": gpt_validation_result.get("rejectionReasons", []),
                "expiration_date": gpt_validation_result.get(
                    "expiration_date", "Not calculated"
                ),
                # Legacy/internal testing fields (for backward compatibility and detailed analysis)
                "document_id": gpt_validation_result.get("fileName", document_id),
                "validation_status": gpt_validation_result.get("status", "REJECT"),
                "ocr_success": extraction_log["extraction_success"],
                "extracted_text_length": len(extracted_text),
                "all_rejection_reasons": gpt_validation_result.get(
                    "rejectionReasons", []
                ),
                # Layout analysis metadata for internal testing
                "layout_processing_time": extraction_log["processing_time_seconds"],
                "layout_page_count": extraction_log["page_count"],
                "processing_timestamp": datetime.now().isoformat(),
                "extracted_text": extracted_text,
                "gpt_response_raw": gpt_response,
                "gpt_parsing_error": None,
                "selection_marks_detected": len(selection_marks),
                "key_value_pairs_detected": len(key_value_pairs),
                "potential_initials_detected": len(potential_initials),
                "layout_analysis_results": {
                    "selected_marks": [
                        mark
                        for mark in selection_marks
                        if mark.get("state") == "selected"
                    ],
                    "unselected_marks": [
                        mark
                        for mark in selection_marks
                        if mark.get("state") == "unselected"
                    ],
                    "form_fields": key_value_pairs,
                    "potential_initials": potential_initials,
                },
                "file_source": "LOA document",
                # State/utility detection for internal analysis
                "utility_identified": (
                    self.provided_udc if self.provided_udc else "Unknown"
                ),
                "state_identified": detected_state,
                "cres_provider_identified": "Constellation",
                "is_constellation_loa": True,
                "recommendation": gpt_validation_result.get("status", "REJECT"),
                # Enhanced analysis fields for internal debugging
                "initial_recognition_analysis": f"Found {len(potential_initials)} potential initials: {[p.get('text', '') for p in potential_initials[:5]]}",
            }

            # Add LOA expiration date information (enhanced with calculation details)
            if loa_expiration_result:
                validation_result["expiration_details"] = {
                    "expiration_date": loa_expiration_result.get(
                        "expiration_date_formatted", "Not calculated"
                    ),
                    "months_until_expiration": loa_expiration_result.get(
                        "months_until_expiration"
                    ),
                    "days_until_expiration": loa_expiration_result.get(
                        "days_until_expiration"
                    ),
                    "expiration_months_used": loa_expiration_result.get(
                        "expiration_months_used"
                    ),
                    "expiration_rule_used": loa_expiration_result.get(
                        "expiration_rule_used"
                    ),
                    "explicit_expiration_found": loa_expiration_result.get(
                        "explicit_expiration_found", False
                    ),
                    "explicit_expiration_months": loa_expiration_result.get(
                        "explicit_expiration_months"
                    ),
                    "is_expired": loa_expiration_result.get("is_expired", False),
                    "signature_date": loa_expiration_result.get(
                        "signature_date", "Not found"
                    ),
                    "calculation_details": loa_expiration_result.get(
                        "calculation_details", "No calculation performed"
                    ),
                }
            else:
                # Use the expiration_date from GPT response if available, otherwise default
                gpt_expiration_date = gpt_validation_result.get(
                    "expiration_date", "Not calculated"
                )

                validation_result["expiration_details"] = {
                    "expiration_date": gpt_expiration_date,
                    "months_until_expiration": None,
                    "days_until_expiration": None,
                    "expiration_months_used": None,
                    "expiration_rule_used": (
                        f"GPT analysis calculation: {gpt_expiration_date}"
                        if gpt_expiration_date != "Not calculated"
                        else "No calculation available"
                    ),
                    "explicit_expiration_found": False,
                    "explicit_expiration_months": None,
                    "is_expired": False,
                    "signature_date": "GPT analysis determined signature date",
                    "calculation_details": f"GPT-4o calculated expiration date as {gpt_expiration_date}. Python regex did not detect signature date patterns.",
                }

            return validation_result

        except json.JSONDecodeError as e:
            # Enhanced fallback with detailed error info
            return {
                "document_id": document_id,
                "validation_status": "ERROR",
                "ocr_success": extraction_log["extraction_success"],
                "extracted_text": extracted_text,
                "extracted_text_length": len(extracted_text),
                "error": f"GPT-4o JSON parsing failed: {str(e)}",
                "gpt_response_raw": gpt_response,
                "gpt_parsing_error": str(e),
                "processing_timestamp": datetime.now().isoformat(),
                "validation_results": [
                    {
                        "category": "GPT_PARSING_ERROR",
                        "status": "FAIL",
                        "details": f"Failed to parse GPT-4o response as JSON: {str(e)}",
                        "rejection_reason": "System error - could not parse AI response",
                        "relevant_text": "See gpt_response_raw field for full response",
                        "text_evidence": "GPT-4o returned invalid JSON format",
                    }
                ],
                "all_rejection_reasons": [
                    f"System error: GPT-4o JSON parsing failed - {str(e)}"
                ],
                "file_source": "LOA document",
            }
