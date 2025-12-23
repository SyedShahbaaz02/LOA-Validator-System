"""
Enhanced GPT-4o Verification Integration Module for Production LOA Validation
This module extends the GPT-4o OCR capabilities with specific verification for critical checkboxes
and New England service options.
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional

import fitz  # PyMuPDF

from intelligentflow.business_logic.openai_4o_service import Openai4oService

from .gpt4o_ocr_integration import GPT4oOCRIntegration
from ..utils.retry import aggressive_retry


class GPT4oVerificationIntegration:
    """Production GPT-4o verification integration using dependency injection.

    Args:
        openai_4o_service: The OpenAI service for GPT-4o processing
    """

    # UDC-specific regex patterns for account number extraction (flexible - captures range of lengths)
    # NOTE: These patterns are guidelines, not strict requirements
    # Account numbers that don't match the expected format will still be extracted
    ACCOUNT_NUMBER_PATTERNS = {
        "CEI": r"\b\d{11,25}\b",  # FirstEnergy CEI: typically 20 digits, but allow 11-25
        "OE": r"\b\d{11,25}\b",  # FirstEnergy OE: typically 20 digits, but allow 11-25
        "TE": r"\b\d{11,25}\b",  # FirstEnergy TE: typically 20 digits, but allow 11-25
        "AEP": r"\b\d{11,25}\b",  # AEP: typically 17 digits, but allow 11-25
        "CSPC": r"\b\d{11,25}\b",  # AEP CSPC: typically 17 digits, but allow 11-25
        "OPC": r"\b\d{11,25}\b",  # AEP OPC: typically 17 digits, but allow 11-25
        "ComEd": r"\b\d{8,25}\b",  # ComEd: 8-25 digits (flexible)
        "Ameren": r"\b\d{8,25}\b",  # Ameren: 8-25 digits (flexible)
        "Dayton": r"\b(?:\d{11,30}|\d{8,15}[Zz\W]?\d{8,12})\b",  # Dayton: 11-30 consecutive digits OR 8-15 digits + optional Z/separator + 8-12 digits
        "DPL": r"\b(?:\d{11,30}|\d{8,15}[Zz\W]?\d{8,12})\b",  # Dayton Power & Light: same as Dayton
        "Duke": r"\b(?:\d{11,30}|\d{8,15}[Zz\W]?\d{8,12})\b",  # Duke: same flexible pattern as Dayton for better Z handling
        "Cinergy": r"\b(?:\d{11,30}|\d{8,15}[Zz\W]?\d{8,12})\b",  # Cinergy: same flexible pattern as Dayton for better Z handling
    }

    # Ohio UDCs that require specific field name handling
    OHIO_UDCS = [
        "CEI",
        "OE",
        "TE",  # FirstEnergy utilities
        "AEP",
        "CSPC",
        "OPC",  # AEP utilities
        "Dayton",
        "DPL",  # Dayton utilities
        "Duke",
        "Cinergy",  # Duke/Cinergy utilities
    ]

    def __init__(self, openai_4o_service: Openai4oService):
        if openai_4o_service is None:
            raise ValueError("openai_4o_service is required and cannot be None")

        self.openai_4o_service = openai_4o_service
        self.logger = logging.getLogger(__name__)

        # Use the OCR integration for image extraction
        self.ocr_integration = GPT4oOCRIntegration(openai_4o_service)

    @staticmethod
    def normalize_account_flexible(acc: str) -> str:
        """
        Create a flexible normalized version for Cinergy/Duke Energy accounts.
        Converts both 'Z' and '2' at position 13 to a common character for comparison.

        Args:
            acc: Account number string to normalize

        Returns:
            Normalized account number with only digits, Z/2 treated as equivalent
        """
        # Check if this looks like a Cinergy/Duke Energy account
        if "Z" in acc.upper() or (len(acc) == 23 and acc[12] in ["Z", "z", "2"]):
            # Replace both Z and 2 with a common placeholder for comparison
            normalized = acc.upper().replace("Z", "2")
            return re.sub(r"[^0-9]", "", normalized)
        else:
            return re.sub(r"[^0-9]", "", acc)

    def verify_critical_checkboxes(
        self,
        pdf_path: str,
        extraction_log: Optional[Dict] = None,
        critical_keywords: Optional[List[str]] = None,
    ) -> Dict:
        """
        Second pass verification for critical checkboxes using GPT-4o

        Args:
            pdf_path: Path to the PDF file
            extraction_log: The current extraction log from Azure Layout
            critical_keywords: List of keywords to look for in checkbox text

        Returns:
            Updated extraction log with verified checkbox states
        """
        # Default critical keywords to check for if not provided
        if critical_keywords is None:
            critical_keywords = [
                "Interval Historical Energy Usage Data Release",
                "Account/SDI Number Release",
                "Historical Usage Data Release",
            ]

        if extraction_log is None:
            extraction_log = {
                "pdf_path": pdf_path,
                "extraction_timestamp": datetime.now().isoformat(),
                "selection_marks": [],
            }

        self.logger.info(
            "Performing second pass verification with GPT-4o for critical checkboxes..."
        )

        # Extract the first page as an image
        image_data = self.ocr_integration.extract_pdf_image(pdf_path, page_num=0)

        if not image_data:
            self.logger.error("Failed to extract image from PDF")
        return extraction_log

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def verify_cmp_billing_options_with_gpt4o(
        self, pdf_path: str, extraction_log: Dict
    ) -> Dict:
        """
        GPT-4o verification for CMP billing options checkboxes.
        CMP has 2 billing options and exactly ONE must be selected.

        Args:
            pdf_path (str): Path to the PDF file.
            extraction_log (Dict): The current extraction log from Azure Layout.

        Returns:
            Dict: Updated extraction log with verified CMP billing option states.
        """
        self.logger.info("Performing GPT-4o verification for CMP billing options...")

        image_data = self.ocr_integration.extract_pdf_image(pdf_path, page_num=0)
        if not image_data:
            self.logger.error(
                "Failed to extract image from PDF for CMP billing options verification."
            )
            return extraction_log

        base64_image = self.ocr_integration.encode_image_to_base64(image_data)

        cmp_options_prompt = """CRITICAL: Carefully examine this document image for CMP (Central Maine Power) billing options.

        STEP 1: DETECT IF BILLING OPTIONS SECTION EXISTS
        First, look for a section with text "***CHECK ONE" or "CHECK ONE" followed by billing options.
        IMPORTANT: Some CMP LOAs do NOT have this billing section at all - this is VALID and should not cause rejection.

        If the billing section EXISTS, it should have TWO options:
        1. "Invoice the customer" (Not applicable to NSTAR)
        2. "Invoice the supplier/broker as follows:" (with contact information fields)

        STEP 2: IF SECTION EXISTS, CHECK WHICH OPTION IS SELECTED
        - Look for checkboxes, X marks, checkmarks, or filled boxes near each option
        - The checkboxes are typically small square boxes to the left of the text
        - Look for text patterns indicating selection
        - Pay attention to visual differences between the two options
        - Common patterns: ☐ (empty), ☑ (checked), ☒ (X marked)

        VALIDATION REQUIREMENTS:
        - If billing section does NOT exist: Mark billing_section_exists=false - this is VALID, no rejection
        - If billing section EXISTS but no option selected: selection_count=0 - INVALID
        - If billing section EXISTS and exactly 1 option selected: selection_count=1 - VALID
        - If billing section EXISTS and 2 options selected: selection_count=2 - INVALID

        Return your findings in JSON format:
        {
        "cmp_billing_verification": {
            "billing_section_exists": true/false,
            "invoice_customer_selected": true/false,
            "invoice_supplier_selected": true/false,
            "selection_count": 0, 1, or 2,
            "confidence": 0-100,
            "reasoning": "Detailed explanation: Does billing section exist? If yes, which option(s) appear selected and why?"
        }
        }"""

        # Create the user prompt for vision analysis
        user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
        {cmp_options_prompt}

        [ATTACHED_IMAGE]
        data:image/png;base64,{base64_image}"""

        # System prompt for GPT-4o vision
        system_prompt = "You are an expert document analyzer specializing in CMP (Central Maine Power) LOA forms. Your task is to accurately determine which of the 2 billing options are selected in the '***CHECK ONE' section."

        try:
            start_time = datetime.now()

            # Use the production OpenAI service
            gpt_response_data = self.openai_4o_service.process_with_prompts(
                system_prompt, user_prompt, max_token=1000, raw_response=True
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Extract the actual response content from the service response
            if gpt_response_data and gpt_response_data[0]["ai_result"]:
                analysis_text = gpt_response_data[0]["ai_result"][0]["result"]
            else:
                self.logger.error(
                    "No response from OpenAI service for CMP billing options verification"
                )
                if "cmp_billing_options" not in extraction_log:
                    extraction_log["cmp_billing_options"] = {}
                extraction_log["cmp_billing_options"][
                    "gpt4o_verification_error"
                ] = "No response from OpenAI service"
                return extraction_log

            json_match = re.search(r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(1))
            else:
                analysis_json = json.loads(analysis_text)

            self.logger.info(
                f"GPT-4o CMP billing options verification completed in {processing_time:.2f} seconds"
            )

            verification_data = analysis_json.get("cmp_billing_verification")
            if verification_data:
                billing_section_exists = verification_data.get(
                    "billing_section_exists", True
                )
                invoice_customer_selected = verification_data.get(
                    "invoice_customer_selected", False
                )
                invoice_supplier_selected = verification_data.get(
                    "invoice_supplier_selected", False
                )
                selection_count = int(invoice_customer_selected) + int(
                    invoice_supplier_selected
                )

                self.logger.info("GPT-4o CMP Findings:")
                self.logger.info(
                    f"      - Billing Section Exists: {'Yes' if billing_section_exists else 'No'}"
                )
                self.logger.info(
                    f"      - Invoice Customer: {'Selected' if invoice_customer_selected else 'Not Selected'}"
                )
                self.logger.info(
                    f"      - Invoice Supplier/Broker: {'Selected' if invoice_supplier_selected else 'Not Selected'}"
                )
                self.logger.info(f"      - Final Selection Count: {selection_count}")

                # Initialize cmp_billing_options if not present
                if "cmp_billing_options" not in extraction_log:
                    extraction_log["cmp_billing_options"] = {}

                # Override the original cmp_billing_options in the extraction log
                extraction_log["cmp_billing_options"][
                    "billing_section_exists"
                ] = billing_section_exists
                extraction_log["cmp_billing_options"][
                    "invoice_customer_selected"
                ] = invoice_customer_selected
                extraction_log["cmp_billing_options"][
                    "invoice_supplier_selected"
                ] = invoice_supplier_selected
                extraction_log["cmp_billing_options"][
                    "selection_count"
                ] = selection_count
                extraction_log["cmp_billing_options"]["gpt4o_verified"] = True
                extraction_log["cmp_billing_options"][
                    "gpt4o_verification_details"
                ] = verification_data
                extraction_log["cmp_billing_options"][
                    "detected"
                ] = billing_section_exists

                self.logger.info(
                    "Overwrote initial CMP billing detection with GPT-4o results."
                )
            else:
                self.logger.warning(
                    "GPT-4o did not return the expected 'cmp_billing_verification' key."
                )
                if "cmp_billing_options" not in extraction_log:
                    extraction_log["cmp_billing_options"] = {}
                extraction_log["cmp_billing_options"][
                    "gpt4o_verification_error"
                ] = "Invalid JSON structure from GPT-4o"

        except Exception as e:
            error_msg = f"GPT-4o CMP billing options verification failed: {str(e)}"
            self.logger.error(error_msg)
            if "cmp_billing_options" not in extraction_log:
                extraction_log["cmp_billing_options"] = {}
            extraction_log["cmp_billing_options"][
                "gpt4o_verification_error"
            ] = f"Exception: {str(e)}"

        return extraction_log

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def verify_document_integrity_with_gpt4o(self, pdf_path: str) -> Dict:
        """
        Use GPT-4o Vision to verify document integrity - detect interleaved pages,
        misaligned content, and document corruption.

        This is the Layer 2 verification that catches subtle issues text heuristics miss.

        CRITICAL: This function uses aggressive retry logic to guarantee success.
        It will retry up to 50 times with exponential backoff.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dict with:
                - success: bool
                - is_valid: bool (True if document is intact)
                - confidence: float (0.0-1.0)
                - issues: List[Dict] (detected issues)
                - reasoning: str (GPT-4o's analysis)
        """
        try:
            self.logger.info("Performing GPT-4o document integrity verification...")

            # Use existing OCR integration to extract image
            image_data = self.ocr_integration.extract_pdf_image(pdf_path, page_num=0)
            if not image_data:
                self.logger.error(
                    "Failed to extract image from PDF for integrity check."
                )
                return {
                    "success": False,
                    "error": "Failed to extract image from PDF",
                    "is_valid": True,  # Assume valid if we can't check
                    "confidence": 0.5,
                }

            # Encode image to base64
            base64_image = self.ocr_integration.encode_image_to_base64(image_data)

            # Build GPT-4o Vision prompt for integrity checking
            system_prompt = """You are a document integrity verification specialist for LOA (Letter of Authorization) documents.

**YOUR PRIMARY TASK: Detect SEVERE text interleaving and document corruption that makes the document UNREADABLE**

Your task is to detect SEVERE document integrity problems that make documents unreadable or unusable.

WHAT TO IGNORE (NORMAL - DO NOT FLAG):
**These are NORMAL and should NOT be flagged as issues:**

1. **NORMAL UTILITY ABBREVIATIONS**:
   - "TE", "OE", "CEI" (FirstEnergy utilities - Toledo Edison, Ohio Edison, Cleveland Electric Illuminating)
   - "AEP", "OVEC", "PSO" (AEP utilities)
   - "CMP", "BECO", "NECO", "MECO", "WMECO" (New England utilities)
   - These abbreviations appearing anywhere in the document are EXPECTED and NORMAL

2. **NORMAL OCR/PDF ARTIFACTS**:
   - Slight text misalignment due to scanning/OCR
   - Minor spacing irregularities
   - Faint or light text
   - Standard form field formatting (boxes, underlines, lines)
   - Text at different sizes (headers, body text, fine print)
   - Overlapping form fields (checkboxes near text is normal)

3. **NORMAL FORM CHARACTERISTICS**:
   - Multiple sections with different layouts (customer section, billing section, authorization section)
   - Tables, checkboxes, and form fields
   - Headers and footers
   - Legal text in small font
   - Signature fields with printed names nearby
   - Contact information in different locations

4. **NORMAL DOCUMENT VARIATIONS**:
   - Different font sizes for emphasis
   - Bold/italic text for section headers
   - Watermarks or form numbers
   - Date stamps or version numbers
   - Pre-filled vs handwritten/typed fields

WHAT TO FLAG (SEVERE ISSUES ONLY):
**ONLY flag if you see these SEVERE issues (be 100% certain):**

1. **CLEARLY INTERLEAVED PAGES**:
   - Customer name changes COMPLETELY between pages (e.g., "ABC Company" on page 1, "XYZ Corporation" on page 2)
   - Two completely different document types mixed together (e.g., LOA mixed with invoice)
   - Headers/footers that reference DIFFERENT utilities on SAME document (must be different utilities, not just different abbreviations of same utility)

2. **SEVERE CORRUPTION**:
   - Large sections of completely unreadable/garbled text (not just minor OCR errors)
   - Entire paragraphs missing or replaced with gibberish
   - Critical sections completely obscured or illegible

3. **OBVIOUS PAGE ORDER PROBLEMS**:
   - Page numbers clearly out of sequence (page 5, then page 2, then page 7)
   - Content that makes no logical sense in sequence (conclusion before introduction)

VALIDATION RULES:
**CRITICAL DECISION CRITERIA:**

- **IF IN DOUBT, MARK AS VALID** - Favor accepting documents unless you're CERTAIN of severe issues
- **Minor formatting issues = VALID** - Slight misalignment, OCR artifacts, formatting variations are NORMAL
- **Utility abbreviations = VALID** - "TE", "OE", "CEI", etc. are expected utility names
- **Mixed sections = VALID** - LOAs naturally have different sections with different layouts
- **Only flag SEVERE, OBVIOUS issues** - If you need to look closely to find an issue, it's probably normal

**EXAMPLES OF WHAT NOT TO FLAG (READ THESE CAREFULLY):**
- ❌ "The text 'TE' appears misaligned" → NORMAL utility abbreviation, DO NOT FLAG
- ❌ "The text 'TE' is inserted in an unnatural position" → NORMAL utility name, DO NOT FLAG
- ❌ "The text 'TE' appears to be added/overlaid" → NORMAL utility abbreviation, DO NOT FLAG
- ❌ "Signature field has overlapping text" → NORMAL form layout, DO NOT FLAG
- ❌ "Signature field shows overlapping text and garbled formatting" → NORMAL OCR artifact, DO NOT FLAG
- ❌ "Text appears misaligned or poorly formatted" → NORMAL scanning/OCR issue, DO NOT FLAG
- ❌ "Some text appears faint or light" → NORMAL scanning artifact, DO NOT FLAG
- ❌ "Different sections have different formatting" → NORMAL LOA structure, DO NOT FLAG
- ❌ "Form has checkboxes near text" → NORMAL form design, DO NOT FLAG
- ❌ ANY comment about "TE", "OE", "CEI" text positioning → These are utility names, ALWAYS NORMAL
- ❌ ANY comment about signature field formatting → This is ALWAYS normal form design

**EXAMPLES OF WHAT TO FLAG:**
- ✓ "Customer name changes from 'Acme Corp' on page 1 to 'Beta Inc' on page 2 with no explanation" → FLAG
- ✓ "Document contains pages from both an LOA and a utility bill mixed together" → FLAG
- ✓ "Entire middle section is completely unreadable gibberish with no structure" → FLAG

Return your analysis in JSON format:
{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "issues": [
        {
            "type": "INTERLEAVED_PAGES" | "SEVERE_CORRUPTION",
            "description": "Clear description of the SEVERE issue",
            "evidence": "Specific visual evidence proving this is a severe integrity problem",
            "page_numbers": [1, 2, ...]
        }
    ],
    "reasoning": "Your detailed analysis - if marking as invalid, explain why you're 100% certain this is a SEVERE integrity issue and not a normal formatting variation"
}

**FINAL REMINDERS BEFORE YOU RESPOND:**
1. Utility abbreviations ("TE", "OE", "CEI", etc.) are ALWAYS NORMAL - never flag these
2. Signature field formatting issues are ALWAYS NORMAL - never flag these
3. Slight text misalignment is ALWAYS NORMAL OCR behavior - never flag this
4. If you're not 100% certain it's CATASTROPHIC corruption, mark as VALID
5. When in doubt: MARK AS VALID

**DEFAULT ASSUMPTION: Documents are VALID unless you have CLEAR, UNDENIABLE evidence of SEVERE integrity issues.**

**REPEAT: IF YOU SEE "TE" TEXT OR SIGNATURE FORMATTING ISSUES, THESE ARE NORMAL. DO NOT FLAG THEM.**"""

            user_prompt = f"""Analyze this LOA document for CATASTROPHIC integrity issues ONLY.

**ABSOLUTE REQUIREMENTS:**
- START BY ASSUMING THE DOCUMENT IS VALID
- ONLY mark as invalid if you find UNDENIABLE evidence of CATASTROPHIC corruption
- DO NOT flag "TE" text positioning - it's a utility name
- DO NOT flag signature field formatting - it's normal form design
- DO NOT flag text misalignment - it's normal OCR behavior
- DO NOT flag overlapping text in forms - it's normal layout
- IF YOU HAVE ANY DOUBT, MARK AS VALID

**REMEMBER:** Unless this document is completely destroyed or has pages from different documents mixed together, it's VALID.

**YOUR DEFAULT ANSWER SHOULD BE: is_valid: true**

[ATTACHED_IMAGE]
data:image/png;base64,{base64_image}"""

            # Call GPT-4o service (same pattern as other methods in this file)
            start_time = datetime.now()
            gpt_response_data = self.openai_4o_service.process_with_prompts(
                system_prompt, user_prompt, max_token=1500, raw_response=True
            )
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            if not gpt_response_data or not gpt_response_data[0]["ai_result"]:
                self.logger.error(
                    "No response from OpenAI service for document integrity check"
                )
                return {
                    "success": False,
                    "error": "No response from OpenAI service",
                    "is_valid": True,  # Assume valid if we can't check
                    "confidence": 0.5,
                }

            # Extract response
            gpt_response = gpt_response_data[0]["ai_result"][0]["result"]
            self.logger.info(
                f"GPT-4o document integrity check completed in {processing_time:.2f} seconds"
            )

            # Extract JSON from response
            if "```json" in gpt_response:
                json_start = gpt_response.find("```json") + 7
                json_end = gpt_response.find("```", json_start)
                json_text = gpt_response[json_start:json_end].strip()
            else:
                json_text = gpt_response.strip()

            try:
                integrity_data = json.loads(json_text)

                return {
                    "success": True,
                    "is_valid": integrity_data.get("is_valid", True),
                    "confidence": integrity_data.get("confidence", 1.0),
                    "issues": integrity_data.get("issues", []),
                    "reasoning": integrity_data.get("reasoning", ""),
                    "raw_response": gpt_response,
                }

            except json.JSONDecodeError as e:
                self.logger.error(
                    f"Failed to parse GPT-4o integrity response: {str(e)}"
                )
                return {
                    "success": False,
                    "error": f"JSON parsing failed: {str(e)}",
                    "is_valid": True,  # Assume valid on parse error
                    "confidence": 0.5,
                    "raw_response": gpt_response,
                }

        except Exception as e:
            self.logger.error(f"Document integrity verification error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "is_valid": True,  # Assume valid on error
                "confidence": 0.5,
            }

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def extract_customer_signature_date_with_gpt4o(self, pdf_path: str) -> Dict:
        """
        Extract customer signature date using GPT-4o vision.
        CRITICAL: Must extract CUSTOMER signature date, not supplier/broker date.

        Uses aggressive retry logic (50 attempts) to ensure success.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            Dict: Contains customer_signature_date and verification details.
        """
        self.logger.info("Performing GPT-4o customer signature date extraction...")

        image_data = self.ocr_integration.extract_pdf_image(pdf_path, page_num=0)
        if not image_data:
            self.logger.error(
                "Failed to extract image from PDF for signature date extraction."
            )
            return {
                "success": False,
                "customer_signature_date": None,
                "error": "Failed to extract image from PDF",
            }

        base64_image = self.ocr_integration.encode_image_to_base64(image_data)

        signature_date_prompt = """CRITICAL: Extract the CUSTOMER signature date from this document.

        IMPORTANT DISTINCTIONS:
        1. Documents often have TWO signature sections:
           - Customer Information section (at BOTTOM) - THIS IS WHAT WE NEED
           - Supplier/Broker section (often at top or middle) - IGNORE THIS

        2. Look for sections labeled:
           - "Customer Information"
           - "To be completed by the Customer"
           - "Customer Signature"
           - "Authorization" (customer section)

        3. DO NOT use dates from:
           - "Supplier/Broker" sections
           - "Third Party Representative" sections
           - "To be completed by Supplier" sections

        4. The customer signature date is typically:
           - Near customer name, address, and phone fields
           - In the section where customer signs to authorize
           - At the BOTTOM of the form (not top or middle)

        5. Date formats to look for:
           - MM/DD/YYYY (e.g., 01/15/2024)
           - MM-DD-YYYY (e.g., 01-15-2024)
           - M/D/YYYY (e.g., 1/5/2024)

        HANDWRITTEN DATE CAUTION:
        - Pay SPECIAL ATTENTION to handwritten years (e.g., 2023 vs 2025)
        - Handwritten digits like '3' and '5' can be easily confused
        - Handwritten '0' and '6' can also be confused
        - If the year appears handwritten and you're uncertain, indicate your confidence level
        - Consider the context: a 2023 date on a form from 2025 might be a misread

        VALIDATION:
        - Only return a date if you're confident it's the CUSTOMER's signature date
        - If you see multiple dates, choose the one in the customer section
        - If uncertain, look for "Date:" field near "Customer Signature"

        Return your findings in JSON format:
        {
        "customer_signature_extraction": {
            "customer_signature_date": "MM/DD/YYYY or null",
            "date_found": true/false,
            "confidence": 0-100,
            "location_description": "Description of where the date was found (e.g., 'Customer Information section at bottom')",
            "reasoning": "Explanation of how you determined this is the customer date and not supplier date",
            "year_confidence": 0-100,
            "year_warning": "Warning message if year is uncertain, or null",
            "possible_alternate_year": "YYYY or null (if you think the year might be different)"
        }
        }"""

        # Create the user prompt for vision analysis
        user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
        {signature_date_prompt}

        [ATTACHED_IMAGE]
        data:image/png;base64,{base64_image}"""

        # System prompt for GPT-4o vision
        system_prompt = "You are an expert document analyzer specializing in extracting customer signature dates from LOA forms. Your task is to accurately identify the CUSTOMER signature date, not supplier or broker dates."

        try:
            start_time = datetime.now()

            # Use the production OpenAI service
            gpt_response_data = self.openai_4o_service.process_with_prompts(
                system_prompt, user_prompt, max_token=1000, raw_response=True
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Extract the actual response content from the service response
            if gpt_response_data and gpt_response_data[0]["ai_result"]:
                analysis_text = gpt_response_data[0]["ai_result"][0]["result"]
            else:
                self.logger.error(
                    "No response from OpenAI service for customer signature date extraction"
                )
                return {
                    "success": False,
                    "customer_signature_date": None,
                    "error": "No response from OpenAI service",
                }

            json_match = re.search(r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(1))
            else:
                analysis_json = json.loads(analysis_text)

            self.logger.info(
                f"GPT-4o customer signature date extraction completed in {processing_time:.2f} seconds"
            )

            verification_data = analysis_json.get("customer_signature_extraction")
            if verification_data:
                customer_date = verification_data.get("customer_signature_date")
                date_found = verification_data.get("date_found", False)
                confidence = verification_data.get("confidence", 0)
                location = verification_data.get("location_description", "Unknown")
                reasoning = verification_data.get("reasoning", "No reasoning provided")

                # Get year confidence and alternate year for auto-correction
                year_confidence = verification_data.get("year_confidence", 100)
                year_warning = verification_data.get("year_warning")
                possible_alternate_year = verification_data.get(
                    "possible_alternate_year"
                )

                # Auto-correction: If year confidence is low and alternate year is suggested, use it
                if customer_date and year_confidence < 95 and possible_alternate_year:
                    try:
                        original_date = datetime.strptime(customer_date, "%m/%d/%Y")
                        corrected_date = original_date.replace(
                            year=int(possible_alternate_year)
                        )
                        original_year = original_date.year
                        customer_date = corrected_date.strftime("%m/%d/%Y")
                        self.logger.warning(
                            f"Auto-corrected year from {original_year} to {possible_alternate_year} due to low confidence ({year_confidence}%)"
                        )
                        self.logger.warning(f"Year warning: {year_warning}")
                    except Exception as e:
                        self.logger.error(f"Failed to auto-correct year: {str(e)}")

                # Build log message parts

                log_parts = [
                    "GPT-4o Customer Signature Date Findings:",
                    f"      - Date Found: {date_found}",
                    f"      - Customer Signature Date: {customer_date}",
                    f"      - Confidence: {confidence}%",
                    f"      - Year Confidence: {year_confidence}%",
                ]

                # Add warning if present

                if year_warning:

                    log_parts.append(f"      - Year Warning: {year_warning}")

                # Add remaining fields

                log_parts.extend(
                    [f"      - Location: {location}", f"      - Reasoning: {reasoning}"]
                )

                # Log as single statement

                self.logger.info("\n".join(log_parts))

                return {
                    "success": True,
                    "customer_signature_date": customer_date,
                    "date_found": date_found,
                    "confidence": confidence,
                    "location_description": location,
                    "reasoning": reasoning,
                    "gpt4o_verified": True,
                    "verification_details": verification_data,
                }
            else:
                self.logger.warning(
                    "GPT-4o did not return the expected 'customer_signature_extraction' key."
                )
                return {
                    "success": False,
                    "customer_signature_date": None,
                    "error": "Invalid JSON structure from GPT-4o",
                }

        except Exception as e:
            error_msg = f"GPT-4o customer signature date extraction failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "customer_signature_date": None,
                "error": f"Exception: {str(e)}",
            }

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def extract_requestor_signature_date_with_gpt4o(self, pdf_path: str) -> Dict:
        """
        Extract requestor/billing signature date using GPT-4o vision.
        CRITICAL: Must extract REQUESTOR/BILLING signature date, not customer date.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            Dict: Contains requestor_signature_date and verification details.
        """
        self.logger.info("Performing GPT-4o requestor signature date extraction...")

        image_data = self.ocr_integration.extract_pdf_image(pdf_path, page_num=0)
        if not image_data:
            self.logger.error(
                "Failed to extract image from PDF for requestor signature date extraction."
            )
            return {
                "success": False,
                "requestor_signature_date": None,
                "error": "Failed to extract image from PDF",
            }

        base64_image = self.ocr_integration.encode_image_to_base64(image_data)

        requestor_date_prompt = """CRITICAL: Extract the REQUESTOR/BILLING signature date from this document.

        IMPORTANT DISTINCTIONS:
        1. Documents often have TWO signature sections:
           - Customer Information section (often at top or middle) - IGNORE THIS
           - Requestor/Billing section (typically at BOTTOM) - THIS IS WHAT WE NEED

        2. Look for sections labeled:
           - "Requestor & Billing Information"
           - "Requestor/Billing Signature"
           - "To be completed by Supplier/Broker"
           - "Dated Signed by Requestor/Billing Co."

        3. DO NOT use dates from:
           - "Customer Information" sections
           - "Customer Signature" sections
           - "Authorization" (customer section)

        4. The requestor signature date is typically:
           - Near requestor/billing company name and contact fields
           - In the section where supplier/broker signs
           - At the BOTTOM of the form (not top or middle)

        5. Date formats to look for:
           - MM/DD/YYYY (e.g., 01/15/2024)
           - MM-DD-YYYY (e.g., 01-15-2024)
           - M/D/YYYY (e.g., 1/5/2024)

        VALIDATION:
        - Only return a date if you're confident it's the REQUESTOR/BILLING signature date
        - If you see multiple dates, choose the one in the requestor/billing section
        - If uncertain, look for "Dated Signed by Requestor" or similar field

        Return your findings in JSON format:
        {
        "requestor_signature_extraction": {
            "requestor_signature_date": "MM/DD/YYYY or null",
            "date_found": true/false,
            "confidence": 0-100,
            "location_description": "Description of where the date was found",
            "reasoning": "Explanation of how you determined this is the requestor date and not customer date"
        }
        }"""

        user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
        {requestor_date_prompt}

        [ATTACHED_IMAGE]
        data:image/png;base64,{base64_image}"""

        system_prompt = "You are an expert document analyzer specializing in extracting requestor/billing signature dates from LOA forms. Your task is to accurately identify the REQUESTOR/BILLING signature date, not customer dates."

        try:
            start_time = datetime.now()

            gpt_response_data = self.openai_4o_service.process_with_prompts(
                system_prompt, user_prompt, max_token=1000, raw_response=True
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            if gpt_response_data and gpt_response_data[0]["ai_result"]:
                analysis_text = gpt_response_data[0]["ai_result"][0]["result"]
            else:
                self.logger.error(
                    "No response from OpenAI service for requestor signature date extraction"
                )
                return {
                    "success": False,
                    "requestor_signature_date": None,
                    "error": "No response from OpenAI service",
                }

            json_match = re.search(r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(1))
            else:
                analysis_json = json.loads(analysis_text)

            self.logger.info(
                f"GPT-4o requestor signature date extraction completed in {processing_time:.2f} seconds"
            )

            verification_data = analysis_json.get("requestor_signature_extraction")
            if verification_data:
                requestor_date = verification_data.get("requestor_signature_date")
                date_found = verification_data.get("date_found", False)
                confidence = verification_data.get("confidence", 0)
                location = verification_data.get("location_description", "Unknown")
                reasoning = verification_data.get("reasoning", "No reasoning provided")

                self.logger.info("GPT-4o Requestor Signature Date Findings:")
                self.logger.info(f"      - Date Found: {date_found}")
                self.logger.info(f"      - Requestor Signature Date: {requestor_date}")
                self.logger.info(f"      - Confidence: {confidence}%")
                self.logger.info(f"      - Location: {location}")
                self.logger.info(f"      - Reasoning: {reasoning}")

                return {
                    "success": True,
                    "requestor_signature_date": requestor_date,
                    "date_found": date_found,
                    "confidence": confidence,
                    "location_description": location,
                    "reasoning": reasoning,
                    "gpt4o_verified": True,
                    "verification_details": verification_data,
                }
            else:
                self.logger.warning(
                    "GPT-4o did not return the expected 'requestor_signature_extraction' key."
                )
                return {
                    "success": False,
                    "requestor_signature_date": None,
                    "error": "Invalid JSON structure from GPT-4o",
                }

        except Exception as e:
            error_msg = f"GPT-4o requestor signature date extraction failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "requestor_signature_date": None,
                "error": f"Exception: {str(e)}",
            }

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def extract_signatures_with_gpt4o(self, pdf_path: str) -> Dict:
        """
        Use GPT-4o to detect signatures in a document, especially for cases where normal OCR might fail.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            Dict: Contains signature detection results for both customer and requestor signatures.
        """
        self.logger.info("Performing GPT-4o signature detection...")

        image_data = self.ocr_integration.extract_pdf_image(pdf_path, page_num=0)
        if not image_data:
            self.logger.error(
                "Failed to extract image from PDF for signature detection."
            )
            return {
                "success": False,
                "customer_signature_present": False,
                "requestor_signature_present": False,
                "error": "Failed to extract image from PDF",
            }

        base64_image = self.ocr_integration.encode_image_to_base64(image_data)

        signature_prompt = """CRITICAL: Analyze this document for signatures in both the customer section and the requestor/billing section.

        ULTRA-LENIENT SIGNATURE DETECTION - BE EXTREMELY PERMISSIVE:

        **IMPORTANT: This LOA may be from different utilities with different field names:**
        - New England documents: "Customer's Signature (please print)"
        - COMED (Illinois) documents: "Signature of Authorized Person"
        - Other utilities: "Customer Signature", "Authorized Person Signature", etc.

        **CRITICAL INSTRUCTION: DISTINGUISH BETWEEN PRINTED NAME AND ACTUAL SIGNATURE**

        **IMPORTANT: Documents have SEPARATE fields for "Printed Name" vs "Signature":**
        - "Authorized Person" field = PRINTED NAME (typed/written name of person)
        - "Signature of Authorized Person" field = ACTUAL SIGNATURE (handwritten mark or e-signature)

        SIGNATURES TO IDENTIFY:
        1. CUSTOMER SIGNATURE:
           - Located in "Customer Information" section
           - Look specifically for fields labeled:
             * "Signature of Authorized Person"
             * "Customer's Signature"
             * "Signature" (not "Authorized Person" or "Name")

           - **ACCEPT AS VALID SIGNATURE (mark as PRESENT):**
             * **HANDWRITTEN signature marks** - cursive writing, scribbles, pen marks
             * **HANDWRITTEN initials** - e.g., "JD", "AB" written by hand
             * **E-SIGNATURE MARKERS with authentication**:
               - "E-Signed: [name] [timestamp]"
               - "DocuSign: [name]"
               - "Electronically signed by [name]"
             * **VISUAL SIGNATURE EVIDENCE** - you can SEE handwritten marks in the signature field

           - **DO NOT ACCEPT AS SIGNATURE (mark as MISSING):**
             * **TYPED NAMES** in "Authorized Person" or "Printed Name" fields
             * **FIELD LABELS** like "Signature of Authorized Person", "Signature"
             * **CHECKBOX TEXT** like "as agent", "By checking this box"
             * **BLANK/EMPTY FIELDS** with no marks
             * **PLACEHOLDER TEXT** like "________" or "please sign here"

           - **CRITICAL DISTINCTION RULE:**
             * If you see "Eamon O'Malley" in the "Authorized Person" field → That's the PRINTED NAME, NOT a signature
             * If you see "Eamon O'Malley" in the "Signature" field → Check if it's handwritten or typed
             * HANDWRITTEN in signature field = SIGNATURE (mark as PRESENT)
             * TYPED/PRINTED in signature field = NOT A SIGNATURE (mark as MISSING)
             * **ONLY handwritten marks, initials, or e-signature markers count as signatures**

           - **IF UNCERTAIN:** Look at the visual characteristics
             * Handwriting (cursive, varied pen strokes) = SIGNATURE
             * Typed/printed text (uniform font) = NOT a signature
             * E-signature marker with timestamp = SIGNATURE
             * Just a name without signature characteristics = NOT a signature

        2. REQUESTOR/BILLING SIGNATURE:
           - Located in "Requestor & Billing Information" section
           - Look for "Requestor/Billing Signature" field
           - **SAME LENIENT RULES:** ANY text or marks = valid signature
           - E-signature markers in requestor section also count as valid signature

        **BROKER SIGNATURE DETECTION - CRITICAL:**
        - If you see "as agent", "as Agent", "Authorized Agent", or similar in the Customer Information section, this indicates a BROKER signed
        - In this case, still mark signature as PRESENT, but note in the text_found field that it says "as agent"
        - The validation system will handle broker rejection separately

        **CRITICAL VALIDATION RULES - FAVOR DETECTING SIGNATURES:**
        - If in doubt, mark signature as PRESENT rather than missing
        - Truncated text like "Signatur", "aten", "Sig" = VALID signature
        - Typed names in signature field = VALID signature
        - Handwritten initials = VALID signature
        - Partial text = VALID signature
        - Faint or light marks = VALID signature
        - Electronic signature markers = VALID signature
        - E-signature text ANYWHERE in customer section = VALID signature

        **ONLY mark as MISSING if field is visually empty with no marks at all AND no e-signature markers in section**

        For EACH signature field, determine:
        1. Is there ANY visible mark, text, or writing in the signature field area?
        2. Is there an e-signature marker ANYWHERE in the customer/requestor section?
        3. What text/marks appear (even if partial, faint, or fragmentary)?
        4. Where is this located on the document?
        5. Does it contain "as agent" or similar broker indicators?

        Return your findings in JSON format:
        {
        "signature_detection": {
            "customer_signature": {
                "present": true|false,
                "text_found": "Exact text you found in signature field",
                "confidence": 0-100,
                "location": "Description of where found"
            },
            "requestor_signature": {
                "present": true|false,
                "text_found": "Exact text you found in signature field",
                "confidence": 0-100,
                "location": "Description of where found"
            },
            "reasoning": "Your explanation of your findings"
        }
        }"""

        # Create the user prompt for vision analysis
        user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
        {signature_prompt}

        [ATTACHED_IMAGE]
        data:image/png;base64,{base64_image}"""

        # System prompt for GPT-4o vision
        system_prompt = "You are an expert document analyzer specializing in signature detection. Your task is to identify ANY text or marks in signature fields, being extremely lenient about what counts as a signature."

        try:
            start_time = datetime.now()

            # Use the production OpenAI service
            gpt_response_data = self.openai_4o_service.process_with_prompts(
                system_prompt, user_prompt, max_token=1000, raw_response=True
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Extract the actual response content from the service response
            if gpt_response_data and gpt_response_data[0]["ai_result"]:
                analysis_text = gpt_response_data[0]["ai_result"][0]["result"]
            else:
                self.logger.error(
                    "No response from OpenAI service for signature detection"
                )
                return {
                    "success": False,
                    "customer_signature_present": False,
                    "requestor_signature_present": False,
                    "error": "No response from OpenAI service",
                }

            json_match = re.search(r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(1))
            else:
                analysis_json = json.loads(analysis_text)

            self.logger.info(
                f"GPT-4o signature detection completed in {processing_time:.2f} seconds"
            )

            signature_data = analysis_json.get("signature_detection")
            if signature_data:
                customer_sig = signature_data.get("customer_signature", {})
                requestor_sig = signature_data.get("requestor_signature", {})

                customer_present = customer_sig.get("present", False)
                customer_text = customer_sig.get("text_found", "")
                customer_confidence = customer_sig.get("confidence", 0)
                customer_location = customer_sig.get("location", "Unknown")

                requestor_present = requestor_sig.get("present", False)
                requestor_text = requestor_sig.get("text_found", "")
                requestor_confidence = requestor_sig.get("confidence", 0)
                requestor_location = requestor_sig.get("location", "Unknown")

                reasoning = signature_data.get("reasoning", "No reasoning provided")

                self.logger.info("GPT-4o Signature Detection Findings:")
                self.logger.info(
                    f"      - Customer Signature Present: {customer_present}"
                )
                self.logger.info(f"      - Customer Signature Text: '{customer_text}'")
                self.logger.info(f"      - Customer Confidence: {customer_confidence}%")
                self.logger.info(
                    f"      - Customer Signature Location: {customer_location}"
                )
                self.logger.info(
                    f"      - Requestor Signature Present: {requestor_present}"
                )
                self.logger.info(
                    f"      - Requestor Signature Text: '{requestor_text}'"
                )
                self.logger.info(
                    f"      - Requestor Confidence: {requestor_confidence}%"
                )
                self.logger.info(
                    f"      - Requestor Signature Location: {requestor_location}"
                )
                self.logger.info(f"      - Reasoning: {reasoning}")

                return {
                    "success": True,
                    "customer_signature_present": customer_present,
                    "customer_signature_text": customer_text,
                    "customer_confidence": customer_confidence,
                    "customer_location": customer_location,
                    "requestor_signature_present": requestor_present,
                    "requestor_signature_text": requestor_text,
                    "requestor_confidence": requestor_confidence,
                    "requestor_location": requestor_location,
                    "reasoning": reasoning,
                    "gpt4o_verified": True,
                    "verification_details": signature_data,
                }
            else:
                self.logger.warning(
                    "GPT-4o did not return the expected 'signature_detection' key."
                )
                return {
                    "success": False,
                    "customer_signature_present": False,
                    "requestor_signature_present": False,
                    "error": "Invalid JSON structure from GPT-4o",
                }

        except Exception as e:
            error_msg = f"GPT-4o signature detection failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "customer_signature_present": False,
                "requestor_signature_present": False,
                "error": f"Exception: {str(e)}",
            }

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def verify_ne_service_options_with_gpt4o(
        self, pdf_path: str, extraction_log: Dict
    ) -> Dict:
        """
        Second pass verification for New England service options checkboxes using GPT-4o.
        This is triggered when the initial layout analysis finds an ambiguous selection (0 or >1 options selected).

        Args:
            pdf_path (str): Path to the PDF file.
            extraction_log (Dict): The current extraction log from Azure Layout.

        Returns:
            Dict: Updated extraction log with verified service option states.
        """
        self.logger.info(
            "Performing GPT-4o fallback for New England service options..."
        )

        image_data = self.ocr_integration.extract_pdf_image(pdf_path, page_num=0)
        if not image_data:
            self.logger.error(
                "Failed to extract image from PDF for service options verification."
            )
            return extraction_log

        base64_image = self.ocr_integration.encode_image_to_base64(image_data)

        service_options_prompt = """CRITICAL: Carefully examine this document image to determine which New England service options are selected.

        The TWO options are:
        1. "One Time Request, $50.00 per account number"
        2. "Annual Subscription, $300.00 per account per year"

        STEP-BY-STEP CHECKBOX DETECTION PROCESS:

        STEP 1: LOCATE THE SERVICE OPTIONS SECTION
        - Find the section with these two pricing options
        - This is typically in the upper portion of the document
        - Each option should have a small checkbox (square box) to its left

        STEP 2: EXAMINE EACH CHECKBOX CAREFULLY
        For EACH checkbox, determine if it is:
        - SELECTED: Contains an X, checkmark (✓), filled square (■), or any mark inside
        - UNSELECTED: Empty box (☐), blank, or contains only a light outline

        VISUAL PATTERNS TO IDENTIFY:
        - Selected checkbox: ☑ ☒ ✓ X [X] ■ or any visible mark inside the box
        - Unselected checkbox: ☐ [ ] or completely empty square
        - BE VERY CAREFUL: An empty square = UNSELECTED, not selected!

        STEP 3: COUNT THE SELECTIONS
        - Count how many checkboxes have marks inside them
        - If NO marks visible in either box = 0 selected (INVALID - must reject)
        - If marks in BOTH boxes = 2 selected (INVALID - must reject)
        - If mark in ONLY ONE box = 1 selected (VALID - correct)

        STEP 4: VERIFY YOUR ANALYSIS
        - Double-check: Are you CERTAIN you saw a mark inside the checkbox?
        - Don't confuse the checkbox outline with a filled checkbox
        - Don't assume a checkbox is selected without seeing a clear mark

        COMMON MISTAKES TO AVOID:
        - DON'T mark as selected if the box is empty (common error!)
        - DON'T confuse checkbox borders with checkmarks
        - DON'T assume both are selected unless you see TWO distinct marks
        - DON'T assume one is selected just because text exists near it

        IMPORTANT: If you cannot see ANY marks in ANY checkboxes, report selection_count=0

        Return your findings in JSON format:
        {
        "service_options_verification": {
            "one_time_request_selected": true/false,
            "annual_subscription_selected": true/false,
            "selection_count": 0, 1, or 2,
            "confidence": 0-100,
            "reasoning": "DETAILED explanation: For EACH option, describe what you see in the checkbox (empty box, X mark, checkmark, etc.) and why you determined it is selected or unselected."
        }
        }"""

        # Create the user prompt for vision analysis
        user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
        {service_options_prompt}

        [ATTACHED_IMAGE]
        data:image/png;base64,{base64_image}"""

        # System prompt for GPT-4o vision
        system_prompt = "You are an expert document analyzer specializing in form verification for the energy sector. Your task is to accurately determine the state of specific checkboxes in New England LOA documents."

        try:
            start_time = datetime.now()

            # Use the production OpenAI service
            gpt_response_data = self.openai_4o_service.process_with_prompts(
                system_prompt, user_prompt, max_token=1000, raw_response=True
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Extract the actual response content from the service response
            if gpt_response_data and gpt_response_data[0]["ai_result"]:
                analysis_text = gpt_response_data[0]["ai_result"][0]["result"]
            else:
                self.logger.error(
                    "No response from OpenAI service for service options verification"
                )
                extraction_log["service_options"][
                    "gpt4o_verification_error"
                ] = "No response from OpenAI service"
                return extraction_log

            json_match = re.search(r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(1))
            else:
                analysis_json = json.loads(analysis_text)

            self.logger.info(
                f"GPT-4o service options verification completed in {processing_time:.2f} seconds"
            )

            verification_data = analysis_json.get("service_options_verification")
            if verification_data:
                one_time_selected = verification_data.get(
                    "one_time_request_selected", False
                )
                annual_selected = verification_data.get(
                    "annual_subscription_selected", False
                )
                selection_count = int(one_time_selected) + int(annual_selected)

                self.logger.info("GPT-4o Findings:")
                self.logger.info(
                    f"      - One Time Request: {'Selected' if one_time_selected else 'Not Selected'}"
                )
                self.logger.info(
                    f"      - Annual Subscription: {'Selected' if annual_selected else 'Not Selected'}"
                )
                self.logger.info(f"      - Final Selection Count: {selection_count}")

                # Override the original service_options in the extraction log
                extraction_log["service_options"][
                    "one_time_selected"
                ] = one_time_selected
                extraction_log["service_options"][
                    "annual_subscription_selected"
                ] = annual_selected
                extraction_log["service_options"]["selection_count"] = selection_count
                extraction_log["service_options"]["gpt4o_verified"] = True
                extraction_log["service_options"][
                    "gpt4o_verification_details"
                ] = verification_data

                self.logger.info(
                    "Overwrote initial service option detection with GPT-4o results."
                )
            else:
                self.logger.warning(
                    "GPT-4o did not return the expected 'service_options_verification' key."
                )
                extraction_log["service_options"][
                    "gpt4o_verification_error"
                ] = "Invalid JSON structure from GPT-4o"

        except Exception as e:
            error_msg = f"GPT-4o service options verification failed: {str(e)}"
            self.logger.error(error_msg)
            extraction_log["service_options"][
                "gpt4o_verification_error"
            ] = f"Exception: {str(e)}"

        return extraction_log

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def verify_meco_subscription_options_with_gpt4o(
        self, pdf_path: str, extraction_log: Dict
    ) -> Dict:
        """
        GPT-4o verification for MECO subscription options checkboxes.
        MECO has 3 subscription options and exactly ONE must be selected.

        Args:
            pdf_path (str): Path to the PDF file.
            extraction_log (Dict): The current extraction log from Azure Layout.

        Returns:
            Dict: Updated extraction log with verified MECO subscription option states.
        """
        self.logger.info(
            "Performing GPT-4o verification for MECO subscription options..."
        )

        image_data = self.ocr_integration.extract_pdf_image(pdf_path, page_num=0)
        if not image_data:
            self.logger.error(
                "Failed to extract image from PDF for MECO subscription options verification."
            )
            return extraction_log

        base64_image = self.ocr_integration.encode_image_to_base64(image_data)

        meco_options_prompt = """CRITICAL: Carefully examine this document image to determine which MECO subscription options are selected.

        The document should have a section titled "Type of Interval Data Request" with THREE options:
        1. "Two Weeks Online Access to Data" (with pricing)
        2. "One Year Online Access to Data" (with pricing)
        3. "Auto-Renewing, One Year Online Access to Data" (with pricing)

        IMPORTANT VISUAL CLUES TO LOOK FOR:
        - Look for checkboxes, X marks, checkmarks, or filled boxes near each option
        - Look for text patterns indicating selection
        - Some documents may have selection marks to the left or right of the option text
        - Pay attention to visual differences between the three options

        VALIDATION REQUIREMENT:
        - EXACTLY ONE option must be selected
        - If 0 options are selected = INVALID
        - If 2 or 3 options are selected = INVALID
        - If exactly 1 option is selected = VALID

        Return your findings in JSON format:
        {
        "meco_subscription_verification": {
            "two_weeks_selected": true/false,
            "one_year_selected": true/false,
            "auto_renewing_selected": true/false,
            "selection_count": 0, 1, 2, or 3,
            "confidence": 0-100,
            "reasoning": "Detailed explanation of what you observed - describe which option(s) appear selected and why."
        }
        }"""

        # Create the user prompt for vision analysis
        user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
        {meco_options_prompt}

        [ATTACHED_IMAGE]
        data:image/png;base64,{base64_image}"""

        # System prompt for GPT-4o vision
        system_prompt = "You are an expert document analyzer specializing in MECO (Massachusetts Electric) LOA forms. Your task is to accurately determine which of the 3 subscription options are selected in the 'Type of Interval Data Request' section."

        try:
            start_time = datetime.now()

            # Use the production OpenAI service
            gpt_response_data = self.openai_4o_service.process_with_prompts(
                system_prompt, user_prompt, max_token=1000, raw_response=True
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Extract the actual response content from the service response
            if gpt_response_data and gpt_response_data[0]["ai_result"]:
                analysis_text = gpt_response_data[0]["ai_result"][0]["result"]
            else:
                self.logger.error(
                    "No response from OpenAI service for MECO subscription options verification"
                )
                if "meco_subscription_options" not in extraction_log:
                    extraction_log["meco_subscription_options"] = {}
                extraction_log["meco_subscription_options"][
                    "gpt4o_verification_error"
                ] = "No response from OpenAI service"
                return extraction_log

            json_match = re.search(r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(1))
            else:
                analysis_json = json.loads(analysis_text)

            self.logger.info(
                f"GPT-4o MECO subscription options verification completed in {processing_time:.2f} seconds"
            )

            verification_data = analysis_json.get("meco_subscription_verification")
            if verification_data:
                two_weeks_selected = verification_data.get("two_weeks_selected", False)
                one_year_selected = verification_data.get("one_year_selected", False)
                auto_renewing_selected = verification_data.get(
                    "auto_renewing_selected", False
                )
                selection_count = (
                    int(two_weeks_selected)
                    + int(one_year_selected)
                    + int(auto_renewing_selected)
                )

                self.logger.info("GPT-4o MECO Findings:")
                self.logger.info(
                    f"      - Two Weeks Online: {'Selected' if two_weeks_selected else 'Not Selected'}"
                )
                self.logger.info(
                    f"      - One Year Online: {'Selected' if one_year_selected else 'Not Selected'}"
                )
                self.logger.info(
                    f"      - Auto-Renewing: {'Selected' if auto_renewing_selected else 'Not Selected'}"
                )
                self.logger.info(f"      - Final Selection Count: {selection_count}")

                # Initialize meco_subscription_options if not present
                if "meco_subscription_options" not in extraction_log:
                    extraction_log["meco_subscription_options"] = {}

                # Override the original meco_subscription_options in the extraction log
                extraction_log["meco_subscription_options"][
                    "two_weeks_selected"
                ] = two_weeks_selected
                extraction_log["meco_subscription_options"][
                    "one_year_selected"
                ] = one_year_selected
                extraction_log["meco_subscription_options"][
                    "auto_renewing_selected"
                ] = auto_renewing_selected
                extraction_log["meco_subscription_options"][
                    "selection_count"
                ] = selection_count
                extraction_log["meco_subscription_options"]["gpt4o_verified"] = True
                extraction_log["meco_subscription_options"][
                    "gpt4o_verification_details"
                ] = verification_data

                self.logger.info(
                    "Overwrote initial MECO subscription detection with GPT-4o results."
                )
            else:
                self.logger.warning(
                    "GPT-4o did not return the expected 'meco_subscription_verification' key."
                )
                if "meco_subscription_options" not in extraction_log:
                    extraction_log["meco_subscription_options"] = {}
                extraction_log["meco_subscription_options"][
                    "gpt4o_verification_error"
                ] = "Invalid JSON structure from GPT-4o"

        except Exception as e:
            error_msg = (
                f"GPT-4o MECO subscription options verification failed: {str(e)}"
            )
            self.logger.error(error_msg)
            if "meco_subscription_options" not in extraction_log:
                extraction_log["meco_subscription_options"] = {}
            extraction_log["meco_subscription_options"][
                "gpt4o_verification_error"
            ] = f"Exception: {str(e)}"

        return extraction_log

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def verify_neco_subscription_options_with_gpt4o(
        self, pdf_path: str, extraction_log: Dict
    ) -> Dict:
        """
        GPT-4o verification for NECO subscription options checkboxes.
        NECO has 2 subscription options and exactly ONE must be selected.

        Args:
            pdf_path (str): Path to the PDF file.
            extraction_log (Dict): The current extraction log from Azure Layout.

        Returns:
            Dict: Updated extraction log with verified NECO subscription option states.
        """
        self.logger.info(
            "Performing GPT-4o verification for NECO subscription options..."
        )

        image_data = self.ocr_integration.extract_pdf_image(pdf_path, page_num=0)
        if not image_data:
            self.logger.error(
                "Failed to extract image from PDF for NECO subscription options verification."
            )
            return extraction_log

        base64_image = self.ocr_integration.encode_image_to_base64(image_data)

        neco_options_prompt = """CRITICAL: Carefully examine this document image to determine which NECO subscription options are selected.

        The document should have a section titled "Type of Interval Data Request - Please choose 1 ONLY" with TWO options:
        1. "Two Weeks Online Access to data" (with pricing)
        2. "One Year Online Access to Data" (with pricing)

        **CRITICAL CHECKBOX DETECTION RULES - FOLLOW EXACTLY:**

        STEP 1: LOCATE THE CHECKBOXES
        - Find the small square boxes to the LEFT of each option text
        - Each option should have its own checkbox

        STEP 2: EXAMINE EACH CHECKBOX CAREFULLY
        For EACH checkbox, look INSIDE the box and determine:

        **SELECTED** checkbox shows:
        ✓ An X mark inside the box (☒)
        ✓ A checkmark inside the box (☑)
        ✓ A filled/shaded box (■)
        ✓ ANY visible mark, symbol, or fill INSIDE the checkbox

        **UNSELECTED** checkbox shows:
        ✗ EMPTY box with only border/outline (☐)
        ✗ Blank square with NO marks inside
        ✗ Just the checkbox border with white/empty interior

        **CRITICAL WARNING - COMMON FALSE POSITIVE:**
        - DO NOT confuse the checkbox BORDER with a filled checkbox
        - An empty box ☐ (border only) = UNSELECTED, NOT selected!
        - You MUST see a mark INSIDE the box to call it selected
        - If the checkbox interior is blank/white = UNSELECTED

        STEP 3: VERIFY YOUR ANALYSIS
        - For EACH option, ask: "Do I see a mark INSIDE this checkbox?"
        - If answer is NO = checkbox is UNSELECTED
        - If answer is YES = checkbox is SELECTED
        - Double-check before finalizing your count

        STEP 4: COUNT SELECTIONS
        - Count only checkboxes with marks INSIDE them
        - selection_count = number of checkboxes with internal marks
        - If BOTH checkboxes are empty = selection_count MUST be 0

        **BE EXTREMELY CONSERVATIVE:**
        - When in doubt, mark checkbox as UNSELECTED
        - Only mark as SELECTED if you clearly see a mark inside
        - Empty checkbox ≠ Selected checkbox

        Return your findings in JSON format:
        {
        "neco_subscription_verification": {
            "two_weeks_selected": true/false,
            "one_year_selected": true/false,
            "selection_count": 0, 1, or 2,
            "confidence": 0-100,
            "reasoning": "For EACH option, describe EXACTLY what you see INSIDE the checkbox: Is it empty/blank, or does it contain a mark? Be specific."
        }
        }"""

        # Create the user prompt for vision analysis
        user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
        {neco_options_prompt}

        [ATTACHED_IMAGE]
        data:image/png;base64,{base64_image}"""

        # System prompt for GPT-4o vision
        system_prompt = "You are an expert document analyzer specializing in NECO (Narragansett Electric) LOA forms. Your task is to accurately determine which of the 2 subscription options are selected in the 'Type of Interval Data Request' section."

        try:
            start_time = datetime.now()

            # Use the production OpenAI service
            gpt_response_data = self.openai_4o_service.process_with_prompts(
                system_prompt, user_prompt, max_token=1000, raw_response=True
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Extract the actual response content from the service response
            if gpt_response_data and gpt_response_data[0]["ai_result"]:
                analysis_text = gpt_response_data[0]["ai_result"][0]["result"]
            else:
                self.logger.error(
                    "No response from OpenAI service for NECO subscription options verification"
                )
                if "neco_subscription_options" not in extraction_log:
                    extraction_log["neco_subscription_options"] = {}
                extraction_log["neco_subscription_options"][
                    "gpt4o_verification_error"
                ] = "No response from OpenAI service"
                return extraction_log

            json_match = re.search(r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(1))
            else:
                analysis_json = json.loads(analysis_text)

            self.logger.info(
                f"GPT-4o NECO subscription options verification completed in {processing_time:.2f} seconds"
            )

            verification_data = analysis_json.get("neco_subscription_verification")
            if verification_data:
                two_weeks_selected = verification_data.get("two_weeks_selected", False)
                one_year_selected = verification_data.get("one_year_selected", False)
                selection_count = int(two_weeks_selected) + int(one_year_selected)

                self.logger.info("GPT-4o NECO Findings:")
                self.logger.info(
                    f"      - Two Weeks Online: {'Selected' if two_weeks_selected else 'Not Selected'}"
                )
                self.logger.info(
                    f"      - One Year Online: {'Selected' if one_year_selected else 'Not Selected'}"
                )
                self.logger.info(f"      - Final Selection Count: {selection_count}")

                # Initialize neco_subscription_options if not present
                if "neco_subscription_options" not in extraction_log:
                    extraction_log["neco_subscription_options"] = {}

                # Override the original neco_subscription_options in the extraction log
                extraction_log["neco_subscription_options"][
                    "two_weeks_selected"
                ] = two_weeks_selected
                extraction_log["neco_subscription_options"][
                    "one_year_selected"
                ] = one_year_selected
                extraction_log["neco_subscription_options"][
                    "selection_count"
                ] = selection_count
                extraction_log["neco_subscription_options"]["gpt4o_verified"] = True
                extraction_log["neco_subscription_options"][
                    "gpt4o_verification_details"
                ] = verification_data

                self.logger.info(
                    "Overwrote initial NECO subscription detection with GPT-4o results."
                )
            else:
                self.logger.warning(
                    "GPT-4o did not return the expected 'neco_subscription_verification' key."
                )
                if "neco_subscription_options" not in extraction_log:
                    extraction_log["neco_subscription_options"] = {}
                extraction_log["neco_subscription_options"][
                    "gpt4o_verification_error"
                ] = "Invalid JSON structure from GPT-4o"

        except Exception as e:
            error_msg = (
                f"GPT-4o NECO subscription options verification failed: {str(e)}"
            )
            self.logger.error(error_msg)
            if "neco_subscription_options" not in extraction_log:
                extraction_log["neco_subscription_options"] = {}
            extraction_log["neco_subscription_options"][
                "gpt4o_verification_error"
            ] = f"Exception: {str(e)}"

        return extraction_log

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def verify_nhec_request_type_options_with_gpt4o(
        self, pdf_path: str, extraction_log: Dict
    ) -> Dict:
        """
        GPT-4o verification for NHEC request type options checkboxes.
        NHEC has 2 request type options and exactly ONE must be selected.

        Args:
            pdf_path (str): Path to the PDF file.
            extraction_log (Dict): The current extraction log from Azure Layout.

        Returns:
            Dict: Updated extraction log with verified NHEC request type option states.
        """
        self.logger.info(
            "Performing GPT-4o verification for NHEC request type options..."
        )

        image_data = self.ocr_integration.extract_pdf_image(pdf_path, page_num=0)
        if not image_data:
            self.logger.error(
                "Failed to extract image from PDF for NHEC request type options verification."
            )
            return extraction_log

        base64_image = self.ocr_integration.encode_image_to_base64(image_data)

        nhec_options_prompt = """CRITICAL: Carefully examine this document image to determine which NHEC request type options are selected.

        The document should have a section titled "Request Type (Select One)" with TWO options:
        1. "Ad-hoc Request for Historic Data" (with date range fields)
        2. "Subscription Request for Future Data" (with date fields)

        IMPORTANT VISUAL CLUES TO LOOK FOR:
        - Look for checkboxes, X marks, checkmarks, or filled boxes near each option
        - Look for text patterns indicating selection
        - Some documents may have selection marks to the left or right of the option text
        - Pay attention to visual differences between the two options

        VALIDATION REQUIREMENT:
        - EXACTLY ONE option must be selected
        - If 0 options are selected = INVALID
        - If 2 options are selected = INVALID
        - If exactly 1 option is selected = VALID

        Return your findings in JSON format:
        {
        "nhec_request_type_verification": {
            "adhoc_selected": true/false,
            "subscription_selected": true/false,
            "selection_count": 0, 1, or 2,
            "confidence": 0-100,
            "reasoning": "Detailed explanation of what you observed - describe which option(s) appear selected and why."
        }
        }"""

        # Create the user prompt for vision analysis
        user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
        {nhec_options_prompt}

        [ATTACHED_IMAGE]
        data:image/png;base64,{base64_image}"""

        # System prompt for GPT-4o vision
        system_prompt = "You are an expert document analyzer specializing in NHEC (New Hampshire Electric Co-op) LOA forms. Your task is to accurately determine which of the 2 request type options are selected in the 'Request Type' section."

        try:
            start_time = datetime.now()

            # Use the production OpenAI service
            gpt_response_data = self.openai_4o_service.process_with_prompts(
                system_prompt, user_prompt, max_token=1000, raw_response=True
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Extract the actual response content from the service response
            if gpt_response_data and gpt_response_data[0]["ai_result"]:
                analysis_text = gpt_response_data[0]["ai_result"][0]["result"]
            else:
                self.logger.error(
                    "No response from OpenAI service for NHEC request type options verification"
                )
                if "nhec_request_type_options" not in extraction_log:
                    extraction_log["nhec_request_type_options"] = {}
                extraction_log["nhec_request_type_options"][
                    "gpt4o_verification_error"
                ] = "No response from OpenAI service"
                return extraction_log

            json_match = re.search(r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(1))
            else:
                analysis_json = json.loads(analysis_text)

            self.logger.info(
                f"GPT-4o NHEC request type options verification completed in {processing_time:.2f} seconds"
            )

            verification_data = analysis_json.get("nhec_request_type_verification")
            if verification_data:
                adhoc_selected = verification_data.get("adhoc_selected", False)
                subscription_selected = verification_data.get(
                    "subscription_selected", False
                )
                selection_count = int(adhoc_selected) + int(subscription_selected)

                self.logger.info("GPT-4o NHEC Findings:")
                self.logger.info(
                    f"      - Ad-hoc Request: {'Selected' if adhoc_selected else 'Not Selected'}"
                )
                self.logger.info(
                    f"      - Subscription Request: {'Selected' if subscription_selected else 'Not Selected'}"
                )
                self.logger.info(f"      - Final Selection Count: {selection_count}")

                # Initialize nhec_request_type_options if not present
                if "nhec_request_type_options" not in extraction_log:
                    extraction_log["nhec_request_type_options"] = {}

                # Override the original nhec_request_type_options in the extraction log
                extraction_log["nhec_request_type_options"][
                    "adhoc_selected"
                ] = adhoc_selected
                extraction_log["nhec_request_type_options"][
                    "subscription_selected"
                ] = subscription_selected
                extraction_log["nhec_request_type_options"][
                    "selection_count"
                ] = selection_count
                extraction_log["nhec_request_type_options"]["gpt4o_verified"] = True
                extraction_log["nhec_request_type_options"][
                    "gpt4o_verification_details"
                ] = verification_data

                self.logger.info(
                    "Overwrote initial NHEC request type detection with GPT-4o results."
                )
            else:
                self.logger.warning(
                    "GPT-4o did not return the expected 'nhec_request_type_verification' key."
                )
                if "nhec_request_type_options" not in extraction_log:
                    extraction_log["nhec_request_type_options"] = {}
                extraction_log["nhec_request_type_options"][
                    "gpt4o_verification_error"
                ] = "Invalid JSON structure from GPT-4o"

        except Exception as e:
            error_msg = (
                f"GPT-4o NHEC request type options verification failed: {str(e)}"
            )
            self.logger.error(error_msg)
            if "nhec_request_type_options" not in extraction_log:
                extraction_log["nhec_request_type_options"] = {}
            extraction_log["nhec_request_type_options"][
                "gpt4o_verification_error"
            ] = f"Exception: {str(e)}"

        return extraction_log

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def verify_psnh_subscription_options_with_gpt4o(
        self, pdf_path: str, extraction_log: Dict
    ) -> Dict:
        """
        GPT-4o verification for PSNH subscription options checkboxes.
        PSNH has 3 subscription options and exactly ONE must be selected.

        Args:
            pdf_path (str): Path to the PDF file.
            extraction_log (Dict): The current extraction log from Azure Layout.

        Returns:
            Dict: Updated extraction log with verified PSNH subscription option states.
        """
        self.logger.info(
            "Performing GPT-4o verification for PSNH subscription options..."
        )

        image_data = self.ocr_integration.extract_pdf_image(pdf_path, page_num=0)
        if not image_data:
            self.logger.error(
                "Failed to extract image from PDF for PSNH subscription options verification."
            )
            return extraction_log

        base64_image = self.ocr_integration.encode_image_to_base64(image_data)

        psnh_options_prompt = """CRITICAL: Carefully examine this document image to determine which PSNH (Public Service of New Hampshire - Eversource) subscription options are selected.

        The document should have a section titled "Service Options - select one" with THREE options:
        1. "One Time Request, $50.00 per account number" (All interval data available at the time of request will be provided online. Data will not be updated. The user id and password will expire 30 days after the start of service.)
        2. "Annual Subscription, $300.00 per account number" (All interval data available at the time of the request will be provided online. For phone access meters, data will be updated daily. Data may be delayed due to meter or communication difficulties. The subscription is automatically renewed and billed each year.)
        3. "Annual subscription, $25 per account number per month" (All interval data available at the time of the request will be provided online. For phone access meters, data will be updated daily. Data may be delayed due to meter or communication difficulties. The subscription automatically renews each year and bills monthly, typically within 30 to 60 days after initial sign up)

        IMPORTANT VISUAL CLUES TO LOOK FOR:
        - Look for checkboxes, X marks, checkmarks, or filled boxes near each option
        - Look for text patterns indicating selection (like ":selected:" markers)
        - Some documents may have selection marks to the left or right of the option text
        - Pay attention to visual differences between the three options
        - The checkbox might be a small square box that when selected shows an X or checkmark

        VALIDATION REQUIREMENT:
        - EXACTLY ONE option must be selected
        - If 0 options are selected = INVALID
        - If 2 or 3 options are selected = INVALID
        - If exactly 1 option is selected = VALID

        Return your findings in JSON format:
        {
        "psnh_subscription_verification": {
            "one_time_request_selected": true/false,
            "annual_subscription_300_selected": true/false,
            "annual_subscription_monthly_selected": true/false,
            "selection_count": 0, 1, 2, or 3,
            "confidence": 0-100,
            "reasoning": "Detailed explanation of what you observed - describe which option(s) appear selected and why."
        }
        }"""

        # Create the user prompt for vision analysis
        user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
        {psnh_options_prompt}

        [ATTACHED_IMAGE]
        data:image/png;base64,{base64_image}"""

        # System prompt for GPT-4o vision
        system_prompt = "You are an expert document analyzer specializing in PSNH (Public Service of New Hampshire - Eversource) LOA forms. Your task is to accurately determine which of the 3 subscription options are selected in the 'Service Options' section."

        try:
            start_time = datetime.now()

            # Use the production OpenAI service
            gpt_response_data = self.openai_4o_service.process_with_prompts(
                system_prompt, user_prompt, max_token=1000, raw_response=True
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # Extract the actual response content from the service response
            if gpt_response_data and gpt_response_data[0]["ai_result"]:
                analysis_text = gpt_response_data[0]["ai_result"][0]["result"]
            else:
                self.logger.error(
                    "No response from OpenAI service for PSNH subscription options verification"
                )
                if "psnh_subscription_options" not in extraction_log:
                    extraction_log["psnh_subscription_options"] = {}
                extraction_log["psnh_subscription_options"][
                    "gpt4o_verification_error"
                ] = "No response from OpenAI service"
                return extraction_log

            json_match = re.search(r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(1))
            else:
                analysis_json = json.loads(analysis_text)

            self.logger.info(
                f"GPT-4o PSNH subscription options verification completed in {processing_time:.2f} seconds"
            )

            verification_data = analysis_json.get("psnh_subscription_verification")
            if verification_data:
                one_time_selected = verification_data.get(
                    "one_time_request_selected", False
                )
                annual_300_selected = verification_data.get(
                    "annual_subscription_300_selected", False
                )
                annual_monthly_selected = verification_data.get(
                    "annual_subscription_monthly_selected", False
                )
                selection_count = (
                    int(one_time_selected)
                    + int(annual_300_selected)
                    + int(annual_monthly_selected)
                )

                self.logger.info("GPT-4o PSNH Findings:")
                self.logger.info(
                    f"      - One Time Request ($50): {'Selected' if one_time_selected else 'Not Selected'}"
                )
                self.logger.info(
                    f"      - Annual Subscription ($300): {'Selected' if annual_300_selected else 'Not Selected'}"
                )
                self.logger.info(
                    f"      - Annual Subscription Monthly ($25/month): {'Selected' if annual_monthly_selected else 'Not Selected'}"
                )
                self.logger.info(f"      - Final Selection Count: {selection_count}")

                # Initialize psnh_subscription_options if not present
                if "psnh_subscription_options" not in extraction_log:
                    extraction_log["psnh_subscription_options"] = {}

                # Override the original psnh_subscription_options in the extraction log
                extraction_log["psnh_subscription_options"][
                    "one_time_request_selected"
                ] = one_time_selected
                extraction_log["psnh_subscription_options"][
                    "annual_subscription_300_selected"
                ] = annual_300_selected
                extraction_log["psnh_subscription_options"][
                    "annual_subscription_monthly_selected"
                ] = annual_monthly_selected
                extraction_log["psnh_subscription_options"][
                    "selection_count"
                ] = selection_count
                extraction_log["psnh_subscription_options"]["gpt4o_verified"] = True
                extraction_log["psnh_subscription_options"][
                    "gpt4o_verification_details"
                ] = verification_data

                self.logger.info(
                    "Overwrote initial PSNH subscription detection with GPT-4o results."
                )
            else:
                self.logger.warning(
                    "GPT-4o did not return the expected 'psnh_subscription_verification' key."
                )
                if "psnh_subscription_options" not in extraction_log:
                    extraction_log["psnh_subscription_options"] = {}
                extraction_log["psnh_subscription_options"][
                    "gpt4o_verification_error"
                ] = "Invalid JSON structure from GPT-4o"

        except Exception as e:
            error_msg = (
                f"GPT-4o PSNH subscription options verification failed: {str(e)}"
            )
            self.logger.error(error_msg)
            if "psnh_subscription_options" not in extraction_log:
                extraction_log["psnh_subscription_options"] = {}
            extraction_log["psnh_subscription_options"][
                "gpt4o_verification_error"
            ] = f"Exception: {str(e)}"

        return extraction_log

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def verify_comed_required_fields_with_gpt4o(
        self, pdf_path: str, extraction_log: Dict
    ) -> Dict:
        """
        GPT-4o verification for COMED required fields.
        COMED LOAs have flexible formats, so we use GPT-4o vision to extract all required fields.

        CRITICAL: This function uses aggressive retry logic to guarantee success.
        It will retry up to 50 times with exponential backoff rather than falling back to code-level validation.

        Args:
            pdf_path (str): Path to the PDF file.
            extraction_log (Dict): The current extraction log from Azure Layout.

        Returns:
            Dict: Updated extraction log with verified COMED field data.
        """
        self.logger.info("Performing GPT-4o verification for COMED required fields...")

        image_data = self.ocr_integration.extract_pdf_image(pdf_path, page_num=0)
        if not image_data:
            self.logger.error(
                "Failed to extract image from PDF for COMED field verification."
            )
            return extraction_log

        base64_image = self.ocr_integration.encode_image_to_base64(image_data)

        comed_fields_prompt = """CRITICAL: Extract ALL required fields from this COMED (Commonwealth Edison - Illinois) LOA document.

        **IMPORTANT**: COMED LOAs do NOT have a fixed format - different structures are acceptable.
        Search the ENTIRE document for these required fields:

        ========================================
        REQUIRED FIELDS TO EXTRACT:
        ========================================

        1. CUSTOMER NAME:
           - Labels: "Customer Name", "Company Name", "Business Name", "Account Holder"
           - Extract the actual customer/company name value

        2. CUSTOMER ADDRESS:
           - Labels: "Customer Address", "Address", "Service Location", "Street Address"
           - Extract the full address (street, city, state, ZIP)

        3. AUTHORIZED PERSON:
           - Labels: "Authorized Person", "Authorized Representative", "Printed Name", "Name"
           - Extract the person's name who is authorized to sign

        4. AUTHORIZED PERSON TITLE:
           - Labels: "Title", "Position", "Authorized Person Title"
           - Extract the job title/position

        5. CUSTOMER SIGNATURE:
           - Look for signature field in customer section
           - Determine if signature is PRESENT (any marks/text) or MISSING (blank)

        6. SIGNATURE DATE:
           - Labels: "Date", "Signature Date", "Date Signed"
           - Extract date in MM/DD/YYYY format

        7. ACCOUNT NUMBERS:
           - Labels: "Account Number(s)", "Acct", "Account #"
           - **STRICT VALIDATION**: Account numbers MUST be at least 8 digits long
           - Look for 8-20 digit numbers (e.g., 12345678, 1234567890123456)
           - **CRITICAL**: "Number of Accounts: X" is NOT an account number - it's just a count
           - **REJECT these patterns**:
             * "Number of Accounts: 2" → NOT a valid account number
             * "2 accounts" → NOT a valid account number
             * Single digit or short numbers (< 8 digits) → NOT valid
           - **ACCEPT these patterns**:
             * "Account Number: 12345678" → Valid (8+ digits)
             * "1234567890123456" → Valid (8+ digits)
             * "See attached" or "Attached" → Valid attachment indicator
           - Extract up to 3 ACTUAL account numbers (8+ digits each) as examples
           - If you only see "Number of Accounts: X", return empty array and mark has_attachment_indicator=false

        8. INTERVAL AUTHORIZATION:
           - Look ANYWHERE in document for these keywords:
           - "interval data", "interval usage", "15-minute", "30-minute", "hourly"
           - "usage data", "meter data", "monthly billing", "billing data"
           - Mark as true if ANY interval/data keywords found

        9. SUPPLIER (CONSTELLATION) INFORMATION:
           - Look for mentions of "Constellation", "CNE", "CRES provider"
           - OR Constellation email addresses (@constellation.com)
           - Mark as true if found

        10. ILLINOIS AUTHORIZATION LANGUAGE:
           - **CRITICAL**: Check for "Usage Data Type" radio button section FIRST
           - **STEP 1: Look for "Usage Data Type" section with radio buttons**
             * If section exists with options "Summary" and "Interval"
             * Check which radio button is SELECTED
             * **INTERVAL SELECTED** → Mark illinois_authorization_found=TRUE
             * **SUMMARY SELECTED** or **NEITHER SELECTED** → Mark illinois_authorization_found=FALSE, interval_data_mentioned=FALSE
             * Look for selection marks: filled circles ●, checkmarks in circles, or text indicators near "Interval"

           - **STEP 2: If NO "Usage Data Type" section exists, check authorization text**
             * Authorization must SPECIFICALLY mention "interval" data access
             * Type 1: Authorization statement + "INTERVAL" + ("data" OR "usage") within same clause
               - **VALID EXAMPLES**: "authorize interval data", "access to interval usage", "permission to release interval information"
               - **NOT VALID - Just Descriptions**: These are just descriptions of interval data, NOT authorization:
                 * "Interval Data – Half-hour demand data for non-residential accounts having recording-type meters. A $1.18 fee per meter on the account will be charged for all interval data requests."
                 * "Please accept this letter as a formal request and authorization for to release energy usage data, including kWh, kVA or kW, in both summary and interval data at the following loca"
                 * ANY text that just DESCRIBES what interval data is or MENTIONS it without authorizing access
               - **NOT VALID**: "historical usage" without "interval" mention
               - **NOT VALID**: Title mentions "interval" but authorization text doesn't
               - **REQUIREMENT**: Must have BOTH authorization keywords (authorize/permission/access/release) AND "interval" in close proximity
             * Type 2: "Constellation" or "CRES" + "EUI" + time granularity ("15-minute", "30-minute", "hourly")
             * **PROXIMITY RULE**: Authorization keywords and "interval" must be in SAME SENTENCE/CLAUSE (within ~200 characters)
             * **TITLE EXCLUSION**: If "interval" only appears in document TITLE but NOT in authorization text itself, mark as FALSE
             * **CRITICAL**: Merely mentioning "interval data" is NOT enough - must explicitly authorize/grant permission/access

           - **CRITICAL**: If "Usage Data Type" section exists but "Interval" is NOT selected, ALWAYS mark as FALSE regardless of any text mentions

        11. AGENT AUTHORIZATION CHECKBOX (CONDITIONAL):
           - Look for text: "By checking this box the Authorized Person indicates that s/he is an agent"
           - IF this text is present, check if the checkbox next to it is MARKED
           - IF this text is NOT present, mark agent_auth_section_found=false

        **EXTRACTION RULES:**
        - Extract ACTUAL VALUES, not field labels
        - If field is blank/empty, return null
        - Search ENTIRE document, not just specific locations
        - Be flexible with field label variations

        Return JSON with this structure:
        {
          "comed_fields_extraction": {
            "customer_name": "extracted value or null",
            "customer_address": "extracted value or null",
            "authorized_person": "extracted value or null",
            "authorized_person_title": "extracted value or null",
            "signature_present": true/false,
            "signature_date": "MM/DD/YYYY or null",
            "account_numbers": ["number1", "number2", "number3"] or [],
            "has_attachment_indicator": true/false,
            "interval_authorization_found": true/false,
            "supplier_info_found": true/false,
            "illinois_authorization_found": true/false,
            "authorization_type": "type1" or "type2" or "none",
            "interval_data_mentioned": true/false,
            "agent_auth_section_found": true/false,
            "agent_checkbox_marked": true/false (only if section found),
            "confidence": 0-100,
            "reasoning": "Detailed explanation of findings for each field"
          }
        }"""

        user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
        {comed_fields_prompt}

        [ATTACHED_IMAGE]
        data:image/png;base64,{base64_image}"""

        system_prompt = "You are an expert document analyzer specializing in COMED (Commonwealth Edison - Illinois) LOA forms. Extract all required fields accurately, handling flexible document formats."

        try:
            start_time = datetime.now()

            gpt_response_data = self.openai_4o_service.process_with_prompts(
                system_prompt, user_prompt, max_token=2000, raw_response=True
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            if gpt_response_data and gpt_response_data[0]["ai_result"]:
                analysis_text = gpt_response_data[0]["ai_result"][0]["result"]
            else:
                self.logger.error(
                    "No response from OpenAI service for COMED field verification"
                )
                if "comed_validation" not in extraction_log:
                    extraction_log["comed_validation"] = {}
                extraction_log["comed_validation"][
                    "gpt4o_verification_error"
                ] = "No response from OpenAI service"
                return extraction_log

            json_match = re.search(r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(1))
            else:
                analysis_json = json.loads(analysis_text)

            self.logger.info(
                f"GPT-4o COMED field verification completed in {processing_time:.2f} seconds"
            )

            comed_data = analysis_json.get("comed_fields_extraction")
            if comed_data:
                self.logger.info("GPT-4o COMED Field Extraction Results:")
                self.logger.info(
                    f"  - Customer Name: {comed_data.get('customer_name', 'Not found')}"
                )
                self.logger.info(
                    f"  - Authorized Person: {comed_data.get('authorized_person', 'Not found')}"
                )
                self.logger.info(
                    f"  - Account Numbers: {len(comed_data.get('account_numbers', []))} found"
                )
                self.logger.info(
                    f"  - Illinois Auth Found: {comed_data.get('illinois_authorization_found', False)}"
                )
                self.logger.info(
                    f"  - Interval Data Mentioned: {comed_data.get('interval_data_mentioned', False)}"
                )

                # Update or initialize comed_validation in extraction_log
                if "comed_validation" not in extraction_log:
                    extraction_log["comed_validation"] = {}

                # Map GPT-4o extracted fields to comed_validation structure
                # CRITICAL FIX: Exclude "Constellation" as it's our company (supplier), not the customer
                extracted_customer_name = comed_data.get("customer_name")

                # Validate that extracted name is NOT "Constellation" or our company variations
                if extracted_customer_name:
                    name_lower = extracted_customer_name.lower()
                    # Exclude if it's our company name
                    if "constellation" not in name_lower:
                        extraction_log["comed_validation"]["customer_name_found"] = True
                        extraction_log["comed_validation"][
                            "customer_name"
                        ] = extracted_customer_name
                    else:
                        # This is our company name, not the customer - treat as not found
                        extraction_log["comed_validation"][
                            "customer_name_found"
                        ] = False
                        extraction_log["comed_validation"]["customer_name"] = None
                        extraction_log["comed_validation"][
                            "customer_name_rejected_reason"
                        ] = "GPT-4o extracted 'Constellation' (our company, not customer)"
                        self.logger.warning(
                            "GPT-4o extracted 'Constellation' as customer name - this is our supplier, not the customer. Treating as not found."
                        )
                else:
                    extraction_log["comed_validation"]["customer_name_found"] = False
                    extraction_log["comed_validation"]["customer_name"] = None
                extraction_log["comed_validation"]["customer_address_found"] = bool(
                    comed_data.get("customer_address")
                )
                extraction_log["comed_validation"]["customer_address"] = comed_data.get(
                    "customer_address"
                )
                extraction_log["comed_validation"]["authorized_person_found"] = bool(
                    comed_data.get("authorized_person")
                )
                extraction_log["comed_validation"]["authorized_person"] = (
                    comed_data.get("authorized_person")
                )
                extraction_log["comed_validation"]["authorized_person_title_found"] = (
                    bool(comed_data.get("authorized_person_title"))
                )
                extraction_log["comed_validation"]["authorized_person_title"] = (
                    comed_data.get("authorized_person_title")
                )

                # CRITICAL: Only update signature_found if it wasn't already set by dedicated GPT-4o vision signature verification
                # The dedicated extract_signatures_with_gpt4o() is more accurate for signature detection
                if (
                    extraction_log["comed_validation"].get(
                        "signature_verification_method"
                    )
                    != "gpt4o_vision"
                ):
                    extraction_log["comed_validation"]["signature_found"] = (
                        comed_data.get("signature_present", False)
                    )
                else:
                    # Signature was already verified by dedicated vision call - don't overwrite it
                    self.logger.info(
                        "Signature status already set by dedicated GPT-4o vision verification - preserving that result"
                    )

                extraction_log["comed_validation"]["signature_date_found"] = bool(
                    comed_data.get("signature_date")
                )
                extraction_log["comed_validation"]["signature_date"] = comed_data.get(
                    "signature_date"
                )

                account_nums = comed_data.get("account_numbers", [])
                has_attachment = comed_data.get("has_attachment_indicator", False)
                extraction_log["comed_validation"]["account_numbers_found"] = bool(
                    account_nums or has_attachment
                )
                extraction_log["comed_validation"]["account_numbers"] = account_nums
                extraction_log["comed_validation"]["account_count"] = len(account_nums)
                extraction_log["comed_validation"][
                    "has_attachment_indicator"
                ] = has_attachment

                extraction_log["comed_validation"]["interval_authorization_found"] = (
                    comed_data.get("interval_authorization_found", False)
                )
                extraction_log["comed_validation"]["supplier_info_found"] = (
                    comed_data.get("supplier_info_found", False)
                )
                extraction_log["comed_validation"]["illinois_authorization_found"] = (
                    comed_data.get("illinois_authorization_found", False)
                )
                extraction_log["comed_validation"]["authorization_type"] = (
                    comed_data.get("authorization_type", "none")
                )
                extraction_log["comed_validation"]["interval_data_in_auth"] = (
                    comed_data.get("interval_data_mentioned", False)
                )
                extraction_log["comed_validation"]["agent_auth_section_found"] = (
                    comed_data.get("agent_auth_section_found", False)
                )
                extraction_log["comed_validation"]["agent_checkbox_marked"] = (
                    comed_data.get("agent_checkbox_marked", False)
                )

                extraction_log["comed_validation"]["gpt4o_verified"] = True
                extraction_log["comed_validation"][
                    "gpt4o_verification_details"
                ] = comed_data

                self.logger.info(
                    "Updated COMED validation with GPT-4o extracted fields."
                )

                return {"success": True, "extraction_log": extraction_log}
            else:
                self.logger.warning(
                    "GPT-4o did not return the expected 'comed_fields_extraction' key."
                )
                if "comed_validation" not in extraction_log:
                    extraction_log["comed_validation"] = {}
                extraction_log["comed_validation"][
                    "gpt4o_verification_error"
                ] = "Invalid JSON structure from GPT-4o"
                return {"success": False, "extraction_log": extraction_log}

        except Exception as e:
            error_msg = f"GPT-4o COMED field verification failed: {str(e)}"
            self.logger.error(error_msg)
            if "comed_validation" not in extraction_log:
                extraction_log["comed_validation"] = {}
            extraction_log["comed_validation"][
                "gpt4o_verification_error"
            ] = f"Exception: {str(e)}"
            return {"success": False, "extraction_log": extraction_log, "error": str(e)}

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def verify_aep_interval_granularity_with_gpt4o(
        self, pdf_path: str, extraction_log: Dict
    ) -> Dict:
        """
        GPT-4o verification for AEP interval data granularity text.
        AEP LOAs (like FirstEnergy) often have text like "IDR, summary, interval" that OCR misses
        because it's positioned in unusual locations on the form.

        Args:
            pdf_path (str): Path to the PDF file.
            extraction_log (Dict): The current extraction log from Azure Layout.

        Returns:
            Dict: Updated extraction_log with verified interval granularity detection.
        """
        self.logger.info(
            "Performing GPT-4o verification for AEP interval data granularity..."
        )

        image_data = self.ocr_integration.extract_pdf_image(pdf_path, page_num=0)
        if not image_data:
            self.logger.error(
                "Failed to extract image from PDF for AEP interval granularity verification."
            )
            return extraction_log

        base64_image = self.ocr_integration.encode_image_to_base64(image_data)

        aep_prompt = """CRITICAL: Examine this AEP LOA document for interval data granularity specifications.

        **TASK**: Search the ENTIRE document for data granularity text - it appears DIRECTLY IN THE SENTENCE, not in a separate blank space.

        **CRITICAL: AEP FORM STRUCTURE IS DIFFERENT!**

        On AEP forms, the granularity text is typed/written DIRECTLY INTO THE SENTENCE, like:
        - "The above named General Service customer authorizes the release of up to 12 months of **Summary, IDR, Cap/Tran** kwh data"
        - The text "Summary, IDR, Cap/Tran" appears RIGHT BEFORE "kwh data" - NOT in a separate blank area!

        **WHAT TO LOOK FOR**:

        Find the authorization sentence that contains "kwh data" and check what text appears BEFORE it:
        - Look for: "...of ______ kwh data" OR "...of [SOME TEXT] kwh data"
        - The granularity text is the words between "of" and "kwh data"
        - Examples of FILLED granularity:
          * "...of Summary, IDR, Cap/Tran kwh data" → Extract: "Summary, IDR, Cap/Tran"
          * "...of interval kwh data" → Extract: "interval"
          * "...of Summary kwh data" → Extract: "Summary"

        **ACCEPT as VALID granularity text**:
        - "Summary", "IDR", "Cap/Tran", "Train/cap"
        - "Summary, IDR, Cap/Tran" (comma-separated combinations)
        - "interval", "hourly", "15-minute", "30-minute"
        - "IU", "HU", "IU/HU"
        - Any combination of these terms

        **IGNORE** (these are instructions, NOT the actual data):
        - Text after "e.g.": "e.g., Summary, IDR, Cap/Tran, Hourly..."
        - Text in parentheses with "Please fill in..."
        - ANY text that is part of the instruction line

        **CRITICAL DISTINCTION**:

        ✅ VALID (this is the actual filled-in granularity):
        "The above named General Service customer authorizes the release of up to 12 months of **Summary, IDR, Cap/Tran** kwh data"
        → Extract: "Summary, IDR, Cap/Tran" (text_found: TRUE)

        ❌ NOT the granularity (this is instruction text):
        "(Please fill in the blank with your request, e.g., Summary, IDR, Cap/Tran, Hourly, 30-minute, 15-minute, etc.)"
        → IGNORE this - it's just instructions

        **SEARCH LOCATIONS** (in order):

        1. **PRIMARY**: The authorization sentence with "kwh data"
           - Find text like "...of [GRANULARITY] kwh data"
           - The granularity appears BETWEEN "of" and "kwh data"

        2. **SECONDARY**: Near initial boxes
           - Look for handwritten text near "Interval Historical Energy Usage Data Release"

        3. **TERTIARY**: Anywhere else in document
           - Scan for any handwritten granularity specifications

        **EXTRACTION RULES**:
        - If you find ANY granularity text (Summary, IDR, etc.) in the authorization sentence → text_found: TRUE
        - Only mark text_found: FALSE if the sentence shows "...of ______ kwh data" with ONLY underscores/blanks
        - Report the exact text you found

        Return your findings in JSON format:
        {
          "aep_interval_granularity": {
            "text_found": true/false,
            "extracted_text": "exact text found (e.g., 'Summary, IDR, Cap/Tran') or null",
            "location_description": "where on the document this text appears",
            "confidence": 0-100,
            "reasoning": "Detailed explanation: What text did you find in the authorization sentence before 'kwh data'?"
          }
        }"""

        user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
        {aep_prompt}

        [ATTACHED_IMAGE]
        data:image/png;base64,{base64_image}"""

        system_prompt = "You are an expert document analyzer specializing in AEP LOA forms. Your task is to find interval data granularity specifications that may appear in unusual positions on the form."

        try:
            start_time = datetime.now()

            gpt_response_data = self.openai_4o_service.process_with_prompts(
                system_prompt, user_prompt, max_token=1000, raw_response=True
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            if gpt_response_data and gpt_response_data[0]["ai_result"]:
                analysis_text = gpt_response_data[0]["ai_result"][0]["result"]
            else:
                self.logger.error(
                    "No response from OpenAI service for AEP interval granularity verification"
                )
                if "aep_interval_granularity" not in extraction_log:
                    extraction_log["aep_interval_granularity"] = {}
                extraction_log["aep_interval_granularity"][
                    "gpt4o_verification_error"
                ] = "No response from OpenAI service"
                return extraction_log

            json_match = re.search(r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(1))
            else:
                analysis_json = json.loads(analysis_text)

            self.logger.info(
                f"GPT-4o AEP interval granularity verification completed in {processing_time:.2f} seconds"
            )

            verification_data = analysis_json.get("aep_interval_granularity")
            if verification_data:
                text_found = verification_data.get("text_found", False)
                extracted_text = verification_data.get("extracted_text", None)
                location = verification_data.get("location_description", "Unknown")
                confidence = verification_data.get("confidence", 0)
                reasoning = verification_data.get("reasoning", "No reasoning provided")

                self.logger.info("GPT-4o AEP Interval Granularity Findings:")
                self.logger.info(f"      - Text Found: {text_found}")
                self.logger.info(f"      - Extracted Text: '{extracted_text}'")
                self.logger.info(f"      - Location: {location}")
                self.logger.info(f"      - Confidence: {confidence}%")
                self.logger.info(f"      - Reasoning: {reasoning}")

                # Initialize aep_interval_granularity if not present
                if "aep_interval_granularity" not in extraction_log:
                    extraction_log["aep_interval_granularity"] = {}

                # Store the verification results
                extraction_log["aep_interval_granularity"]["text_found"] = text_found
                extraction_log["aep_interval_granularity"][
                    "extracted_text"
                ] = extracted_text
                extraction_log["aep_interval_granularity"]["location"] = location
                extraction_log["aep_interval_granularity"]["confidence"] = confidence
                extraction_log["aep_interval_granularity"]["gpt4o_verified"] = True
                extraction_log["aep_interval_granularity"][
                    "gpt4o_verification_details"
                ] = verification_data

                self.logger.info("Stored AEP interval granularity detection results.")
            else:
                self.logger.warning(
                    "GPT-4o did not return the expected 'aep_interval_granularity' key."
                )
                if "aep_interval_granularity" not in extraction_log:
                    extraction_log["aep_interval_granularity"] = {}
                extraction_log["aep_interval_granularity"][
                    "gpt4o_verification_error"
                ] = "Invalid JSON structure from GPT-4o"

        except Exception as e:
            error_msg = f"GPT-4o AEP interval granularity verification failed: {str(e)}"
            self.logger.error(error_msg)
            if "aep_interval_granularity" not in extraction_log:
                extraction_log["aep_interval_granularity"] = {}
            extraction_log["aep_interval_granularity"][
                "gpt4o_verification_error"
            ] = f"Exception: {str(e)}"

        return extraction_log

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def verify_firstenergy_interval_granularity_with_gpt4o(
        self, pdf_path: str, extraction_log: Dict
    ) -> Dict:
        """
        GPT-4o verification for FirstEnergy interval data granularity text.
        FirstEnergy LOAs often have text like "IDR, Train/cap, summary, interval" that OCR misses
        because it's positioned in unusual locations on the form.

        Args:
            pdf_path (str): Path to the PDF file.
            extraction_log (Dict): The current extraction log from Azure Layout.

        Returns:
            Dict: Updated extraction log with verified interval granularity detection.
        """
        self.logger.info(
            "Performing GPT-4o verification for FirstEnergy interval data granularity..."
        )

        image_data = self.ocr_integration.extract_pdf_image(pdf_path, page_num=0)
        if not image_data:
            self.logger.error(
                "Failed to extract image from PDF for FirstEnergy interval granularity verification."
            )
            return extraction_log

        base64_image = self.ocr_integration.encode_image_to_base64(image_data)

        firstenergy_prompt = """CRITICAL: Examine this FirstEnergy LOA document for interval data granularity specifications.

        **TASK**: Search the ENTIRE document for data granularity text - it may appear in MULTIPLE locations.

        **SEARCH ALL THESE LOCATIONS** (in order of priority):

        **LOCATION 1: Primary Blank Space**
        - Sentence: "The above named customer authorizes the release of up to 12/24 months of ______ kwh data"
        - Check if blank space is filled with handwritten/typed text

        **LOCATION 2: Initial Boxes Section** (CRITICAL - often missed!)
        - Look for TWO initial boxes near "Account/SDI Number Release" and "Interval Historical Energy Usage Data Release"
        - Check for text NEXT TO, NEAR, or WITHIN the initial box areas
        - Look for handwritten text like "IU", "HU", "IU/HU", "IDR" near checkboxes/initials

        **LOCATION 3: Authorization Checkboxes**
        - Near text about "Interval Historical Energy Usage Data Release"
        - Look for filled-in text, selections, or written specifications

        **LOCATION 4: Anywhere Else**
        - Scan the ENTIRE document for any handwritten or typed granularity specifications
        - Check margins, between lines, near signature areas

        **IGNORE** (instruction/example text):
        - Text in parentheses: "(Hourly, 30-minute, 15-minute)"
        - Text after "e.g.": "e.g. 'Summary, IDR, Cap/Tran, Hourly'"
        - Pre-printed form instructions

        **ACCEPT** (user-filled text anywhere in document):

        **TIME-BASED INTERVALS**:
        - "15-minute", "30-minute", "60-minute", "hourly"

        **UTILITY INDUSTRY TERMS**:
        - "IDR", "Cap/Tran", "Train/cap"
        - "Summary", "Interval"
        - "IU", "HU", "IU/HU", "HU/IU" ← COMMON on FirstEnergy forms!
        - "ALL"

        **CRITICAL**: IU/HU is often written:
        - In the initial box areas (near checkmarks or X marks)
        - Next to "Interval Historical Energy Usage Data Release" text
        - As small handwritten text that may be easy to miss

        **EXTRACTION RULES**:
        - If you find granularity text ANYWHERE → text_found: TRUE
        - Only text_found: FALSE if NO granularity text exists in entire document
        - Report WHERE you found it (which location)

        Return your findings in JSON format:
        {
          "firstenergy_interval_granularity": {
            "text_found": true/false,
            "extracted_text": "exact text found or null",
            "location_description": "where on the document this text appears (blank space, initial box area, checkbox area, etc.)",
            "confidence": 0-100,
            "reasoning": "Detailed explanation: Which locations did you check? What did you find in each?"
          }
        }"""

        user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
        {firstenergy_prompt}

        [ATTACHED_IMAGE]
        data:image/png;base64,{base64_image}"""

        system_prompt = "You are an expert document analyzer specializing in FirstEnergy LOA forms. Your task is to find interval data granularity specifications that may appear in unusual positions on the form."

        try:
            start_time = datetime.now()

            gpt_response_data = self.openai_4o_service.process_with_prompts(
                system_prompt, user_prompt, max_token=1000, raw_response=True
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            if gpt_response_data and gpt_response_data[0]["ai_result"]:
                analysis_text = gpt_response_data[0]["ai_result"][0]["result"]
            else:
                self.logger.error(
                    "No response from OpenAI service for FirstEnergy interval granularity verification"
                )
                if "firstenergy_interval_granularity" not in extraction_log:
                    extraction_log["firstenergy_interval_granularity"] = {}
                extraction_log["firstenergy_interval_granularity"][
                    "gpt4o_verification_error"
                ] = "No response from OpenAI service"
                return extraction_log

            json_match = re.search(r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(1))
            else:
                analysis_json = json.loads(analysis_text)

            self.logger.info(
                f"GPT-4o FirstEnergy interval granularity verification completed in {processing_time:.2f} seconds"
            )

            verification_data = analysis_json.get("firstenergy_interval_granularity")
            if verification_data:
                text_found = verification_data.get("text_found", False)
                extracted_text = verification_data.get("extracted_text", None)
                location = verification_data.get("location_description", "Unknown")
                confidence = verification_data.get("confidence", 0)
                reasoning = verification_data.get("reasoning", "No reasoning provided")

                self.logger.info("GPT-4o FirstEnergy Interval Granularity Findings:")
                self.logger.info(f"      - Text Found: {text_found}")
                self.logger.info(f"      - Extracted Text: '{extracted_text}'")
                self.logger.info(f"      - Location: {location}")
                self.logger.info(f"      - Confidence: {confidence}%")
                self.logger.info(f"      - Reasoning: {reasoning}")

                # Initialize firstenergy_interval_granularity if not present
                if "firstenergy_interval_granularity" not in extraction_log:
                    extraction_log["firstenergy_interval_granularity"] = {}

                # Store the verification results
                extraction_log["firstenergy_interval_granularity"][
                    "text_found"
                ] = text_found
                extraction_log["firstenergy_interval_granularity"][
                    "extracted_text"
                ] = extracted_text
                extraction_log["firstenergy_interval_granularity"][
                    "location"
                ] = location
                extraction_log["firstenergy_interval_granularity"][
                    "confidence"
                ] = confidence
                extraction_log["firstenergy_interval_granularity"][
                    "gpt4o_verified"
                ] = True
                extraction_log["firstenergy_interval_granularity"][
                    "gpt4o_verification_details"
                ] = verification_data

                self.logger.info(
                    "Stored FirstEnergy interval granularity detection results."
                )
            else:
                self.logger.warning(
                    "GPT-4o did not return the expected 'firstenergy_interval_granularity' key."
                )
                if "firstenergy_interval_granularity" not in extraction_log:
                    extraction_log["firstenergy_interval_granularity"] = {}
                extraction_log["firstenergy_interval_granularity"][
                    "gpt4o_verification_error"
                ] = "Invalid JSON structure from GPT-4o"

        except Exception as e:
            error_msg = (
                f"GPT-4o FirstEnergy interval granularity verification failed: {str(e)}"
            )
            self.logger.error(error_msg)
            if "firstenergy_interval_granularity" not in extraction_log:
                extraction_log["firstenergy_interval_granularity"] = {}
            extraction_log["firstenergy_interval_granularity"][
                "gpt4o_verification_error"
            ] = f"Exception: {str(e)}"

        return extraction_log

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def verify_aep_comprehensive_with_gpt4o(
        self, pdf_path: str, extraction_log: Dict
    ) -> Dict:
        """
        GPT-4o comprehensive verification for AEP (CSPC, OPC) LOAs.
        Extracts ALL required fields including customer, CRES provider, Ohio phrase sections.

        CRITICAL: This function uses aggressive retry logic to guarantee success.
        It will retry up to 50 times with exponential backoff.

        Args:
            pdf_path (str): Path to the PDF file.
            extraction_log (Dict): The current extraction log from Azure Layout.

        Returns:
            Dict: Updated extraction log with ALL verified AEP fields.
        """
        self.logger.info("Performing GPT-4o comprehensive verification for AEP LOA...")

        image_data = self.ocr_integration.extract_pdf_image(pdf_path, page_num=0)
        if not image_data:
            self.logger.error(
                "Failed to extract image from PDF for AEP comprehensive verification."
            )
            return {"success": False, "extraction_log": extraction_log}

        base64_image = self.ocr_integration.encode_image_to_base64(image_data)

        aep_comprehensive_prompt = """CRITICAL: Extract ALL required fields from this AEP LOA document (CSPC/OPC - Ohio utilities).

        ========================================
        SECTION 1: CUSTOMER INFORMATION (Top of Document)
        ========================================

        1. CUSTOMER NAME:
           - Labels: "CUSTOMER NAME", "Customer Name"
           - Extract the actual customer/company name value
           - If field is blank/empty, return null

        2. PHONE NUMBER:
           - Labels: "Phone Number"
           - Format: (XXX) XXX-XXXX or variations
           - Extract the actual phone number

        3. CUSTOMER ADDRESS:
           - Labels: "Customer Address"
           - Extract full address (street, city, state, ZIP)

        4. AUTHORIZED PERSON/TITLE:
           - Labels: "Authorized Person/Title"
           - Extract the person's name and/or title

        5. ACCOUNT/SDI NUMBERS: **CRITICAL - AEP ACCEPTS TWO FORMATS**

           **AEP ACCEPTS EITHER FORMAT**:

           **FORMAT 1: Standalone 17 digits** (most common)
           - Example: "00140060748972843" (17 digits)
           - OCR Tolerance: Accept 16-18 digits
           - Valid if: 16 ≤ digit_count ≤ 18

           **FORMAT 2: 11/17 format with slash**
           - Example: "12345678901/12345678901234567"
           - Part 1 tolerance: 10-12 digits
           - Part 2 tolerance: 16-18 digits
           - Extract as SINGLE account (keep slash)
           - DO NOT split by slash

           **EXTRACTION & VALIDATION RULES**:
           - Extract account numbers (with or without slash)
           - Remove spaces/dashes EXCEPT slash
           - Count digits in each number
           - Check if it matches EITHER valid format:
             * 16-18 digits (no slash) = VALID ✓
             * 10-12 / 16-18 digits (with slash) = VALID ✓
             * Any other length = INVALID ✗

           **ATTACHMENT DETECTION**:
           - "See attached" / "Attached" = TRUE (actual attachment confirmation)
           - "Please attach" / "For multiple...attach" = FALSE (just instructions)

           **EMPTY FIELD CHECK**:
           - If no numbers AND no attachment confirmation → account_field_empty: true

        ========================================
        SECTION 2: CRES PROVIDER SECTION (Middle of Document)
        ========================================

        6-9. CRES FIELDS (Name, Address, Phone, Email):
           - Extract from CRES Provider section
           - Return null if field is blank/empty

        ========================================
        SECTION 3: OHIO AUTHORIZATION STATEMENT (Bottom)
        ========================================

        10. OHIO STATEMENT SIGNATURE:
           - Extract actual signature text (not just presence)
           - Look for broker indicators: "on behalf of", "utilities group"

        11. OHIO STATEMENT DATE:
           - Extract date in MM/DD/YYYY format

        12. UTILITY NAME IN OHIO PHRASE:
           - **VALID AEP UTILITIES**: AEP, AEP Ohio, CSPC, OPC, Columbus Southern Power, Ohio Power
           - Extract utility name from "allow [UTILITY] to release" phrase
           - Validate it's an AEP utility

        ========================================
        SECTION 4: INITIAL BOXES (TWO BOXES REQUIRED) - CRITICAL ACCURACY RULES
        ========================================

        **ULTRA-CRITICAL: DISTINGUISH BETWEEN LETTER INITIALS AND X MARKS**

        **STEP 1: SEARCH AREA EXPANSION**
        - Search for initials IN the box AND NEAR the box (floating initials are common!)
        - Initials may be:
          * Perfectly inside the box
          * Slightly above/below the box line
          * Floating near the box boundary
          * Written over or across box edges
        - **Accept initials ANYWHERE within the general box area** - don't require perfect positioning

        **STEP 2: DISTINGUISH LETTER INITIALS FROM X MARKS**

        **LETTER INITIALS** (VALID - mark as filled_with_initials=true):
        - **Two or more letters together**: PS, JD, AB, WA, EO, MK, TC, etc.
        - **Letters with periods/dots**: P.S., J.D., A.B., W.A. (ALWAYS initials, NEVER X)
        - **Letters with slashes**: P/S, J/D (some people write this way)
        - **Cursive/connected letters**: Two letters written in flowing script
        - **Single letters**: J, P, D, M (valid if clearly representing a name)
        - **Stacked letters**: One letter above another (some people write this way)

        **CRITICAL RULE**: If you see ANY recognizable letters (not just X), treat as VALID initials!

        **X MARK** (INVALID - mark as has_x_mark=true):
        - **ONLY flag as X if**: It's a single, standalone "X" character used as a checkmark
        - **NOT an X if**: Two letters that might visually resemble X when combined (like P and S crossing)
        - **NOT an X if**: Any text that includes letters other than X

        **COMMON INITIAL PATTERNS TO RECOGNIZE**:
        - First + Last name initials: PS (Peter Smith), JD (John Doe), AB (Alice Brown)
        - With periods: P.S., J.D., A.B., W.A., M.K.
        - Single letter initials: J, P, D, M, A
        - Stylized/cursive versions of the above
        - Underlined initials
        - Circled initials

        **STEP 3: EXAMINE EACH BOX**

        For EACH of the TWO boxes:
        1. Look IN and AROUND the box area
        2. Check for ANY letter-like marks (not just X)
        3. If you see letters → filled_with_initials=true
        4. If you see ONLY a single X → has_x_mark=true
        5. If you see NOTHING → box is empty

        **ULTRA-IMPORTANT: INITIALS DON'T NEED TO BE INSIDE BOXES**

        The boxes are just a GUIDE for where to put initials. What matters is:
        - Are there INITIALS present in the general area?
        - The initials can be:
          * Inside the box (perfect)
          * Outside the box but near it (still valid!)
          * Floating above/below the box line (still valid!)
          * Written across box boundaries (still valid!)
          * Anywhere in the initial section area (still valid!)

        **AS LONG AS INITIALS EXIST, MARK AS VALID** regardless of exact positioning!

        **FINAL VALIDATION**:
        - filled_box_count = boxes/areas with LETTER INITIALS anywhere nearby (any letters)
        - empty_box_count = boxes/areas with NO marks anywhere nearby
        - x_mark_count = boxes/areas with ONLY a standalone X mark (rare!)

        **REMINDER**: When in doubt, if you see ANY letters in or near the box area, treat as valid initials!

        ========================================
        SECTION 5: FORM TYPE VALIDATION - AEP FORM PHRASE
        ========================================

        15. FORM IDENTIFYING MARKERS - **CRITICAL: TEXT SPANS MULTIPLE LINES**

           **STEP 1: Look at TOP of document for these KEY COMPONENTS**:

           Component 1: "Ohio Customer Letter of Authorization"
           Component 2: "For Release of Customer's Electric Utility Account" OR "Account Number" OR "SDI"
           Component 3: "Historical Interval Data" OR "Interval Data"
           Component 4: "Non-Residential" (with OR without "General Service/" prefix)

           **STEP 2: Check if you see ALL FOUR components near each other**
           - Text typically spans 3-5 lines at top of form
           - Line breaks between components are NORMAL
           - Order may vary slightly
           - Spacing variations OK

           **ACCEPTABLE VARIATIONS** (all are VALID):
           ✓ "Ohio Customer Letter of Authorization\nFor Release of Customer's Electric Utility Account\nNumber/SDI and/or General Service/Non-Residential Historical Interval Data"
           ✓ "Ohio Customer Letter of Authorization\nFor Release of Customer's Electric Utility Account\nNumber/SDI and/or Non-Residential Historical Interval Data"
           ✓ Same text but with different line break positions
           ✓ Minor spacing differences (e.g., "Number / SDI" vs "Number/SDI")

           **HOW TO CHECK**:
           - Look at the header/title area (first 10-15 lines)
           - Find "Ohio Customer Letter of Authorization" (Component 1)
           - Within the SAME general area (next 5-10 lines), look for the other components
           - If you find ALL key components in the header area → form_type_valid=TRUE
           - If ANY component is missing → form_type_valid=FALSE

           **CRITICAL**: Do NOT require exact line-by-line matching
           - Components may be split across lines differently
           - Focus on PRESENCE of all components, not exact formatting

           **EXAMPLE**:
           Document shows:
           ```
           Ohio Customer Letter of Authorization
           For Release of Customer's Electric Utility Account
           Number/SDI and/or Non-Residential Historical Interval Data
           ```
           → form_type_valid=TRUE (all components present) ✓

        Return JSON with this structure:
        {
          "aep_comprehensive_extraction": {
            "customer_name": "value or null",
            "customer_phone": "value or null",
            "customer_address": "value or null",
            "authorized_person_title": "value or null",
            "account_numbers": ["numbers"] or [],
            "has_attachment_indicator": true/false,
            "account_field_empty": true/false,
            "account_length_valid": true/false,
            "invalid_length_accounts": [] or ["account (X digits)"],
            "cres_name": "value or null",
            "cres_address": "value or null",
            "cres_phone": "value or null",
            "cres_email": "value or null",
            "ohio_signature_present": true/false,
            "ohio_signature_text": "text or null",
            "ohio_signature_date": "MM/DD/YYYY or null",
            "ohio_phrase_utility": "AEP/CSPC/OPC or null",
            "ohio_phrase_utility_valid": true/false,
            "form_type_valid": true/false,
            "form_identifiers_found": ["identifiers found"] or [],
            "initial_boxes": {
              "box_1_filled_with_initials": true/false,
              "box_1_content": "initials or 'EMPTY' or 'X'",
              "box_2_filled_with_initials": true/false,
              "box_2_content": "initials or 'EMPTY' or 'X'",
              "filled_box_count": 0-2,
              "empty_box_count": 0-2,
              "x_mark_count": 0-2,
              "all_boxes_valid": true/false
            },
            "confidence": 0-100,
            "reasoning": "Detailed explanation"
          }
        }"""

        user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
        {aep_comprehensive_prompt}

        [ATTACHED_IMAGE]
        data:image/png;base64,{base64_image}"""

        system_prompt = "You are an expert document analyzer specializing in AEP (CSPC/OPC) LOA forms. Extract ALL required fields with high accuracy for AEP Ohio utilities."

        try:
            start_time = datetime.now()

            gpt_response_data = self.openai_4o_service.process_with_prompts(
                system_prompt, user_prompt, max_token=2500, raw_response=True
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            if gpt_response_data and gpt_response_data[0]["ai_result"]:
                analysis_text = gpt_response_data[0]["ai_result"][0]["result"]
            else:
                self.logger.error(
                    "No response from OpenAI service for AEP comprehensive verification"
                )
                if "aep_validation" not in extraction_log:
                    extraction_log["aep_validation"] = {}
                extraction_log["aep_validation"][
                    "gpt4o_verification_error"
                ] = "No response from OpenAI service"
                return {"success": False, "extraction_log": extraction_log}

            json_match = re.search(r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(1))
            else:
                analysis_json = json.loads(analysis_text)

            self.logger.info(
                f"GPT-4o AEP comprehensive verification completed in {processing_time:.2f} seconds"
            )

            aep_data = analysis_json.get("aep_comprehensive_extraction")
            if aep_data:
                self.logger.info("GPT-4o AEP Comprehensive Extraction Results:")
                self.logger.info(
                    f"  - Customer Name: {aep_data.get('customer_name', 'Not found')}"
                )
                self.logger.info(
                    f"  - Account Numbers: {len(aep_data.get('account_numbers', []))} found"
                )
                self.logger.info(
                    f"  - Account Format Valid (11/17): {aep_data.get('account_length_valid', True)}"
                )

                if "aep_validation" not in extraction_log:
                    extraction_log["aep_validation"] = {}

                # Map all extracted fields
                extraction_log["aep_validation"]["customer_name_found"] = bool(
                    aep_data.get("customer_name")
                )
                extraction_log["aep_validation"]["customer_name"] = aep_data.get(
                    "customer_name"
                )
                extraction_log["aep_validation"]["customer_phone_found"] = bool(
                    aep_data.get("customer_phone")
                )
                extraction_log["aep_validation"]["customer_phone"] = aep_data.get(
                    "customer_phone"
                )
                extraction_log["aep_validation"]["customer_address_found"] = bool(
                    aep_data.get("customer_address")
                )
                extraction_log["aep_validation"]["customer_address"] = aep_data.get(
                    "customer_address"
                )
                extraction_log["aep_validation"]["authorized_person_title_found"] = (
                    bool(aep_data.get("authorized_person_title"))
                )
                extraction_log["aep_validation"]["authorized_person_title"] = (
                    aep_data.get("authorized_person_title")
                )

                account_nums = aep_data.get("account_numbers", [])
                has_attachment = aep_data.get("has_attachment_indicator", False)
                account_field_empty = aep_data.get("account_field_empty", False)
                account_length_valid = aep_data.get("account_length_valid", True)
                invalid_length_accounts = aep_data.get("invalid_length_accounts", [])

                extraction_log["aep_validation"]["account_numbers_found"] = bool(
                    account_nums or has_attachment
                )
                extraction_log["aep_validation"]["account_numbers"] = account_nums
                extraction_log["aep_validation"]["account_count"] = len(account_nums)
                extraction_log["aep_validation"][
                    "has_attachment_indicator"
                ] = has_attachment
                extraction_log["aep_validation"][
                    "account_field_empty"
                ] = account_field_empty
                extraction_log["aep_validation"][
                    "account_length_valid"
                ] = account_length_valid
                extraction_log["aep_validation"][
                    "invalid_length_accounts"
                ] = invalid_length_accounts

                extraction_log["aep_validation"]["cres_name_found"] = bool(
                    aep_data.get("cres_name")
                )
                extraction_log["aep_validation"]["cres_name"] = aep_data.get(
                    "cres_name"
                )
                extraction_log["aep_validation"]["cres_address_found"] = bool(
                    aep_data.get("cres_address")
                )
                extraction_log["aep_validation"]["cres_address"] = aep_data.get(
                    "cres_address"
                )
                extraction_log["aep_validation"]["cres_phone_found"] = bool(
                    aep_data.get("cres_phone")
                )
                extraction_log["aep_validation"]["cres_phone"] = aep_data.get(
                    "cres_phone"
                )
                extraction_log["aep_validation"]["cres_email_found"] = bool(
                    aep_data.get("cres_email")
                )
                extraction_log["aep_validation"]["cres_email"] = aep_data.get(
                    "cres_email"
                )

                extraction_log["aep_validation"]["ohio_signature_found"] = aep_data.get(
                    "ohio_signature_present", False
                )
                extraction_log["aep_validation"]["ohio_signature_text"] = aep_data.get(
                    "ohio_signature_text"
                )
                extraction_log["aep_validation"]["ohio_signature_date"] = aep_data.get(
                    "ohio_signature_date"
                )
                extraction_log["aep_validation"]["ohio_date_found"] = bool(
                    aep_data.get("ohio_signature_date")
                )
                extraction_log["aep_validation"]["ohio_phrase_utility"] = aep_data.get(
                    "ohio_phrase_utility"
                )
                extraction_log["aep_validation"]["ohio_phrase_utility_valid"] = (
                    aep_data.get("ohio_phrase_utility_valid", False)
                )

                # CRITICAL: Store form type validation results (like FirstEnergy does)
                extraction_log["aep_validation"]["form_type_valid"] = aep_data.get(
                    "form_type_valid", False
                )
                extraction_log["aep_validation"]["form_identifiers_found"] = (
                    aep_data.get("form_identifiers_found", [])
                )

                # Extract initial boxes data
                initial_boxes = aep_data.get("initial_boxes", {})
                if initial_boxes:
                    extraction_log["aep_validation"]["initial_boxes"] = {
                        "box_1_filled_with_initials": initial_boxes.get(
                            "box_1_filled_with_initials", False
                        ),
                        "box_1_content": initial_boxes.get("box_1_content", "EMPTY"),
                        "box_2_filled_with_initials": initial_boxes.get(
                            "box_2_filled_with_initials", False
                        ),
                        "box_2_content": initial_boxes.get("box_2_content", "EMPTY"),
                        "filled_box_count": initial_boxes.get("filled_box_count", 0),
                        "empty_box_count": initial_boxes.get("empty_box_count", 0),
                        "x_mark_count": initial_boxes.get("x_mark_count", 0),
                        "all_boxes_valid": initial_boxes.get("all_boxes_valid", False),
                    }
                else:
                    extraction_log["aep_validation"]["initial_boxes"] = {
                        "filled_box_count": 0,
                        "empty_box_count": 0,
                        "x_mark_count": 0,
                        "detection_error": "Initial boxes data not returned by GPT-4o",
                    }

                extraction_log["aep_validation"]["gpt4o_verified"] = True
                extraction_log["aep_validation"][
                    "gpt4o_verification_details"
                ] = aep_data
                extraction_log["aep_validation"]["validation_method"] = "gpt4o_vision"

                self.logger.info(
                    "Updated AEP validation with GPT-4o comprehensive extraction."
                )

                return {"success": True, "extraction_log": extraction_log}
            else:
                self.logger.warning(
                    "GPT-4o did not return the expected 'aep_comprehensive_extraction' key."
                )
                if "aep_validation" not in extraction_log:
                    extraction_log["aep_validation"] = {}
                extraction_log["aep_validation"][
                    "gpt4o_verification_error"
                ] = "Invalid JSON structure from GPT-4o"
                return {"success": False, "extraction_log": extraction_log}

        except Exception as e:
            error_msg = f"GPT-4o AEP comprehensive verification failed: {str(e)}"
            self.logger.error(error_msg)
            if "aep_validation" not in extraction_log:
                extraction_log["aep_validation"] = {}
            extraction_log["aep_validation"][
                "gpt4o_verification_error"
            ] = f"Exception: {str(e)}"
            return {"success": False, "extraction_log": extraction_log, "error": str(e)}

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def verify_initial_boxes_with_gpt4o(
        self, pdf_path: str, extraction_log: Dict, udc_name: str, validation_key: str
    ) -> Dict:
        """
        GPT-4o verification for utility company initial boxes.
        Many LOAs have TWO initial boxes that must be filled with letter initials.

        FOCUSED FUNCTION: Only validates initial boxes, not other fields.

        Args:
            pdf_path (str): Path to the PDF file.
            extraction_log (Dict): The current extraction log from Azure Layout.
            udc_name (str): Display name of the utility company (e.g., "CINERGY/Duke Energy Ohio", "Dayton Power & Light")
            validation_key (str): Key for storing validation results (e.g., "cinergy_validation", "dayton_validation")

        Returns:
            Dict: Updated extraction_log with verified initial box data.
        """
        self.logger.info(
            f"Performing GPT-4o verification for {udc_name} initial boxes..."
        )

        image_data = self.ocr_integration.extract_pdf_image(pdf_path, page_num=0)
        if not image_data:
            self.logger.error(
                f"Failed to extract image from PDF for {udc_name} initial box verification."
            )
            return extraction_log

        base64_image = self.ocr_integration.encode_image_to_base64(image_data)

        initial_prompt = f"""CRITICAL: Examine this {udc_name} LOA document for initial box validation ONLY.

        ========================================
        INITIAL BOXES (TWO BOXES REQUIRED) - CRITICAL ACCURACY RULES
        ========================================

        **ULTRA-CRITICAL: DISTINGUISH BETWEEN LETTER INITIALS AND X MARKS**

        **TASK**: Find the TWO initial boxes on this {udc_name} LOA form.

        **STEP 1: LOCATE THE TWO INITIAL BOXES**
        - {udc_name} forms have TWO checkbox-style initial boxes
        - Usually located in the middle section of the document
        - They appear BEFORE or NEAR text like:
          * "Account/SDI Number Release"
          * "Interval Historical Energy Usage Data Release"
          * "Authorization" or "Data Release" text
        - Each box should have space for handwritten initials

        **STEP 2: SEARCH AREA EXPANSION**
        - Search for initials IN the box AND NEAR the box (floating initials are common!)
        - Initials may be:
          * Perfectly inside the box
          * Slightly above/below the box line
          * Floating near the box boundary
          * Written over or across box edges
        - **Accept initials ANYWHERE within the general box area** - don't require perfect positioning

        **STEP 3: DISTINGUISH LETTER INITIALS FROM X MARKS**

        **LETTER INITIALS** (VALID - mark as filled_with_initials=true):
        - **Two or more letters together**: PS, JD, AB, WA, EO, MK, TC, etc.
        - **Letters with periods/dots**: P.S., J.D., A.B., W.A. (ALWAYS initials, NEVER X)
        - **Letters with slashes**: P/S, J/D (some people write this way)
        - **Cursive/connected letters**: Two letters written in flowing script
        - **Single letters**: J, P, D, M (valid if clearly representing a name)
        - **Stacked letters**: One letter above another (some people write this way)

        **CRITICAL RULE**: If you see ANY recognizable letters (not just X), treat as VALID initials!

        **X MARK** (INVALID - mark as has_x_mark=true):
        - **ONLY flag as X if**: It's a single, standalone "X" character used as a checkmark
        - **NOT an X if**: Two letters that might visually resemble X when combined (like P and S crossing)
        - **NOT an X if**: Any text that includes letters other than X

        **COMMON INITIAL PATTERNS TO RECOGNIZE**:
        - First + Last name initials: PS (Peter Smith), JD (John Doe), AB (Alice Brown)
        - With periods: P.S., J.D., A.B., W.A., M.K.
        - Single letter initials: J, P, D, M, A
        - Stylized/cursive versions of the above
        - Underlined initials
        - Circled initials

        **STEP 4: EXAMINE EACH BOX**

        For EACH of the TWO boxes:
        1. Look IN and AROUND the box area
        2. Check for ANY letter-like marks (not just X)
        3. If you see letters → filled_with_initials=true
        4. If you see ONLY a single X → has_x_mark=true
        5. If you see NOTHING → box is empty

        **ULTRA-IMPORTANT: INITIALS DON'T NEED TO BE INSIDE BOXES**

        The boxes are just a GUIDE for where to put initials. What matters is:
        - Are there INITIALS present in the general area?
        - The initials can be:
          * Inside the box (perfect)
          * Outside the box but near it (still valid!)
          * Floating above/below the box line (still valid!)
          * Written across box boundaries (still valid!)
          * Anywhere in the initial section area (still valid!)

        **AS LONG AS INITIALS EXIST, MARK AS VALID** regardless of exact positioning!

        **FINAL VALIDATION**:
        - filled_box_count = boxes/areas with LETTER INITIALS anywhere nearby (any letters)
        - empty_box_count = boxes/areas with NO marks anywhere nearby
        - x_mark_count = boxes/areas with ONLY a standalone X mark (rare!)

        **REMINDER**: When in doubt, if you see ANY letters in or near the box area, treat as valid initials!

        Return JSON with this structure (ONLY initial boxes, nothing else):
        {{
          "initial_boxes": {{
            "box_1_filled_with_initials": true/false,
            "box_1_content": "initials or 'EMPTY' or 'X'",
            "box_2_filled_with_initials": true/false,
            "box_2_content": "initials or 'EMPTY' or 'X'",
            "filled_box_count": 0-2,
            "empty_box_count": 0-2,
            "x_mark_count": 0-2,
            "all_boxes_valid": true/false,
            "confidence": 0-100,
            "reasoning": "Detailed explanation of initial box findings"
          }}
        }}"""

        user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
        {initial_prompt}

        [ATTACHED_IMAGE]
        data:image/png;base64,{base64_image}"""

        system_prompt = f"You are an expert document analyzer specializing in {udc_name} LOA forms. Your task is to accurately validate initial boxes for letter initials."

        try:
            start_time = datetime.now()

            gpt_response_data = self.openai_4o_service.process_with_prompts(
                system_prompt, user_prompt, max_token=1500, raw_response=True
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            if gpt_response_data and gpt_response_data[0]["ai_result"]:
                analysis_text = gpt_response_data[0]["ai_result"][0]["result"]
            else:
                self.logger.error(
                    f"No response from OpenAI service for {udc_name} initial box verification"
                )
                return extraction_log

            json_match = re.search(r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(1))
            else:
                analysis_json = json.loads(analysis_text)

            self.logger.info(
                f"GPT-4o {udc_name} initial box verification completed in {processing_time:.2f} seconds"
            )

            # Use generic "initial_boxes" key
            initial_boxes_data = analysis_json.get("initial_boxes")
            if initial_boxes_data:
                # Initialize validation key if not present
                if validation_key not in extraction_log:
                    extraction_log[validation_key] = {}

                # Store initial boxes data
                extraction_log[validation_key]["initial_boxes"] = initial_boxes_data
                extraction_log[validation_key]["initial_boxes_gpt4o_verified"] = True

                self.logger.info(
                    f"{udc_name} Initial Boxes:\n"
                    f"  - Box 1: {initial_boxes_data.get('box_1_content')} (Valid: {initial_boxes_data.get('box_1_filled_with_initials')})\n"
                    f"  - Box 2: {initial_boxes_data.get('box_2_content')} (Valid: {initial_boxes_data.get('box_2_filled_with_initials')})\n"
                    f"  - Filled: {initial_boxes_data.get('filled_box_count')}, Empty: {initial_boxes_data.get('empty_box_count')}, X marks: {initial_boxes_data.get('x_mark_count')}"
                )

            return extraction_log

        except Exception as e:
            self.logger.error(f"{udc_name} initial box verification failed: {str(e)}")
            return extraction_log

    def verify_cinergy_initial_boxes_with_gpt4o(
        self, pdf_path: str, extraction_log: Dict
    ) -> Dict:
        """Wrapper method for CINERGY initial box verification - maintains backward compatibility."""
        return self.verify_initial_boxes_with_gpt4o(
            pdf_path,
            extraction_log,
            udc_name="CINERGY/Duke Energy Ohio",
            validation_key="cinergy_validation",
        )

    def verify_dayton_initial_boxes_with_gpt4o(
        self, pdf_path: str, extraction_log: Dict
    ) -> Dict:
        """Wrapper method for Dayton initial box verification - maintains backward compatibility."""
        return self.verify_initial_boxes_with_gpt4o(
            pdf_path,
            extraction_log,
            udc_name="Dayton Power & Light",
            validation_key="dayton_validation",
        )

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def scan_all_pages_for_dayton_accounts_with_gpt4o(
        self, pdf_path: str, extraction_log: Dict
    ) -> Dict:
        """
        Scan ALL pages of a PDF for Dayton Power & Light account numbers using GPT-4o Vision.
        ALWAYS scans all pages - never skips for "see attached" text.

        Dayton Account Number Format:
        - Standard: 12 digits + Z + 10 digits (e.g., 123456789012Z1234567890)
        - Lenient range: 11-13 digits before Z, 9-11 digits after Z
        - Total: 22-26 characters including the Z

        Args:
            pdf_path (str): Path to the PDF file.
            extraction_log (Dict): The current extraction log.

        Returns:
            Dict: Updated extraction_log with Dayton account numbers found across all pages.
        """
        self.logger.info(
            "Scanning ALL pages for Dayton Power & Light account numbers..."
        )

        # Extract all pages
        all_pages = self._extract_all_pdf_pages_as_images(pdf_path)
        if not all_pages:
            self.logger.error(
                "Failed to extract pages for Dayton account number scanning"
            )
            return extraction_log

        total_pages = len(all_pages)
        self.logger.info(
            f"Scanning {total_pages} page(s) for Dayton account numbers..."
        )

        # Collect account numbers from all pages
        all_account_numbers = []
        pages_with_accounts = []

        # Dayton-specific account scan prompt - LENIENT: Accept any long number 18-26 digits
        dayton_account_prompt = """CRITICAL: Search this page for Dayton Power & Light (DP&L / AES Ohio) account numbers.

        **DAYTON POWER & LIGHT ACCOUNT NUMBER FORMAT:**

        **LENIENT EXTRACTION: Any long number between 18-26 digits**
        - Accept ANY number that is 18-26 digits long
        - Z letter is OPTIONAL (welcome but not required)
        - Examples:
          * "910117129533Z109008636" (with Z) → VALID ✓
          * "2000010750750520800015560" (25 digits, no Z) → VALID ✓
          * "9101171295330109008636" (22 digits, no Z) → VALID ✓

        **WHERE TO LOOK:**
        - Near labels: "Account Number", "Acct #", "Account/SDI", "Account:", "Account No."
        - In the Account Number field on the form
        - Tables or lists of accounts
        - ANY long number on the page that's 18-26 digits

        **EXTRACTION RULES:**
        - Extract the EXACT digits (and Z if present) as shown
        - Remove spaces and dashes but KEEP the Z if present
        - Extract if total length is 18-26 digits (including Z as 1 character if present)
        - Count digits carefully
        - Do NOT modify or add digits

        **WHAT TO IGNORE:**
        - Phone numbers: (XXX) XXX-XXXX format (10 digits with formatting)
        - ZIP codes: 5 or 9 digits only
        - Dates: XX/XX/XXXX format
        - Short numbers: Less than 18 digits
        - Long numbers: More than 26 digits

        **EXAMPLES OF VALID EXTRACTIONS:**
        - "910117129533Z109008636" (23 chars with Z) → Extract: "910117129533Z109008636" ✓
        - "2000010750750520800015560" (25 digits) → Extract: "2000010750750520800015560" ✓
        - "9101171295330109008636" (22 digits) → Extract: "9101171295330109008636" ✓
        - "910-117129533-Z-109008636" → Extract: "910117129533Z109008636" (remove dashes, keep Z) ✓

        Return your findings in JSON format:
        {
          "account_numbers_found": ["account1", "account2", ...] or [],
          "accounts_found_count": 0 or number,
          "has_accounts_on_page": true/false,
          "confidence": 0-100,
          "notes": "Description of what you found and where"
        }"""

        system_prompt = "You are an expert at extracting Dayton Power & Light (DP&L) account numbers from documents. Find ALL long numbers (18-26 digits) on this page."

        for page_num, base64_image in all_pages:
            page_display = page_num + 1
            self.logger.info(f"  Scanning page {page_display} of {total_pages}...")

            user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
            Scan page {page_display} of {total_pages} for Dayton Power & Light account numbers.

            {dayton_account_prompt}

            [ATTACHED_IMAGE]
            data:image/png;base64,{base64_image}"""

            try:
                gpt_response_data = self.openai_4o_service.process_with_prompts(
                    system_prompt, user_prompt, max_token=1000, raw_response=True
                )

                if gpt_response_data and gpt_response_data[0]["ai_result"]:
                    analysis_text = gpt_response_data[0]["ai_result"][0]["result"]

                    # Parse JSON response
                    json_match = re.search(
                        r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL
                    )
                    if json_match:
                        page_result = json.loads(json_match.group(1))
                    else:
                        page_result = json.loads(analysis_text)

                    # Collect accounts from this page
                    page_accounts = page_result.get("account_numbers_found", [])
                    if page_accounts:
                        self.logger.info(
                            f"    Found {len(page_accounts)} account(s) on page {page_display}: {page_accounts}"
                        )
                        all_account_numbers.extend(page_accounts)
                        pages_with_accounts.append(page_display)
                else:
                    self.logger.warning(f"    No response for page {page_display}")

            except Exception as e:
                self.logger.error(f"    Error scanning page {page_display}: {str(e)}")
                continue

        # Deduplicate account numbers while preserving order
        processed_account_numbers = set()
        unique_accounts = []
        for acc in all_account_numbers:
            # Normalize: remove spaces/dashes but keep Z
            acc_normalized = re.sub(r"[\s\-]", "", str(acc).upper())
            if acc_normalized not in processed_account_numbers:
                processed_account_numbers.add(acc_normalized)
                unique_accounts.append(acc_normalized)

        # Validate Dayton account format - LENIENT: Accept 18-26 digits (with or without Z)
        valid_accounts = []
        invalid_accounts = []

        for acc in unique_accounts:
            acc_str = str(acc).upper()

            # Check if account has Z separator
            if "Z" in acc_str:
                # Format with Z: Total length should be 18-26 characters (digits + Z)
                # Extract digits only (excluding Z)
                digits_only = re.sub(r"[^0-9]", "", acc_str)
                total_length = len(digits_only) + 1  # +1 for the Z character

                if 18 <= total_length <= 26:
                    # Valid: Keep original format with Z
                    # Clean up: remove spaces/dashes but keep Z
                    parts = acc_str.split("Z")
                    if len(parts) == 2:
                        part1_digits = re.sub(r"[^0-9]", "", parts[0])
                        part2_digits = re.sub(r"[^0-9]", "", parts[1])
                        clean_account = f"{part1_digits}Z{part2_digits}"
                        valid_accounts.append(clean_account)
                    else:
                        # Multiple Z's - still try to clean it up
                        valid_accounts.append(re.sub(r"[\s\-]", "", acc_str))
                else:
                    invalid_accounts.append(
                        f"{acc} ({total_length} chars with Z - expected 18-26)"
                    )
            else:
                # No Z - just validate digit count
                digits_only = re.sub(r"[^0-9]", "", acc_str)

                # Accept any number with 18-26 digits
                if 18 <= len(digits_only) <= 26:
                    valid_accounts.append(digits_only)
                else:
                    invalid_accounts.append(
                        f"{acc} ({len(digits_only)} digits - expected 18-26)"
                    )

        # Ensure dayton_validation key exists
        if "dayton_validation" not in extraction_log:
            extraction_log["dayton_validation"] = {}

        validation_data = extraction_log["dayton_validation"]

        # Update validation data with results
        validation_data["account_numbers"] = valid_accounts
        validation_data["account_count"] = len(valid_accounts)
        validation_data["account_numbers_found"] = len(valid_accounts) > 0
        validation_data["invalid_length_accounts"] = invalid_accounts
        validation_data["account_length_valid"] = len(invalid_accounts) == 0
        validation_data["account_field_empty"] = len(valid_accounts) == 0

        # Add multi-page scan metadata
        validation_data["multi_page_scan"] = {
            "total_pages_scanned": total_pages,
            "pages_with_accounts": pages_with_accounts,
            "total_accounts_found": len(unique_accounts),
            "valid_accounts": len(valid_accounts),
            "invalid_accounts": len(invalid_accounts),
        }

        self.logger.info(
            f"Dayton multi-page account scan complete:\n"
            f"  - Total pages scanned: {total_pages}\n"
            f"  - Pages with accounts: {pages_with_accounts}\n"
            f"  - Valid accounts found: {len(valid_accounts)}\n"
            f"  - Invalid accounts: {len(invalid_accounts)}"
        )

        return extraction_log

    def _extract_all_pdf_pages_as_images(self, pdf_path: str) -> list:
        """Extract ALL pages from a PDF as images for comprehensive analysis.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of (page_num, base64_image) tuples
        """
        pages = []
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            self.logger.info(
                f"PDF has {total_pages} page(s) - extracting all for account number search"
            )

            for page_num in range(total_pages):
                page = doc[page_num]
                # Render the page to an image (higher resolution for better OCR)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                image_bytes = pix.tobytes("png")
                base64_image = self.ocr_integration.encode_image_to_base64(image_bytes)
                pages.append((page_num, base64_image))
                self.logger.info(f"  - Extracted page {page_num + 1} of {total_pages}")

            doc.close()
        except Exception as e:
            self.logger.error(f"Error extracting PDF pages: {str(e)}")

        return pages

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def scan_all_pages_for_account_numbers_with_gpt4o(
        self, pdf_path: str, extraction_log: Dict
    ) -> Dict:
        """
        Scan ALL pages of a PDF for account numbers using GPT-4o Vision.
        Works for BOTH FirstEnergy (20-digit) and AEP (17-digit or 11/17 format).

        CRITICAL: Account numbers are often on attachments/additional pages.
        This method scans EVERY page to find them.

        Args:
            pdf_path (str): Path to the PDF file.
            extraction_log (Dict): The current extraction log.

        Returns:
            Dict: Updated extraction_log with account numbers found across all pages.
        """
        # Determine which utility type we're scanning for
        is_aep = "aep_validation" in extraction_log
        is_firstenergy = "firstenergy_validation" in extraction_log

        utility_type = (
            "AEP" if is_aep else "FirstEnergy" if is_firstenergy else "Unknown"
        )
        self.logger.info(f"Scanning ALL pages for {utility_type} account numbers...")

        # Extract all pages
        all_pages = self._extract_all_pdf_pages_as_images(pdf_path)
        if not all_pages:
            self.logger.error("Failed to extract pages for account number scanning")
            return extraction_log

        total_pages = len(all_pages)
        self.logger.info(f"Scanning {total_pages} page(s) for account numbers...")

        # Collect account numbers from all pages
        all_account_numbers = []
        pages_with_accounts = []

        # Use different prompts based on utility type
        if is_aep:
            account_scan_prompt = """CRITICAL: Search this page for AEP account/SDI numbers.

            **AEP ACCOUNT NUMBER FORMATS:**

            **FORMAT 1: Standalone 17 digits** (most common)
            - Example: "00140060748972843" (17 digits)
            - OCR Tolerance: Accept 16-18 digits

            **FORMAT 2: 11/17 format with slash**
            - Example: "12345678901/12345678901234567"
            - Part 1: 10-12 digits
            - Part 2: 16-18 digits
            - Extract as SINGLE account (keep slash)

            **WHERE TO LOOK:**
            - Tables/spreadsheets with account numbers
            - Lists of accounts (one per line)
            - Account number columns
            - Any 16-18 digit numbers on the page
            - Numbers with 11/17 format (with slash)

            **EXTRACTION RULES:**
            - Extract the EXACT digits you see
            - Remove spaces and dashes (but KEEP slashes)
            - Count digits carefully
            - Do NOT modify or add digits

            **WHAT TO IGNORE:**
            - Phone numbers (10 digits)
            - Dates
            - ZIP codes
            - Short numbers (< 15 digits)

            Return your findings in JSON format:
            {
              "account_numbers_found": ["account1", "account2", ...] or [],
              "accounts_found_count": 0 or number,
              "has_accounts_on_page": true/false,
              "confidence": 0-100,
              "notes": "Description of what you found"
            }"""
        else:
            account_scan_prompt = """CRITICAL: Search this page for FirstEnergy account/SDI numbers.

            **FIRSTENERGY ACCOUNT NUMBER FORMATS:**

            **FORMAT 1: 12-digit/20-digit with slash** (e.g., "110011813539/08022677400000678558")
            - First part: 12 digits (account prefix)
            - Slash separator: /
            - Second part: 20 digits (SDI number)
            - Total with slash: 33 characters
            - Extract as SINGLE account: "110011813539/08022677400000678558"

            **FORMAT 2: Standalone 20 digits** (e.g., "08066361495001516800")
            - Exactly 20 digits (OCR tolerance: 19-21 digits acceptable)
            - No slash
            - May have spaces or dashes (remove them)

            **WHERE TO LOOK:**
            - Tables/spreadsheets with account numbers
            - Lists of accounts (one per line)
            - Account number columns
            - Any 20-digit numbers on the page
            - Numbers with 12/20 format (with slash)

            **EXTRACTION RULES:**
            - Extract the EXACT digits you see
            - Remove spaces and dashes (but KEEP slashes)
            - Count digits carefully - FirstEnergy accounts are 20 digits (or 12/20 format)
            - Do NOT modify or add digits
            - Do NOT confuse similar digits (0 vs O, 1 vs l)

            **WHAT TO IGNORE:**
            - Phone numbers (10 digits with area code)
            - Dates
            - ZIP codes
            - Short numbers (< 15 digits standalone)

            Return your findings in JSON format:
            {
              "account_numbers_found": ["account1", "account2", ...] or [],
              "accounts_found_count": 0 or number,
              "has_accounts_on_page": true/false,
              "confidence": 0-100,
              "notes": "Description of what you found"
            }"""

        system_prompt = f"You are an expert at extracting {utility_type} account/SDI numbers from documents. Find ALL account numbers on this page."

        for page_num, base64_image in all_pages:
            page_display = page_num + 1
            self.logger.info(f"  Scanning page {page_display} of {total_pages}...")

            user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
            Scan page {page_display} of {total_pages} for {utility_type} account numbers.

            {account_scan_prompt}

            [ATTACHED_IMAGE]
            data:image/png;base64,{base64_image}"""

            try:
                gpt_response_data = self.openai_4o_service.process_with_prompts(
                    system_prompt, user_prompt, max_token=1000, raw_response=True
                )

                if gpt_response_data and gpt_response_data[0]["ai_result"]:
                    analysis_text = gpt_response_data[0]["ai_result"][0]["result"]

                    # Parse JSON response
                    json_match = re.search(
                        r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL
                    )
                    if json_match:
                        page_result = json.loads(json_match.group(1))
                    else:
                        page_result = json.loads(analysis_text)

                    # Collect accounts from this page
                    page_accounts = page_result.get("account_numbers_found", [])
                    if page_accounts:
                        self.logger.info(
                            f"    Found {len(page_accounts)} account(s) on page {page_display}: {page_accounts}"
                        )
                        all_account_numbers.extend(page_accounts)
                        pages_with_accounts.append(page_display)
                else:
                    self.logger.warning(f"    No response for page {page_display}")

            except Exception as e:
                self.logger.error(f"    Error scanning page {page_display}: {str(e)}")
                continue

        # Deduplicate account numbers while preserving order
        processed_account_numbers = set()
        unique_accounts = []
        for acc in all_account_numbers:
            # Normalize: remove spaces/dashes but keep slashes
            acc_normalized = re.sub(r"[\s\-]", "", str(acc))
            if acc_normalized not in processed_account_numbers:
                processed_account_numbers.add(acc_normalized)
                unique_accounts.append(acc_normalized)

        # Determine which validation key to use
        if is_aep:
            validation_key = "aep_validation"
        elif is_firstenergy:
            validation_key = "firstenergy_validation"
        else:
            self.logger.error(
                "Cannot determine utility type - extraction_log has neither aep_validation nor firstenergy_validation"
            )
            return extraction_log

        # Ensure validation key exists
        if validation_key not in extraction_log:
            extraction_log[validation_key] = {}

        validation_data = extraction_log[validation_key]

        # Merge with existing account numbers (if any from page 1 extraction)
        existing_accounts = validation_data.get("account_numbers", [])
        merged_accounts = list(set(existing_accounts + unique_accounts))

        # Validate account format based on utility type
        valid_accounts = []
        invalid_accounts = []

        if is_aep:
            # AEP format validation: 17 digits or 11/17 format
            for acc in merged_accounts:
                acc_str = str(acc)

                if "/" in acc_str:
                    # 11/17 format validation
                    parts = acc_str.split("/")
                    if len(parts) == 2:
                        part1_digits = re.sub(r"[^0-9]", "", parts[0])
                        part2_digits = re.sub(r"[^0-9]", "", parts[1])

                        # Valid: 10-12 digits / 16-18 digits
                        if (
                            10 <= len(part1_digits) <= 12
                            and 16 <= len(part2_digits) <= 18
                        ):
                            valid_accounts.append(acc)
                        else:
                            invalid_accounts.append(
                                f"{acc} ({len(part1_digits)}/{len(part2_digits)} digits)"
                            )
                    else:
                        invalid_accounts.append(f"{acc} (invalid format)")
                else:
                    # Standalone format validation (16-18 digits for AEP)
                    digits_only = re.sub(r"[^0-9]", "", acc_str)
                    if 16 <= len(digits_only) <= 18:
                        valid_accounts.append(acc)
                    else:
                        invalid_accounts.append(f"{acc} ({len(digits_only)} digits)")
        else:
            # FirstEnergy format validation: 20 digits or 12/20 format
            for acc in merged_accounts:
                acc_str = str(acc)

                if "/" in acc_str:
                    # 12/20 format validation
                    parts = acc_str.split("/")
                    if len(parts) == 2:
                        part1_digits = re.sub(r"[^0-9]", "", parts[0])
                        part2_digits = re.sub(r"[^0-9]", "", parts[1])

                        # Valid: 12 digits / 19-21 digits
                        if len(part1_digits) == 12 and 19 <= len(part2_digits) <= 21:
                            valid_accounts.append(acc)
                        else:
                            invalid_accounts.append(
                                f"{acc} ({len(part1_digits)}/{len(part2_digits)} digits)"
                            )
                    else:
                        invalid_accounts.append(f"{acc} (invalid format)")
                else:
                    # Standalone format validation (19-21 digits with OCR tolerance)
                    digits_only = re.sub(r"[^0-9]", "", acc_str)
                    if 19 <= len(digits_only) <= 21:
                        valid_accounts.append(acc)
                    elif len(digits_only) == 12:
                        # Single 12-digit is invalid (missing SDI part)
                        invalid_accounts.append(f"{acc} (12 digits - needs 20)")
                    else:
                        invalid_accounts.append(f"{acc} ({len(digits_only)} digits)")

        # CRITICAL FIX: Remove has_attachment_indicator logic entirely
        # ONLY set account_numbers_found based on ACTUAL accounts found
        validation_data["account_numbers"] = valid_accounts
        validation_data["account_count"] = len(valid_accounts)
        validation_data["account_numbers_found"] = (
            len(valid_accounts) > 0
        )  # NO has_attachment_indicator!
        validation_data["invalid_length_accounts"] = invalid_accounts
        validation_data["account_length_valid"] = len(invalid_accounts) == 0
        validation_data["account_field_empty"] = (
            len(valid_accounts) == 0
        )  # True if NO accounts found

        # Add multi-page scan metadata
        validation_data["multi_page_scan"] = {
            "total_pages_scanned": total_pages,
            "pages_with_accounts": pages_with_accounts,
            "total_accounts_found": len(merged_accounts),
            "valid_accounts": len(valid_accounts),
            "invalid_accounts": len(invalid_accounts),
        }

        self.logger.info(
            f"Multi-page account scan complete:\n"
            f"  - Total pages scanned: {total_pages}\n"
            f"  - Pages with accounts: {pages_with_accounts}\n"
            f"  - Valid accounts found: {len(valid_accounts)}\n"
            f"  - Invalid accounts: {len(invalid_accounts)}"
        )

        return extraction_log

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def verify_firstenergy_comprehensive_with_gpt4o(
        self, pdf_path: str, extraction_log: Dict
    ) -> Dict:
        """
        GPT-4o comprehensive verification for First Energy (CEI, OE, TE) LOAs.
        Extracts ALL required fields including customer, CRES provider, Ohio phrase sections.

        CRITICAL: This function scans ALL PAGES of the PDF to find account numbers anywhere
        in the document (including attachments). Uses aggressive retry logic (50 attempts).

        Args:
            pdf_path (str): Path to the PDF file.
            extraction_log (Dict): The current extraction log from Azure Layout.

        Returns:
            Dict: Updated extraction log with ALL verified First Energy fields.
        """
        self.logger.info(
            "Performing GPT-4o comprehensive verification for First Energy LOA (ALL PAGES)..."
        )

        # CRITICAL: Extract ALL pages for comprehensive account number search
        all_pages = self._extract_all_pdf_pages_as_images(pdf_path)
        if not all_pages:
            self.logger.error(
                "Failed to extract any images from PDF for First Energy verification."
            )
            return {"success": False, "extraction_log": extraction_log}
        # Defensive check for first page image data
        if (not isinstance(all_pages[0], (list, tuple))) or (len(all_pages[0]) < 2):
            self.logger.error(
                "First page image data is missing or malformed for First Energy verification."
            )
            return {"success": False, "extraction_log": extraction_log}

        # Use first page for main form field extraction
        base64_image = all_pages[0][1]

        firstenergy_comprehensive_prompt = """CRITICAL: Extract ALL required fields from this First Energy LOA document (CEI/OE/TE - Ohio utilities).

        ========================================
        SECTION 1: CUSTOMER INFORMATION (Top of Document)
        ========================================

        1. CUSTOMER NAME: **CRITICAL - EXACT FIELD EXTRACTION**
           - **FIELD LABEL**: "CUSTOMER NAME" or "Customer Name"
           - **EXACT LOCATION**: In the "Customer Information" section at TOP of document
           - **CRITICAL**: Extract ONLY the value that appears IMMEDIATELY AFTER the "CUSTOMER NAME:" label and BEFORE the "Phone Number:" field
           - **POSITION RULE**:
             * Look for the line that starts with "CUSTOMER NAME:" or "Customer Name:"
             * Extract the text that appears on the SAME LINE or IMMEDIATELY BELOW this label
             * STOP extracting when you reach the "Phone Number:" field
           - **VALIDATION**:
             * If you see ONLY the field label "CUSTOMER NAME:" with blank space or underscores after it → return null (EMPTY)
             * DO NOT extract text from other fields (address, city, etc.)
             * City names that appear in address fields are NOT the customer name
           - **EXAMPLE**:
             * "CUSTOMER NAME: ABC Corporation" → Extract "ABC Corporation"
             * "CUSTOMER NAME: __________" → return null (EMPTY)
             * Address shows "Newburgh Heights, Ohio" but Customer Name is blank → return null, do NOT use "Newburgh Heights"

        2. PHONE NUMBER:
           - Labels: "Phone Number"
           - Format: (XXX) XXX-XXXX or variations
           - Extract the actual phone number

        3. CUSTOMER ADDRESS:
           - Labels: "Customer Address"
           - Extract full address (street, city, state, ZIP)

        4. AUTHORIZED PERSON/TITLE:
           - Labels: "Authorized Person/Title"
           - Extract the person's name and/or title

        5. ACCOUNT/SDI NUMBERS: **CAREFULLY READ EACH DIGIT EXACTLY AS SHOWN**

           Look at the document where it says "Account/SDI Number:"

           **CRITICAL: Extract account numbers EXACTLY as shown - do not add or remove any digits**

           **What do you see on the SAME LINE or NEXT LINE after this label?**

           Option A: You see NUMBERS with FORWARD SLASH (/) - This is 12-digit/20-digit format
           → Example: "110011813539/08022677400000678558" → Extract as SINGLE account: "110011813539/08022677400000678558"
           → **CRITICAL**: KEEP the forward slash (/) - this is ONE account number, NOT two!
           → **DO NOT split by slash** - treat "12digits/20digits" as a SINGLE account number
           → Result: ["110011813539/08022677400000678558"]

           Option B: You see NUMBERS WITHOUT slash (may have spaces/dashes)
           → Example: "110 171 225 300" → Extract as "110171225300" (remove spaces)
           → Example: "08066361495001516800" → Extract as "08066361495001516800" (exactly 20 digits)
           → **CRITICAL**: Count each digit carefully - do NOT add extra zeros or duplicate digits
           → **COMMON ERROR**: "08066361495001516800" should NOT become "080663614950051516800"
           → ACTION: Remove ONLY spaces/dashes, preserve all other digits exactly
           → Result: ["110171225300"] or ["08066361495001516800"]

           Option B: You see ONLY blank/"_____" with NO numbers
           → ACTION: account_numbers: []

           **CRITICAL: Extract numbers EVEN IF you also see "attach" text!**
           → "110 171 225 300 For multiple...attach" → STILL extract ["110171225300"]

           Then check SEPARATELY: Is there EXPLICIT confirmation of attachment?

           **CRITICAL: Distinguish between INSTRUCTIONS and ACTUAL ATTACHMENTS**

           INSTRUCTION TEXT (NOT an attachment):
           → "For multiple account/SDI numbers, please attach a spreadsheet"
           → "Please attach account list"
           → "Attach spreadsheet if needed"
           → These are INSTRUCTIONS, not confirmations - Mark as FALSE

           ACTUAL ATTACHMENT CONFIRMATION (IS an attachment):
           → "See attached" or "See attached spreadsheet"
           → "See below" (indicates accounts listed below)
           → "Accounts attached" or "List attached"
           → "Attached" (standalone, indicating attachment is present)
           → "Account numbers on attached spreadsheet"

           **RULE**: Only mark has_attachment_indicator: true if you see CONFIRMATION that attachment exists
           → If you see ONLY instruction text ("please attach"), mark as FALSE
           → If field is blank/empty with no attachment confirmation, mark as FALSE

           Then for EACH extracted number, count its digits:
           → "110171225300" has 12 digits → NOT 20 → Invalid
             → Add: invalid_length_accounts: ["110171225300 (12 digits)"]
             → Set: account_length_valid: false
           → "12345678901234567890" has 20 digits → Valid
             → Set: account_length_valid: true

           **FINAL CHECK - Is field completely EMPTY?**
           → If account_numbers array is EMPTY AND has_attachment_indicator is FALSE:
             → Set: account_field_empty: true
           → Otherwise:
             → Set: account_field_empty: false

        ========================================
        SECTION 2: CRES PROVIDER SECTION (Middle of Document)
        ========================================
        Look for "Competitive Retail Electric Service (CRES) Provider" section.

        6. CRES PROVIDER NAME:
           - Field: "CRES Name" (in CRES section)
           - Expected: "Constellation" or similar

        7. CRES ADDRESS:
           - Field: "Address" (in CRES section)
           - Extract full address

        8. CRES PHONE NUMBER:
           - Field: "Phone Number" (in CRES section)
           - Extract phone number

        9. CRES EMAIL:
           - Field: "Email" (in CRES section)
           - Extract email address

        ========================================
        SECTION 3: OHIO AUTHORIZATION STATEMENT (Bottom of Document)
        ========================================
        Look for the Ohio-specific authorization statement that begins with:
        "I realize that under the rules and regulations of the Public Utilities Commission of Ohio..."

        10. OHIO STATEMENT SIGNATURE:
           - Look for signature field AFTER the Ohio authorization text
           - Check if signature is PRESENT (any marks/text) or MISSING (blank)
           - **CRITICAL**: Extract the ACTUAL TEXT of the signature, not just presence
           - Look for patterns like:
             * "The Utilities Group on behalf of [Customer]" → BROKER signature
             * "John Smith" or actual person name → Valid customer signature
             * Any text containing "on behalf of" → BROKER signature

        11. OHIO STATEMENT DATE:
           - Look for "Date:" field AFTER the Ohio authorization text
           - Extract date in MM/DD/YYYY format

        12. UTILITY NAME IN OHIO PHRASE:
           - **CRITICAL VALIDATION**: The Ohio statement must reference ONE of these utilities:
             * CEI (Cleveland Electric Illuminating)
             * OE (Ohio Edison)
             * TE (Toledo Edison)
             * The Illuminating Company (can also be "Illuminating Co.")
           - Look for utility name in the phrase "allow [UTILITY NAME] to release"
           - **VALIDATE**: Check if it's one of the valid FirstEnergy utilities
           - **REJECT if**: Statement references a different utility (AEP, Duke, etc.)

        ========================================
        SECTION 4: INITIAL BOXES VALIDATION **CRITICAL FOR FIRSTENERGY**
        ========================================

        **ULTRA-CRITICAL**: FirstEnergy LOAs have TWO initial boxes that MUST both be filled with LETTER INITIALS.

        14. INITIAL BOX DETECTION:

           **STEP 1: LOCATE THE TWO INITIAL BOXES**
           - FirstEnergy forms have TWO checkbox-style initial boxes
           - Usually located in the middle section of the document
           - They appear BEFORE or NEAR text like:
             * "Account/SDI Number Release"
             * "Interval Historical Energy Usage Data Release"
           - Each box should have space for handwritten initials

           **STEP 2: EXAMINE EACH BOX CAREFULLY**

           For EACH of the TWO boxes, check INSIDE the box area:

           **FILLED BOX** (Valid - mark as filled_with_initials=true):
           - Contains LETTER initials (e.g., "AB", "JD", "WA", "EO")
           - Contains a person's handwritten initials
           - Contains typed letters that represent initials
           - **REQUIREMENT**: Must be actual LETTERS (A-Z), NOT symbols

           **EMPTY BOX** (Invalid - mark as filled_with_initials=false):
           - Box is completely blank/empty
           - Box contains only an underline: "_____"
           - Box contains only a border with nothing inside
           - Box exists but has NO content whatsoever

           **X MARK** (Invalid - mark as has_x_mark=true):
           - Box contains ONLY the letter "X" (not initials)
           - Box contains a checkmark symbol but NOT letter initials
           - Box is "marked" but NOT with actual letter initials
           - **CRITICAL**: "X" is NOT a valid initial - it must be rejected

           **VISUAL INSPECTION TIPS**:
           - Look for TWO distinct initial box areas
           - Check if each box has content or is blank
           - Distinguish between letter initials (AB, JD) vs X marks (X, ✓)
           - Empty boxes may appear as blank squares or boxes with underscores

           **STEP 3: COUNT THE RESULTS**
           - Count boxes with LETTER INITIALS: filled_box_count
           - Count boxes that are EMPTY: empty_box_count
           - Count boxes with X MARKS (not initials): x_mark_count
           - Total boxes should = 2 (two initial boxes on FirstEnergy forms)

           **CRITICAL VALIDATION**:
           - If filled_box_count = 2 → VALID (both boxes have letter initials)
           - If filled_box_count = 1 → INVALID (one box empty or has X)
           - If filled_box_count = 0 → INVALID (both boxes empty or have X marks)
           - If empty_box_count > 0 → INVALID (at least one box is empty)
           - If x_mark_count > 0 → INVALID (at least one box has X instead of initials)

        ========================================
        SECTION 5: FORM TYPE VALIDATION
        ========================================

        15. FORM IDENTIFYING MARKERS:
           - Look for AT LEAST ONE of these phrases:
             * "Ohio Customer Letter of Authorization For Release of Customer's Electric Utility Account Number/SDI and/or Residential Historical Interval Data"
             * "Rev. 03-01-2016"
             * "FORM NO. X-4428 (04-16)" or "FORM NO. X-4428"
           - Mark form_type_valid=true if you find AT LEAST ONE of these markers
           - Mark form_type_valid=false if NONE are found

        ========================================
        CRITICAL EXTRACTION RULES:
        ========================================
        - Extract ACTUAL VALUES, not field labels
        - For signatures: Look for actual names/marks, NOT just the field label
        - If field is blank/empty, return null
        - For account numbers: Need 8+ digits or attachment indicator
        - For Ohio utility: Must be CEI, OE, TE, or Illuminating Company
        - For form type: Need at least ONE identifying marker

        Return JSON with this structure:
        {
          "firstenergy_comprehensive_extraction": {
            "customer_name": "value or null",
            "customer_phone": "value or null",
            "customer_address": "value or null",
            "authorized_person_title": "value or null",
            "account_numbers": ["num1", "num2"] or [],
            "has_attachment_indicator": true/false,
            "account_field_empty": true/false,
            "account_length_valid": true/false,
            "invalid_length_accounts": ["account1 (X digits)", "account2 (Y digits)"] or [],
            "cres_name": "value or null",
            "cres_address": "value or null",
            "cres_phone": "value or null",
            "cres_email": "value or null",
            "ohio_signature_present": true/false,
            "ohio_signature_text": "actual signature text/name or null",
            "ohio_signature_date": "MM/DD/YYYY or null",
            "ohio_phrase_utility": "CEI/OE/TE/Illuminating Company or null",
            "ohio_phrase_utility_valid": true/false,
            "form_type_valid": true/false,
            "form_identifiers_found": ["list of identifiers found"],
            "initial_boxes": {
              "box_1_filled_with_initials": true/false,
              "box_1_content": "initials text or 'EMPTY' or 'X'",
              "box_2_filled_with_initials": true/false,
              "box_2_content": "initials text or 'EMPTY' or 'X'",
              "filled_box_count": 0, 1, or 2,
              "empty_box_count": 0, 1, or 2,
              "x_mark_count": 0, 1, or 2,
              "all_boxes_valid": true/false
            },
            "confidence": 0-100,
            "reasoning": "Detailed explanation of ALL findings including initial boxes, account fields, and validation results"
          }
        }"""

        user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
        {firstenergy_comprehensive_prompt}

        [ATTACHED_IMAGE]
        data:image/png;base64,{base64_image}"""

        system_prompt = "You are an expert document analyzer specializing in First Energy (CEI/OE/TE) LOA forms. Extract ALL required fields with high accuracy, validating form type and Ohio authorization statement utilities."

        try:
            start_time = datetime.now()

            gpt_response_data = self.openai_4o_service.process_with_prompts(
                system_prompt, user_prompt, max_token=2500, raw_response=True
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            if gpt_response_data and gpt_response_data[0]["ai_result"]:
                analysis_text = gpt_response_data[0]["ai_result"][0]["result"]
            else:
                self.logger.error(
                    "No response from OpenAI service for First Energy comprehensive verification"
                )
                if "firstenergy_validation" not in extraction_log:
                    extraction_log["firstenergy_validation"] = {}
                extraction_log["firstenergy_validation"][
                    "gpt4o_verification_error"
                ] = "No response from OpenAI service"
                return {"success": False, "extraction_log": extraction_log}

            json_match = re.search(r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(1))
            else:
                analysis_json = json.loads(analysis_text)

            self.logger.info(
                f"GPT-4o First Energy comprehensive verification completed in {processing_time:.2f} seconds"
            )

            fe_data = analysis_json.get("firstenergy_comprehensive_extraction")
            if fe_data:
                self.logger.info(
                    "GPT-4o First Energy Comprehensive Extraction Results:"
                )
                self.logger.info(
                    f"  - Customer Name: {fe_data.get('customer_name', 'Not found')}"
                )
                self.logger.info(
                    f"  - Customer Phone: {fe_data.get('customer_phone', 'Not found')}"
                )
                self.logger.info(
                    f"  - Customer Address: {fe_data.get('customer_address', 'Not found')[:50] if fe_data.get('customer_address') else 'Not found'}..."
                )
                self.logger.info(
                    f"  - Authorized Person/Title: {fe_data.get('authorized_person_title', 'Not found')}"
                )
                self.logger.info(
                    f"  - Account Numbers: {len(fe_data.get('account_numbers', []))} found"
                )
                self.logger.info(
                    f"  - Account Numbers: {fe_data.get('account_numbers', [])}"
                )
                self.logger.info(
                    f"  - Has Attachment: {fe_data.get('has_attachment_indicator', False)}"
                )
                self.logger.info(
                    f"  - Account Length Valid (20 digits): {fe_data.get('account_length_valid', True)}"
                )
                self.logger.info(
                    f"  - Invalid Length Accounts: {fe_data.get('invalid_length_accounts', [])}"
                )
                self.logger.info(
                    f"  - CRES Name: {fe_data.get('cres_name', 'Not found')}"
                )
                self.logger.info(
                    f"  - CRES Address: {fe_data.get('cres_address', 'Not found')[:50] if fe_data.get('cres_address') else 'Not found'}..."
                )
                self.logger.info(
                    f"  - CRES Phone: {fe_data.get('cres_phone', 'Not found')}"
                )
                self.logger.info(
                    f"  - CRES Email: {fe_data.get('cres_email', 'Not found')}"
                )
                self.logger.info(
                    f"  - Ohio Signature Present: {fe_data.get('ohio_signature_present', False)}"
                )
                self.logger.info(
                    f"  - Ohio Date: {fe_data.get('ohio_signature_date', 'Not found')}"
                )
                self.logger.info(
                    f"  - Ohio Phrase Utility: {fe_data.get('ohio_phrase_utility', 'Not found')}"
                )
                self.logger.info(
                    f"  - Ohio Phrase Utility Valid: {fe_data.get('ohio_phrase_utility_valid', False)}"
                )
                self.logger.info(
                    f"  - Form Type Valid: {fe_data.get('form_type_valid', False)}"
                )
                self.logger.info(
                    f"  - Form Identifiers Found: {fe_data.get('form_identifiers_found', [])}"
                )

                # Update or initialize firstenergy_validation in extraction_log
                if "firstenergy_validation" not in extraction_log:
                    extraction_log["firstenergy_validation"] = {}

                # Map all extracted fields
                extraction_log["firstenergy_validation"]["customer_name_found"] = bool(
                    fe_data.get("customer_name")
                )
                extraction_log["firstenergy_validation"]["customer_name"] = fe_data.get(
                    "customer_name"
                )

                extraction_log["firstenergy_validation"]["customer_phone_found"] = bool(
                    fe_data.get("customer_phone")
                )
                extraction_log["firstenergy_validation"]["customer_phone"] = (
                    fe_data.get("customer_phone")
                )

                extraction_log["firstenergy_validation"]["customer_address_found"] = (
                    bool(fe_data.get("customer_address"))
                )
                extraction_log["firstenergy_validation"]["customer_address"] = (
                    fe_data.get("customer_address")
                )

                extraction_log["firstenergy_validation"][
                    "authorized_person_title_found"
                ] = bool(fe_data.get("authorized_person_title"))
                extraction_log["firstenergy_validation"]["authorized_person_title"] = (
                    fe_data.get("authorized_person_title")
                )

                account_nums = fe_data.get("account_numbers", [])
                has_attachment = fe_data.get("has_attachment_indicator", False)
                account_field_empty = fe_data.get("account_field_empty", False)

                # CRITICAL: Validate account number length (typically 20 digits, allow 19-21 for OCR variations)
                # ALWAYS validate visible account numbers, regardless of attachment indicator
                account_length_valid = True
                invalid_length_accounts = []

                if account_nums:
                    # Check each account number - accept 19-21 digits OR 12-digit/19-21-digit format
                    for acc_num in account_nums:
                        acc_num_str = str(acc_num)

                        # Check for 12-digit/20-digit format (e.g., "2314253435/92395382057305729897")
                        if "/" in acc_num_str:
                            parts = acc_num_str.split("/")
                            if len(parts) == 2:
                                # Extract digits from each part
                                part1_digits = re.sub(r"[^0-9]", "", parts[0])
                                part2_digits = re.sub(r"[^0-9]", "", parts[1])

                                # Valid format: 12 digits / 19-21 digits (allow OCR tolerance on second part)
                                if (
                                    len(part1_digits) == 12
                                    and 19 <= len(part2_digits) <= 21
                                ):
                                    # Valid 12/20 format with OCR tolerance
                                    continue
                                else:
                                    account_length_valid = False
                                    invalid_length_accounts.append(
                                        f"{acc_num} (invalid format: {len(part1_digits)} / {len(part2_digits)} digits, expected 12 / 19-21)"
                                    )
                            else:
                                # Multiple slashes - invalid format
                                account_length_valid = False
                                invalid_length_accounts.append(
                                    f"{acc_num} (invalid format: multiple slashes)"
                                )
                        else:
                            # No slash - must be standard 20-digit format (allow 19-21 for OCR tolerance)
                            digits_only = re.sub(r"[^0-9]", "", acc_num_str)

                            # Reject single 12-digit numbers
                            if len(digits_only) == 12:
                                account_length_valid = False
                                invalid_length_accounts.append(
                                    f"{acc_num} (12 digits - must be 19-21 digits or 12/20 format)"
                                )
                            # Accept 19-21 digit numbers (20 digits with OCR tolerance)
                            elif len(digits_only) < 19 or len(digits_only) > 21:
                                account_length_valid = False
                                invalid_length_accounts.append(
                                    f"{acc_num} ({len(digits_only)} digits - must be 19-21 digits or 12/20 format)"
                                )

                    self.logger.info("Account length validation results:")
                    self.logger.info(
                        f"  - Valid length (19-21 digits): {account_length_valid}"
                    )
                    if invalid_length_accounts:
                        self.logger.info(
                            f"  - Invalid accounts (must be 19-21 digits): {invalid_length_accounts}"
                        )

                # Determine if account numbers are found (either actual numbers OR attachment)
                extraction_log["firstenergy_validation"]["account_numbers_found"] = (
                    bool(account_nums or has_attachment)
                )
                extraction_log["firstenergy_validation"][
                    "account_numbers"
                ] = account_nums
                extraction_log["firstenergy_validation"]["account_count"] = len(
                    account_nums
                )
                extraction_log["firstenergy_validation"][
                    "has_attachment_indicator"
                ] = has_attachment
                extraction_log["firstenergy_validation"][
                    "account_field_empty"
                ] = account_field_empty
                extraction_log["firstenergy_validation"][
                    "account_length_valid"
                ] = account_length_valid
                extraction_log["firstenergy_validation"][
                    "invalid_length_accounts"
                ] = invalid_length_accounts

                extraction_log["firstenergy_validation"]["cres_name_found"] = bool(
                    fe_data.get("cres_name")
                )
                extraction_log["firstenergy_validation"]["cres_name"] = fe_data.get(
                    "cres_name"
                )

                extraction_log["firstenergy_validation"]["cres_address_found"] = bool(
                    fe_data.get("cres_address")
                )
                extraction_log["firstenergy_validation"]["cres_address"] = fe_data.get(
                    "cres_address"
                )

                extraction_log["firstenergy_validation"]["cres_phone_found"] = bool(
                    fe_data.get("cres_phone")
                )
                extraction_log["firstenergy_validation"]["cres_phone"] = fe_data.get(
                    "cres_phone"
                )

                extraction_log["firstenergy_validation"]["cres_email_found"] = bool(
                    fe_data.get("cres_email")
                )
                extraction_log["firstenergy_validation"]["cres_email"] = fe_data.get(
                    "cres_email"
                )

                extraction_log["firstenergy_validation"]["ohio_signature_found"] = (
                    fe_data.get("ohio_signature_present", False)
                )
                extraction_log["firstenergy_validation"]["ohio_signature_text"] = (
                    fe_data.get("ohio_signature_text")
                )
                extraction_log["firstenergy_validation"]["ohio_signature_date"] = (
                    fe_data.get("ohio_signature_date")
                )
                extraction_log["firstenergy_validation"]["ohio_date_found"] = bool(
                    fe_data.get("ohio_signature_date")
                )

                extraction_log["firstenergy_validation"]["ohio_phrase_utility"] = (
                    fe_data.get("ohio_phrase_utility")
                )
                extraction_log["firstenergy_validation"][
                    "ohio_phrase_utility_valid"
                ] = fe_data.get("ohio_phrase_utility_valid", False)

                extraction_log["firstenergy_validation"]["form_type_valid"] = (
                    fe_data.get("form_type_valid", False)
                )
                extraction_log["firstenergy_validation"]["form_identifiers_found"] = (
                    fe_data.get("form_identifiers_found", [])
                )

                # Extract initial boxes data
                initial_boxes = fe_data.get("initial_boxes", {})
                if initial_boxes:
                    self.logger.info("GPT-4o Initial Boxes Detection:")
                    self.logger.info(
                        f"  - Box 1 Filled: {initial_boxes.get('box_1_filled_with_initials', False)} (Content: {initial_boxes.get('box_1_content', 'Unknown')})"
                    )
                    self.logger.info(
                        f"  - Box 2 Filled: {initial_boxes.get('box_2_filled_with_initials', False)} (Content: {initial_boxes.get('box_2_content', 'Unknown')})"
                    )
                    self.logger.info(
                        f"  - Filled Box Count: {initial_boxes.get('filled_box_count', 0)}"
                    )
                    self.logger.info(
                        f"  - Empty Box Count: {initial_boxes.get('empty_box_count', 0)}"
                    )
                    self.logger.info(
                        f"  - X Mark Count: {initial_boxes.get('x_mark_count', 0)}"
                    )
                    self.logger.info(
                        f"  - All Boxes Valid: {initial_boxes.get('all_boxes_valid', False)}"
                    )

                    # Store initial boxes data in extraction_log
                    extraction_log["firstenergy_validation"]["initial_boxes"] = {
                        "box_1_filled_with_initials": initial_boxes.get(
                            "box_1_filled_with_initials", False
                        ),
                        "box_1_content": initial_boxes.get("box_1_content", "EMPTY"),
                        "box_2_filled_with_initials": initial_boxes.get(
                            "box_2_filled_with_initials", False
                        ),
                        "box_2_content": initial_boxes.get("box_2_content", "EMPTY"),
                        "filled_box_count": initial_boxes.get("filled_box_count", 0),
                        "empty_box_count": initial_boxes.get("empty_box_count", 0),
                        "x_mark_count": initial_boxes.get("x_mark_count", 0),
                        "all_boxes_valid": initial_boxes.get("all_boxes_valid", False),
                    }
                else:
                    self.logger.warning("GPT-4o did not return initial_boxes data")
                    extraction_log["firstenergy_validation"]["initial_boxes"] = {
                        "box_1_filled_with_initials": False,
                        "box_1_content": "NOT_DETECTED",
                        "box_2_filled_with_initials": False,
                        "box_2_content": "NOT_DETECTED",
                        "filled_box_count": 0,
                        "empty_box_count": 0,
                        "x_mark_count": 0,
                        "all_boxes_valid": False,
                        "detection_error": "Initial boxes data not returned by GPT-4o",
                    }

                extraction_log["firstenergy_validation"]["gpt4o_verified"] = True
                extraction_log["firstenergy_validation"][
                    "gpt4o_verification_details"
                ] = fe_data
                extraction_log["firstenergy_validation"][
                    "validation_method"
                ] = "gpt4o_vision"

                self.logger.info(
                    "Updated First Energy validation with GPT-4o comprehensive extraction including initial boxes."
                )

                return {"success": True, "extraction_log": extraction_log}
            else:
                self.logger.warning(
                    "GPT-4o did not return the expected 'firstenergy_comprehensive_extraction' key."
                )
                if "firstenergy_validation" not in extraction_log:
                    extraction_log["firstenergy_validation"] = {}
                extraction_log["firstenergy_validation"][
                    "gpt4o_verification_error"
                ] = "Invalid JSON structure from GPT-4o"
                return {"success": False, "extraction_log": extraction_log}

        except Exception as e:
            error_msg = (
                f"GPT-4o First Energy comprehensive verification failed: {str(e)}"
            )
            self.logger.error(error_msg)
            if "firstenergy_validation" not in extraction_log:
                extraction_log["firstenergy_validation"] = {}
            extraction_log["firstenergy_validation"][
                "gpt4o_verification_error"
            ] = f"Exception: {str(e)}"
            return {"success": False, "extraction_log": extraction_log, "error": str(e)}

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def extract_comprehensive_data_with_gpt4o(self, pdf_path: str) -> Dict:
        """
        Use GPT-4o vision to perform a comprehensive extraction of all critical LOA data in a single call.
        This extracts:
        1. Service options selection
        2. Customer signature presence and date
        3. Requestor/Billing signature presence and date
        4. Requestor/Billing information fields
        """
        self.logger.info(
            "Performing comprehensive data extraction with GPT-4o Vision..."
        )

        image_data = self.ocr_integration.extract_pdf_image(pdf_path, page_num=0)
        if not image_data:
            self.logger.error(
                "Failed to extract image from PDF for comprehensive data extraction."
            )
            return {"success": False, "error": "Failed to extract image from PDF"}

        base64_image = self.ocr_integration.encode_image_to_base64(image_data)

        comprehensive_prompt = """CRITICAL: Analyze this BECO/Eversource LOA document image and extract ALL critical data fields.

        You must extract data from FIVE sections:

        ========================================
        SECTION 1: SERVICE OPTIONS (Top of Document)
        ========================================
        Look for a section with TWO pricing options:
        1. "One Time Request, $50.00 per account number"
        2. "Annual Subscription, $300.00 per account per year"

        For each option, determine if its checkbox is SELECTED or UNSELECTED:
        - SELECTED: Contains X, checkmark (✓), filled square (■), or any mark INSIDE the checkbox
        - UNSELECTED: Empty box (☐) or blank checkbox with NO marks inside

        **CRITICAL**: Be VERY careful - only mark as selected if you see a CLEAR mark INSIDE the checkbox

        ========================================
        SECTION 2: CUSTOMER INFORMATION (Middle/Bottom of Document)
        ========================================
        Look for "Customer Information and Authorization" section.

        Extract:
        A. Customer Signature:
           - Field: "Customer's Signature (please print)" or "Customer Signature"
           - **STRICT VALIDATION**: Check if there's an ACTUAL SIGNATURE (handwritten name, typed name, or initials)
           - **DO NOT accept field labels**: If you only see "Customer's Signature" (the field label itself), mark as MISSING
           - **ONLY mark as PRESENT if**:
             * You see a person's actual name (e.g., "William Anastasia", "John Smith")
             * You see handwritten text/marks that represent a signature
             * You see initials (e.g., "JD", "AB")
             * You see e-signature markers like "E-Signed", "DocuSign" with name
           - **Mark as MISSING if**:
             * Field only contains the label "Customer's Signature" or "Signature"
             * Field is completely blank
             * Field only has placeholder text like "______"

        B. Customer Signature Date:
           - Field: "Date" near customer signature, or "Date signed by customer"
           - Look for date in format MM/DD/YYYY, MM-DD-YYYY, or "E-Signed: MM/DD/YYYY HH:MM"
           - Extract the actual date value

        ========================================
        SECTION 3: REQUESTOR/BILLING INFORMATION (Bottom of Document)
        ========================================
        Look for "Requestor & Billing Information" section.

        Extract ALL of these fields:
        A. Company Name:
           - Field: "Requestor/Billing Company", "Requestor Company"
           - Expected: "Constellation", "Constellation New Energy", etc.

        B. Contact Name:
           - Field: "Requestor/Billing Name", "Contact Name"
           - Expected: A person's name

        C. Phone Number:
           - Field: "Phone Number", "Phone"
           - Format: (XXX) XXX-XXXX or XXX-XXX-XXXX

        D. Email Address:
           - Field: "Email Address", "Email"
           - Expected: email@constellation.com

        E. Billing Address:
           - Field: "Billing Address", "Address"
           - Format: Street, City, State, ZIP

        F. Requestor/Billing Signature:
           - Field: "Requestor/Billing Signature" or "Signature"
           - **STRICT VALIDATION**: Same rules as customer signature
           - **ONLY mark as PRESENT if you see an actual name/signature**
           - **DO NOT accept field labels like "Requestor/Billing Signature"**

        G. Requestor/Billing Signature Date:
           - Field: "Dated Signed by Requestor/Billing Co." or "Date"
           - Look for date in format MM/DD/YYYY or MM-DD-YYYY
           - Extract the actual date value

        ========================================
        SECTION 4: ACCOUNT NUMBERS (Anywhere in Document)
        ========================================
        Look for account numbers in the document:
        - Field labels: "Account Number", "Acct Number", "Account Numbers", "Account List"
        - Account numbers are typically 8-20 digit numbers
        - May appear in tables, lists, or separate sections
        - Check ALL pages for account numbers

        **VALIDATION**:
        - Mark has_account_numbers=true if you find at least ONE account number
        - Mark has_account_numbers=false if no account numbers are visible
        - Extract up to 3 account numbers as examples

        ========================================
        CRITICAL EXTRACTION RULES:
        ========================================
        - Extract ACTUAL VALUES, not field labels
        - If field is garbled but readable, extract closest match
        - If field is truly blank/empty, return null
        - **For signatures: ONLY accept actual names/marks, NOT field labels**
        - For dates: Extract the date value, not the field label
        - **STRICT**: Field label ≠ actual signature

        Return a JSON object with this EXACT structure:
        {
        "comprehensive_extraction": {
            "service_options": {
                "one_time_selected": true/false,
                "annual_subscription_selected": true/false,
                "selection_count": 0, 1, or 2
            },
            "signatures": {
                "customer_signature_present": true/false,
                "customer_signature_text": "actual name/text found or null (NOT field label)",
                "customer_signature_date": "MM/DD/YYYY or null",
                "requestor_signature_present": true/false,
                "requestor_signature_text": "actual name/text found or null (NOT field label)",
                "requestor_signature_date": "MM/DD/YYYY or null"
            },
            "requestor_billing_info": {
                "company_name": "extracted value or null",
                "contact_name": "extracted value or null",
                "phone_number": "extracted value or null",
                "email_address": "extracted value or null",
                "billing_address": "extracted value or null"
            },
            "account_numbers": {
                "has_account_numbers": true/false,
                "account_numbers_found": ["account1", "account2", "account3"] or [],
                "account_count": 0 or number found
            },
            "confidence": 0-100,
            "reasoning": "Detailed explanation for each section"
        }
        }"""

        user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
        {comprehensive_prompt}

        [ATTACHED_IMAGE]
        data:image/png;base64,{base64_image}"""

        system_prompt = "You are an expert document analyzer. Your task is to extract all specified fields from the LOA document image with high accuracy."

        try:
            start_time = datetime.now()

            gpt_response_data = self.openai_4o_service.process_with_prompts(
                system_prompt, user_prompt, max_token=2000, raw_response=True
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            if gpt_response_data and gpt_response_data[0]["ai_result"]:
                analysis_text = gpt_response_data[0]["ai_result"][0]["result"]
            else:
                self.logger.error(
                    "No response from OpenAI service for comprehensive data extraction."
                )
                return {"success": False, "error": "No response from OpenAI service"}

            json_match = re.search(r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(1))
            else:
                analysis_json = json.loads(analysis_text)

            self.logger.info(
                f"GPT-4o comprehensive data extraction completed in {processing_time:.2f} seconds"
            )

            extraction_data = analysis_json.get("comprehensive_extraction")
            if extraction_data:
                return {"success": True, "data": extraction_data}
            else:
                self.logger.warning(
                    "GPT-4o did not return the expected 'comprehensive_extraction' key."
                )
                return {"success": False, "error": "Invalid JSON structure from GPT-4o"}

        except Exception as e:
            error_msg = f"GPT-4o comprehensive data extraction failed: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": f"Exception: {str(e)}"}

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def extract_customer_name_from_great_lakes_loa(
        self, pdf_path: str, udc: str
    ) -> Dict:
        """
        Extract customer name from Great Lakes UDC LOAs using GPT-4o Vision.

        ALL Great Lakes UDCs have customer name in the "Customer Name" field.

        Args:
            pdf_path (str): Path to the PDF file.
            udc (str): The utility distribution company code (e.g., "AEP", "COMED", "DAYTON").

        Returns:
            Dict: Contains extracted customer name and metadata.
        """
        self.logger.info(f"Extracting customer name from {udc} LOA using GPT-4o...")

        image_data = self.ocr_integration.extract_pdf_image(pdf_path, page_num=0)
        if not image_data:
            self.logger.error(
                "Failed to extract image from PDF for customer name extraction."
            )
            return {
                "success": False,
                "customer_name": None,
                "error": "Failed to extract image from PDF",
            }

        base64_image = self.ocr_integration.encode_image_to_base64(image_data)

        # All Great Lakes UDCs use the same Customer Name field
        extraction_prompt = """CRITICAL: Extract the CUSTOMER NAME from this Great Lakes LOA document.

        **WHERE TO LOOK:**
        - Look for a field labeled "Customer Name", "CUSTOMER NAME", or "Customer Name (as it appears on the bill)"
        - This is typically in the "Customer Information" section at the top of the document
        - The customer name should appear IMMEDIATELY AFTER or BELOW this label

        **EXTRACTION RULES:**
        - Extract ONLY the actual customer/company name value
        - DO NOT extract field labels like "Customer Name:" or "CUSTOMER NAME:"
        - DO NOT extract other fields like address, city, or phone number
        - If the field is blank or contains only underscores/placeholders, return null
        - Remove any leading/trailing whitespace

        **VALIDATION:**
        - If you see ONLY the field label with no value → return null (EMPTY)
        - If you see a city name in an address field, that's NOT the customer name
        - The customer name is the business/company name, not a location

        **EXAMPLES:**
        - "CUSTOMER NAME: ABC Corporation" → Extract "ABC Corporation"
        - "Customer Name: __________" → return null (EMPTY)
        - Address shows "Cleveland, Ohio" but Customer Name is blank → return null

        Return your findings in JSON format:
        {
            "customer_name_extraction": {
                "customer_name": "extracted name or null",
                "found": true/false,
                "confidence": 0-100,
                "location": "Description of where found",
                "reasoning": "Explanation of extraction"
            }
        }"""

        user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
        Extract customer name from this {udc} LOA document.

        {extraction_prompt}

        [ATTACHED_IMAGE]
        data:image/png;base64,{base64_image}"""

        system_prompt = f"You are an expert at extracting customer names from {udc} LOA documents. Extract the customer name from the Customer Name field."

        try:
            start_time = datetime.now()

            gpt_response_data = self.openai_4o_service.process_with_prompts(
                system_prompt, user_prompt, max_token=1000, raw_response=True
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            if gpt_response_data and gpt_response_data[0]["ai_result"]:
                analysis_text = gpt_response_data[0]["ai_result"][0]["result"]
            else:
                self.logger.error(
                    "No response from OpenAI service for customer name extraction"
                )
                return {
                    "success": False,
                    "customer_name": None,
                    "error": "No response from OpenAI service",
                }

            json_match = re.search(r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(1))
            else:
                analysis_json = json.loads(analysis_text)

            self.logger.info(
                f"GPT-4o customer name extraction completed in {processing_time:.2f} seconds"
            )

            extraction_data = analysis_json.get("customer_name_extraction")
            if extraction_data:
                customer_name = extraction_data.get("customer_name")
                found = extraction_data.get("found", False)
                confidence = extraction_data.get("confidence", 0)
                location = extraction_data.get("location", "Unknown")
                reasoning = extraction_data.get("reasoning", "No reasoning provided")

                self.logger.info("GPT-4o Customer Name Extraction Results:")
                self.logger.info(f"  - Customer Name: {customer_name}")
                self.logger.info(f"  - Found: {found}")
                self.logger.info(f"  - Confidence: {confidence}%")
                self.logger.info(f"  - Location: {location}")

                return {
                    "success": True,
                    "customer_name": customer_name,
                    "found": found,
                    "confidence": confidence,
                    "location": location,
                    "reasoning": reasoning,
                    "udc": udc,
                }
            else:
                self.logger.warning(
                    "GPT-4o did not return the expected 'customer_name_extraction' key."
                )
                return {
                    "success": False,
                    "customer_name": None,
                    "error": "Invalid JSON structure from GPT-4o",
                }

        except Exception as e:
            error_msg = f"GPT-4o customer name extraction failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "customer_name": None,
                "error": f"Exception: {str(e)}",
            }

    def compare_account_names(
        self, salesforce_account_name: str, loa_customer_name: str
    ) -> Dict:
        """
        Compare account name from Salesforce with customer name extracted from LOA.
        Uses punctuation-tolerant normalization to handle formatting differences.

        Args:
            salesforce_account_name (str): Account name from Salesforce.
            loa_customer_name (str): Customer name extracted from LOA.

        Returns:
            Dict: Comparison result with match status and details.
        """
        self.logger.info("Comparing account names...")
        self.logger.info(f"  - Salesforce Account Name: {salesforce_account_name}")
        self.logger.info(f"  - LOA Customer Name: {loa_customer_name}")

        if not salesforce_account_name or not loa_customer_name:
            return {
                "match": False,
                "reason": "Missing account name - either Salesforce or LOA customer name is empty",
                "salesforce_name": salesforce_account_name,
                "loa_name": loa_customer_name,
            }

        def normalize_name_for_comparison(name: str) -> str:
            """
            Normalize name by removing punctuation and extra spaces.
            This allows matching despite formatting differences like commas, periods, etc.
            Example: "Engineered Profiles, LLC" -> "engineered profiles llc"
            """
            import re

            # Remove common punctuation: commas, periods, hyphens, apostrophes, quotes
            normalized = re.sub(r"[,.\-\'\"()]", "", name)
            # Normalize whitespace and convert to lowercase
            normalized = " ".join(normalized.strip().lower().split())
            return normalized

        # Normalize names for comparison (case-insensitive, remove punctuation and extra spaces)
        sf_name_normalized = normalize_name_for_comparison(salesforce_account_name)
        loa_name_normalized = normalize_name_for_comparison(loa_customer_name)

        # Exact match (after normalization)
        if sf_name_normalized == loa_name_normalized:
            return {
                "match": True,
                "match_type": "exact_normalized",
                "salesforce_name": salesforce_account_name,
                "loa_name": loa_customer_name,
            }

        # Check if one name contains the other (partial match)
        if (
            sf_name_normalized in loa_name_normalized
            or loa_name_normalized in sf_name_normalized
        ):
            return {
                "match": True,
                "match_type": "partial_normalized",
                "salesforce_name": salesforce_account_name,
                "loa_name": loa_customer_name,
            }

        # No match
        self.logger.warning("Account names do NOT match")
        return {
            "match": False,
            "reason": f"Account name mismatch: Salesforce has '{salesforce_account_name}' but LOA shows '{loa_customer_name}'",
            "salesforce_name": salesforce_account_name,
            "loa_name": loa_customer_name,
        }

    def extract_account_numbers_from_azure_ocr(
        self, extraction_log: Dict, udc: str
    ) -> Dict:
        """
        Extract account numbers from Azure Document Intelligence OCR results.
        More accurate than GPT-4o Vision for structured numeric data.
        Uses regex patterns to find account numbers in the OCR text.

        Args:
            extraction_log (Dict): The extraction log containing Azure DI OCR results.
            udc (str): The utility distribution company code.

        Returns:
            Dict: {
                "success": True/False,
                "account_numbers": ["num1", "num2", ...],
                "account_count": int,
                "method": "azure_document_intelligence",
                "udc": str
            }
        """
        self.logger.info(
            f"Extracting account numbers from Azure Document Intelligence OCR for {udc}..."
        )

        # Get the full OCR text (try both field names for compatibility)
        ocr_text = extraction_log.get("full_text", "") or extraction_log.get(
            "extracted_text", ""
        )

        if not ocr_text:
            self.logger.warning(
                "No OCR text found in extraction_log (checked 'full_text' and 'extracted_text' fields)"
            )
            return {
                "success": False,
                "account_numbers": [],
                "account_count": 0,
                "method": "azure_document_intelligence",
                "udc": udc,
            }

        # Get pattern for UDC (default to 8+ digits for maximum flexibility)
        pattern = self.ACCOUNT_NUMBER_PATTERNS.get(udc, r"\b\d{8,}\b")

        # Extract account numbers using regex
        account_numbers = re.findall(pattern, ocr_text)

        # For Cinergy/Duke, also try to find accounts with Z or 2 in the middle
        # This ensures we capture both formats: with Z and without Z (or with 2 instead)
        if udc in ["Duke", "Cinergy"]:
            # Additional pattern: look for any long digit sequence (might have Z/2 embedded)
            additional_pattern = r"\b\d{20,25}\b"
            additional_accounts = re.findall(additional_pattern, ocr_text)
            account_numbers.extend(additional_accounts)

            # Also look for patterns with Z explicitly
            z_pattern = r"\b\d{10,15}Z\d{8,12}\b"
            z_accounts = re.findall(z_pattern, ocr_text)
            account_numbers.extend(z_accounts)

        # Deduplicate while preserving order
        unique_accounts = list(dict.fromkeys(account_numbers))

        return {
            "success": True,
            "account_numbers": unique_accounts,
            "account_count": len(unique_accounts),
            "method": "azure_document_intelligence",
            "udc": udc,
        }

    def _validate_aep_account_format(self, account: str) -> tuple[bool, str, str]:
        """
        Validate a single AEP account number format.

        Helper function to validate AEP account formats:
        1. Standalone: 16-18 digits (e.g., "00140060748972843")
        2. Slash format: 10-12 digits / 16-18 digits (e.g., "12345678901/12345678901234567")

        Args:
            account: Account number string to validate

        Returns:
            tuple: (is_valid, account_str, reason)
                - is_valid: True if format is valid
                - account_str: Cleaned account string
                - reason: Validation failure reason (empty string if valid)
        """
        acc_str = str(account).strip()

        if "/" in acc_str:
            # Slash format validation: 10-12 / 16-18 digits
            parts = acc_str.split("/")
            if len(parts) == 2:
                part1_digits = re.sub(r"[^0-9]", "", parts[0])
                part2_digits = re.sub(r"[^0-9]", "", parts[1])

                if 10 <= len(part1_digits) <= 12 and 16 <= len(part2_digits) <= 18:
                    return (True, acc_str, "")
                else:
                    return (
                        False,
                        acc_str,
                        f"({len(part1_digits)}/{len(part2_digits)} digits, expected 10-12/16-18)",
                    )
            else:
                return (False, acc_str, "(multiple slashes)")
        else:
            # Standalone format validation: 16-18 digits
            digits_only = re.sub(r"[^0-9]", "", acc_str)

            if 16 <= len(digits_only) <= 18:
                return (True, acc_str, "")
            else:
                return (
                    False,
                    acc_str,
                    f"({len(digits_only)} digits, expected 16-18)",
                )

    def extract_aep_accounts_from_azure_ocr(self, extraction_log: Dict) -> Dict:
        """
        Extract and validate AEP account numbers from Azure Document Intelligence OCR results.

        This method extracts account numbers from the OCR text and validates their format
        using the helper function _validate_aep_account_format().

        AEP Account Formats:
        - Standalone: 17 digits (OCR tolerance: 16-18 digits)
        - Slash format: 11/17 (OCR tolerance: 10-12 / 16-18 digits)

        Args:
            extraction_log (Dict): The extraction log containing Azure DI OCR results.

        Returns:
            Dict: {
                "success": True/False,
                "account_numbers": ["all extracted accounts"],
                "valid_accounts": ["accounts that pass format validation"],
                "invalid_accounts": ["account (reason)", ...],
                "account_count": int (count of valid accounts),
                "format_validation_passed": True/False (all accounts valid),
                "method": "azure_document_intelligence"
            }
        """
        # Get OCR text
        ocr_text = extraction_log.get("extracted_text", "")
        if not ocr_text:
            return {
                "success": False,
                "account_numbers": [],
                "valid_accounts": [],
                "invalid_accounts": [],
                "account_count": 0,
                "format_validation_passed": False,
                "method": "azure_document_intelligence",
            }

        # Extract account numbers using regex pattern
        pattern = self.ACCOUNT_NUMBER_PATTERNS.get("AEP", r"\b\d{11,25}\b")
        raw_accounts = re.findall(pattern, ocr_text)

        # Also look for slash format accounts
        slash_pattern = r"\b\d{10,12}/\d{16,18}\b"
        slash_accounts = re.findall(slash_pattern, ocr_text)
        raw_accounts.extend(slash_accounts)

        # Deduplicate
        unique_accounts = list(dict.fromkeys(raw_accounts))

        if not unique_accounts:
            self.logger.info("No account numbers found in Azure OCR text")
            return {
                "success": True,
                "account_numbers": [],
                "valid_accounts": [],
                "invalid_accounts": [],
                "account_count": 0,
                "format_validation_passed": True,  # No accounts = no format issues
                "method": "azure_document_intelligence",
            }

        # Validate each account format using helper function
        valid_accounts = []
        invalid_accounts = []

        for acc in unique_accounts:
            is_valid, acc_str, reason = self._validate_aep_account_format(acc)

            if is_valid:
                valid_accounts.append(acc_str)
            else:
                invalid_accounts.append(f"{acc_str} {reason}")

        format_validation_passed = len(invalid_accounts) == 0

        return {
            "success": True,
            "account_numbers": unique_accounts,
            "valid_accounts": valid_accounts,
            "invalid_accounts": invalid_accounts,
            "account_count": len(valid_accounts),
            "format_validation_passed": format_validation_passed,
            "method": "azure_document_intelligence",
        }

    @aggressive_retry(max_attempts=50, initial_delay=2.0, max_delay=60.0)
    def extract_account_numbers_from_great_lakes_loa(
        self, pdf_path: str, udc: str, extraction_log: Dict = None
    ) -> Dict:
        """
        Extract ALL account numbers from ALL pages of a Great Lakes UDC LOA document (including attachments).

        This method scans the entire document to find account numbers in the appropriate field for each UDC:
        - AEP, FirstEnergy, Dayton, Duke/Cinergy: "Account/SDI Number" field
        - ComEd, Ameren: "Account Numbers" field

        Args:
            pdf_path (str): Path to the PDF file.
            udc (str): The utility distribution company code (e.g., "CEI", "AEP", "ComEd").

        Returns:
            Dict: {
                "success": True/False,
                "account_numbers": ["num1", "num2", ...],
                "account_count": int,
                "total_pages_scanned": int,
                "pages_with_accounts": [1, 2, ...],
                "udc": str
            }
        """
        self.logger.info(f"Extracting account numbers from Great Lakes {udc} LOA...")

        # TIER 1: Try Azure Document Intelligence first (most accurate)
        if extraction_log:
            self.logger.info(
                "  - Attempting extraction from Azure Document Intelligence OCR..."
            )
            azure_result = self.extract_account_numbers_from_azure_ocr(
                extraction_log, udc
            )

            if azure_result.get("success") and azure_result.get("account_numbers"):
                self.logger.info(
                    f"Azure DI found {len(azure_result['account_numbers'])} account number(s)"
                )
                return azure_result
            else:
                self.logger.info(
                    "  - No account numbers found in Azure DI OCR, falling back to GPT-4o Vision"
                )

        # TIER 2: Fall back to GPT-4o Vision (for complex cases or attachments)
        # Extract ALL pages as images
        all_pages = self._extract_all_pdf_pages_as_images(pdf_path)
        if not all_pages:
            self.logger.error(
                "Failed to extract any images from PDF for account number extraction."
            )
            return {
                "success": False,
                "account_numbers": [],
                "account_count": 0,
                "total_pages_scanned": 0,
                "pages_with_accounts": [],
                "method": "gpt4o_vision",
                "udc": udc,
            }

        # Determine field name based on UDC
        if udc in self.OHIO_UDCS:
            field_name = "Account/SDI Number"
        elif udc in ["ComEd", "Ameren"]:
            field_name = "Account Numbers"
        else:
            # Default to Account/SDI Number for unknown UDCs
            field_name = "Account/SDI Number"

        # Collect all account numbers from all pages
        all_account_numbers = []
        pages_with_accounts = []

        for page_num, (page_index, base64_image) in enumerate(all_pages, start=1):

            prompt = f"""Extract ALL account numbers from this page of a {udc} LOA document.

    CRITICAL INSTRUCTIONS:
    1. Look for the "{field_name}" field or section
    2. Extract ALL account numbers you find (can be multiple numbers)
    3. Account numbers may be in various formats:
       - Plain digits: "12345678901234567"
       - With spaces: "123 456 789 012 345 67"
       - With slashes: "123456789012/12345678901234567890"
       - With dashes: "123-456-789-012-345-67"
       - With Z separator: "910117129533Z109008636"
    4. Remove spaces and dashes, but KEEP slashes and letters (like Z)
    5. If you see "See attached" or "Attached" or "See below", still extract any visible numbers
    6. Look in tables, lists, or attachment sections
    7. Return EMPTY array if NO account numbers found on this page

    **CRITICAL - DIGIT RECOGNITION ACCURACY:**
    Pay EXTRA attention to these commonly confused digits:
    - **0 (zero) vs O (letter O)**: Zero is rounder, O may have serifs
    - **1 (one) vs l (lowercase L) vs I (uppercase i)**: Look for serifs and context
    - **6 vs 8**: 6 has ONE loop (top closed, bottom open), 8 has TWO loops
    - **6 vs 9**: 6 opens at TOP, 9 opens at BOTTOM
    - **6 vs 0**: 6 has a tail/stem, 0 is fully closed
    - **5 vs S**: 5 is a digit, S is a letter
    - **2 vs Z**: 2 is a digit, Z is a letter
    - **7 vs 1**: 7 has a horizontal top, 1 is vertical

    **VERIFICATION STEP - DOUBLE-CHECK EACH DIGIT:**
    After extracting each account number, verify:
    - Does each digit look correct in context?
    - Are there any ambiguous or unclear digits?
    - Does the number match the expected format/length?

    **CRITICAL FOR CINERGY/DUKE ENERGY ACCOUNTS:**
    - Cinergy/Duke Energy account numbers are 23 characters long
    - They contain the LETTER 'Z' at position 13 (not the digit '2')
    - Format: 910XXXXXXXXX**Z**XXXXXXXXX (where X = digits)
    - Example: 910117129533Z109008636
    - DO NOT confuse the letter 'Z' with the digit '2'
    - If you see what looks like a '2' in the middle of a 22-23 character account number, it's likely the letter 'Z'
    - PRESERVE the letter 'Z' exactly as shown

    Return JSON:
    {{
      "account_numbers": ["num1", "num2", ...],
      "found_on_page": true/false,
      "notes": "any relevant observations"
    }}"""

            user_prompt = f"""[IMAGE_ANALYSIS_REQUEST]
    {prompt}

    [ATTACHED_IMAGE]
    data:image/png;base64,{base64_image}"""

            system_prompt = f"You are an expert at extracting account numbers from {udc} LOA documents. Extract ALL account numbers exactly as shown."

            try:
                gpt_response_data = self.openai_4o_service.process_with_prompts(
                    system_prompt, user_prompt, max_token=1500, raw_response=True
                )

                if gpt_response_data and gpt_response_data[0]["ai_result"]:
                    analysis_text = gpt_response_data[0]["ai_result"][0]["result"]

                    # Parse JSON response
                    json_match = re.search(
                        r"```json\s*(.*?)\s*```", analysis_text, re.DOTALL
                    )
                    if json_match:
                        analysis_json = json.loads(json_match.group(1))
                    else:
                        analysis_json = json.loads(analysis_text)

                    page_accounts = analysis_json.get("account_numbers", [])
                    if page_accounts:
                        self.logger.info(
                            f"    Found {len(page_accounts)} account number(s) on page {page_num}"
                        )
                        all_account_numbers.extend(page_accounts)
                        pages_with_accounts.append(page_num)
                    else:
                        self.logger.info(
                            f"    No account numbers found on page {page_num}"
                        )

            except Exception as e:
                self.logger.error(
                    f"Error extracting account numbers from page {page_num}: {str(e)}"
                )
                continue

        # Deduplicate account numbers (keep unique ones)
        unique_accounts = list(set(all_account_numbers))

        return {
            "success": True,
            "account_numbers": unique_accounts,
            "account_count": len(unique_accounts),
            "total_pages_scanned": len(all_pages),
            "pages_with_accounts": pages_with_accounts,
            "method": "gpt4o_vision",
            "udc": udc,
        }

    def compare_account_numbers_exact(
        self, salesforce_accounts: str, loa_accounts: List[str]
    ) -> Dict:
        """
        Compare account numbers with EXACT matching - NO tolerance for differences.
        Rejects if even 1 digit doesn't match.

        Matching Strategy:
        - Requires perfect digit-by-digit match
        - Ignores non-digit characters (slashes, spaces)
        - Rejects if ANY digit differs
        - **SPECIAL HANDLING for Cinergy/Duke Energy**: Letter 'Z' and digit '2' at position 13
          are treated as interchangeable (OCR often misreads 'Z' as '2')

            Args:
                salesforce_accounts (str): Comma-separated account numbers from Salesforce.
                loa_accounts (List[str]): List of account numbers extracted from LOA.

            Returns:
                Dict: {
                    "match": True/False,
                    "reason": str (if no match),
                    "salesforce_accounts": [list],
                    "loa_accounts": [list],
                    "matched_accounts": [{"salesforce": str, "loa": str}],
                    "unmatched_sf_accounts": [list],
                    "unmatched_loa_accounts": [list]
                }
        """
        self.logger.info(
            "Comparing account numbers with EXACT matching (no tolerance)..."
        )

        # Parse Salesforce accounts (comma-separated)
        if not salesforce_accounts:
            return {
                "match": False,
                "reason": "No Salesforce account numbers provided",
                "salesforce_accounts": [],
                "loa_accounts": loa_accounts,
                "matched_accounts": [],
                "unmatched_sf_accounts": [],
                "unmatched_loa_accounts": loa_accounts,
            }

        sf_accounts_list = [
            acc.strip() for acc in salesforce_accounts.split(",") if acc.strip()
        ]

        if not loa_accounts:
            return {
                "match": False,
                "reason": f"No account numbers found in LOA - expected {len(sf_accounts_list)} account(s)",
                "salesforce_accounts": sf_accounts_list,
                "loa_accounts": [],
                "matched_accounts": [],
                "unmatched_sf_accounts": sf_accounts_list,
                "unmatched_loa_accounts": [],
            }

        self.logger.info(
            f"  - Salesforce accounts ({len(sf_accounts_list)}): {sf_accounts_list}"
        )
        self.logger.info(f"  - LOA accounts ({len(loa_accounts)}): {loa_accounts}")

        # Track matched and unmatched accounts
        matched_accounts = []
        unmatched_sf_accounts = []
        matched_loa_indices = set()

        # For each Salesforce account, find EXACT match in LOA accounts
        for sf_account in sf_accounts_list:
            sf_normalized = self.normalize_account_flexible(sf_account)
            found_match = False

            for loa_index, loa_account in enumerate(loa_accounts):
                if loa_index in matched_loa_indices:
                    continue  # Already matched

                loa_normalized = self.normalize_account_flexible(loa_account)

                # EXACT match required - all digits must match (with Z/2 flexibility for Cinergy/Duke)
                if sf_normalized == loa_normalized:
                    matched_accounts.append(
                        {"salesforce": sf_account, "loa": loa_account}
                    )
                    matched_loa_indices.add(loa_index)
                    found_match = True
                    # Check if this was a Z/2 flexible match for Cinergy/Duke
                    if "Z" in sf_account.upper() or "Z" in loa_account.upper():
                        self.logger.info(
                            f"Match (Z/2 flexible): SF '{sf_account}' = LOA '{loa_account}'"
                        )
                    else:
                        self.logger.info(
                            f"Exact match: SF '{sf_account}' = LOA '{loa_account}'"
                        )
                    break

            if not found_match:
                unmatched_sf_accounts.append(sf_account)
                self.logger.warning(f"No exact match for SF '{sf_account}'")

        # Find unmatched LOA accounts
        unmatched_loa_accounts = [
            loa_accounts[i]
            for i in range(len(loa_accounts))
            if i not in matched_loa_indices
        ]

        # Determine overall match status
        all_matched = len(unmatched_sf_accounts) == 0

        if all_matched:
            return {
                "match": True,
                "salesforce_accounts": sf_accounts_list,
                "loa_accounts": loa_accounts,
                "matched_accounts": matched_accounts,
                "unmatched_sf_accounts": [],
                "unmatched_loa_accounts": unmatched_loa_accounts,
            }
        else:
            reason = f"Account number mismatch: {len(unmatched_sf_accounts)} Salesforce account(s) not found in LOA: {unmatched_sf_accounts}"
            return {
                "match": False,
                "reason": reason,
                "salesforce_accounts": sf_accounts_list,
                "loa_accounts": loa_accounts,
                "matched_accounts": matched_accounts,
                "unmatched_sf_accounts": unmatched_sf_accounts,
                "unmatched_loa_accounts": unmatched_loa_accounts,
            }
