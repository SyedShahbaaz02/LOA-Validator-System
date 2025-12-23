import json
import logging
import os
import tempfile
from io import BytesIO

from intelligentflow.business_logic.base_workflow_openai_service import (
    BaseWorkflowOpenAIService,
)
from intelligentflow.infrastructure.configuration_service import (
    ConfigurationService,
    ConfigurationType,
)
from intelligentflow.models.shared.workflow_run import WorkflowRun

from ..loa.enhanced_loa_validator import EnhancedLOAValidator


class LOAWorkflowOpenAIService(BaseWorkflowOpenAIService):
    """LOA workflow service for processing Letter of Authorization documents.

    Args:
        openai_4o_service: The OpenAI service for GPT-4o processing
        blob_storage_service: Service for Azure blob storage operations
        configuration_service: Service for workflow configuration management
    """

    def __init__(self, openai_4o_service, blob_storage_service, configuration_service):
        # Create an empty configuration service for LOA workflow
        # LOA uses its own prompt management through EnhancedLOAValidator
        configuration_service = ConfigurationService(
            configuration_type=ConfigurationType.CUSTOM_CONFIGURATION,
            custom_configuration={},
        )
        super().__init__(openai_4o_service, blob_storage_service, configuration_service)
        # Will be initialized in process_workflow with proper region/UDC parameters
        self.loa_validator = None

    def get_openai_output_file_name(self, workflow_run_id, file_name):
        formatted_file_name = self.convert_to_snake_case(file_name)
        return f"{workflow_run_id}/openai_4o_output_{formatted_file_name}.json"

    def process_workflow(
        self, workflow_run_id: str, request_details: WorkflowRun
    ) -> str:
        """Process LOA workflow using OCR result object directly from queue handler.

        This method receives the OCR result object from the queue handler and processes it
        using the enhanced validator with all advanced layout analysis capabilities.
        It extracts caseRegion and udc parameters from the request to avoid OCR-based detection.
        """

        # Extract caseRegion, udc, caseIntervalNeeded, and accountName parameters from request_details if available
        case_region = None
        udc = None
        interval_needed = True  # Default to True for backward compatibility
        account_name = None  # For account name comparison
        service_location_ldc = None  # For account number comparison

        # Add logging to debug parameter extraction
        logging.info(f"Processing workflow {workflow_run_id}")
        logging.info(f"Request details type: {type(request_details)}")

        try:
            # Handle both dict (from Cosmos DB) and WorkflowRun object cases
            if isinstance(request_details, dict):
                # Direct dict access (from Cosmos DB)
                logging.info("Request details is a dict (from Cosmos DB)")
                if (
                    "request" in request_details
                    and "additional_metadata" in request_details["request"]
                ):
                    additional_metadata = request_details["request"][
                        "additional_metadata"
                    ]
                    logging.info(f"Additional metadata: {additional_metadata}")

                    # Removed JSON string parsing code as requested

                    if isinstance(additional_metadata, dict):
                        case_region = additional_metadata.get("caseRegion")
                        udc = additional_metadata.get("udc")
                        # Extract caseIntervalNeeded parameter (default to True if not present)
                        interval_needed = additional_metadata.get(
                            "caseIntervalNeeded", True
                        )
                        # Extract accountName for account name comparison
                        account_name = additional_metadata.get("accountName")
                        # Extract serviceLocationLDC for account number comparison
                        service_location_ldc = additional_metadata.get(
                            "serviceLocationLDC"
                        )
                        logging.info(
                            f"Extracted from dict - caseRegion: '{case_region}', udc: '{udc}', caseIntervalNeeded: {interval_needed}, accountName: '{account_name}', serviceLocationLDC: '{service_location_ldc}'"
                        )
                else:
                    logging.warning(
                        f"No request.additional_metadata in dict. Keys: {request_details.keys() if request_details else 'None'}"
                    )
            else:
                # WorkflowRun object access
                logging.info("Request details is a WorkflowRun object")
                if (
                    request_details
                    and request_details.request
                    and request_details.request.additional_metadata
                ):
                    additional_metadata = request_details.request.additional_metadata
                    logging.info(
                        f"Additional metadata type: {type(additional_metadata)}"
                    )
                    logging.info(f"Additional metadata content: {additional_metadata}")

                    # additional_metadata could be a dict or an object, handle both cases
                    if isinstance(additional_metadata, dict):
                        case_region = additional_metadata.get("caseRegion")
                        udc = additional_metadata.get("udc")
                        # Extract caseIntervalNeeded parameter (default to True if not present)
                        interval_needed = additional_metadata.get(
                            "caseIntervalNeeded", True
                        )
                        # Extract accountName for account name comparison
                        account_name = additional_metadata.get("accountName")
                        # Extract serviceLocationLDC for account number comparison
                        service_location_ldc = additional_metadata.get(
                            "serviceLocationLDC"
                        )
                        logging.info(
                            f"Extracted from dict - caseRegion: '{case_region}', udc: '{udc}', caseIntervalNeeded: {interval_needed}, accountName: '{account_name}', serviceLocationLDC: '{service_location_ldc}'"
                        )
                    else:
                        # If it's an object, try to access as attributes
                        case_region = getattr(additional_metadata, "caseRegion", None)
                        udc = getattr(additional_metadata, "udc", None)
                        # Extract caseIntervalNeeded parameter (default to True if not present)
                        interval_needed = getattr(
                            additional_metadata, "caseIntervalNeeded", True
                        )
                        # Extract accountName for account name comparison
                        account_name = getattr(additional_metadata, "accountName", None)
                        # Extract serviceLocationLDC for account number comparison
                        service_location_ldc = getattr(
                            additional_metadata, "serviceLocationLDC", None
                        )
                        logging.info(
                            f"Extracted from object - caseRegion: '{case_region}', udc: '{udc}', caseIntervalNeeded: {interval_needed}, accountName: '{account_name}', serviceLocationLDC: '{service_location_ldc}'"
                        )
                else:
                    logging.warning(
                        "No additional_metadata found in WorkflowRun object"
                    )
        except Exception as e:
            # Continue with default behavior if parameter extraction fails
            logging.error(f"Error extracting parameters: {e}")
            pass

        # Create enhanced validator with region and UDC information
        # Map caseRegion to validator region format (handle various formats)
        region_mapping = {
            "Great Lakes": "Great Lakes",
            "Great Lakes Region": "Great Lakes",
            "GreatLakes": "Great Lakes",
            "GreatLakesRegion": "Great Lakes",
            "GLR": "Great Lakes",
            "New England": "New England",
            "New England Region": "New England",
            "NewEngland": "New England",
            "NewEnglandRegion": "New England",
            "NE": "New England",
        }

        # CRITICAL DEBUG: Add extensive logging to debug region issues
        logging.warning(
            f"DEBUG: Raw case_region before mapping: '{case_region}' (type: {type(case_region)})"
        )
        if case_region:
            logging.warning(
                f"DEBUG: Exact case_region value (lowercase): '{case_region.lower()}'"
            )
            logging.warning(
                f"DEBUG: Is 'new england' in lowercase case_region? {case_region.lower() == 'new england'}"
            )
            logging.warning(
                f"DEBUG: Is case_region in region_mapping? {case_region in region_mapping}"
            )

        mapped_region = (
            region_mapping.get(case_region, case_region)
            if case_region
            else "Great Lakes"
        )

        # Force New England for testing if needed
        if case_region and "england" in str(case_region).lower():
            logging.warning(
                "DEBUG: Force setting New England region due to 'england' in case_region"
            )
            mapped_region = "New England"

        # Log the final region being used
        logging.warning(
            f"DEBUG: Final region mapping: '{case_region}' -> '{mapped_region}'"
        )
        logging.warning(
            f"DEBUG: Creating validator with region='{mapped_region}', udc='{udc}', interval_needed={interval_needed}"
        )

        # Convert interval_needed to bool to ensure proper type
        interval_needed_bool = bool(interval_needed)
        if isinstance(interval_needed, str):
            # Handle string 'true'/'false' values
            interval_needed_bool = interval_needed.lower() == "true"

        # Create validator with the specified region, UDC, interval_needed, account_name, and service_location_ldc (or defaults)
        self.loa_validator = EnhancedLOAValidator(
            self.openai_4o_service,
            region=mapped_region,
            udc=udc,
            interval_needed=interval_needed_bool,
            account_name=account_name,
            service_location_ldc=service_location_ldc,
        )

        doc_intelligence_output_folder_name = (
            f"{workflow_run_id}/document_intelligence_output/"
        )

        response = []

        for index, file_path in enumerate(
            self.blob_storage_service.get_file_names_in_folder(
                doc_intelligence_output_folder_name
            )
        ):
            try:

                file_name = (file_path.split("/")[-1]).split(".")[-2]

                ocr_result = json.load(
                    BytesIO(self.blob_storage_service.get_blob(file_path))
                )

                if not (
                    ocr_result
                    and "analyzeResult" in ocr_result
                    and "content" in ocr_result["analyzeResult"]
                ):
                    continue

                extraction_log = self.loa_validator.extract_layout_from_ocr_result(
                    ocr_result["analyzeResult"], workflow_run_id
                )

                # Add provided UDC to detected utilities to prioritize it
                # NOTE: We no longer need to set extraction_log['provided_udc'] since the UDC
                # is passed directly to the EnhancedLOAValidator constructor
                if udc:
                    if "detected_utilities" not in extraction_log:
                        extraction_log["detected_utilities"] = []
                    # Insert provided UDC at the beginning to make it the primary utility
                    extraction_log["detected_utilities"].insert(
                        0,
                        {
                            "name": udc,
                            "mentions": 999,  # High mention count to prioritize it
                            "detection_method": "provided_parameter",
                        },
                    )

                # Storing the extraction log for debugging
                self.blob_storage_service.upload_blob(
                    f"{workflow_run_id}/extraction_logs/{file_name}_extraction_log.json",
                    json.dumps(extraction_log, indent=2),
                )

                # DETAILED LOGGING FOR DEBUGGING
                try:
                    logging.warning(f"DETAILED OCR LOG FOR {file_name}:")
                    logging.warning(json.dumps(ocr_result, indent=2))
                except TypeError as e:
                    logging.warning(
                        f"Failed to serialize OCR result for {file_name}: {e}"
                    )

                # Download the original PDF for vision-based validation
                pdf_path = None
                temp_pdf_file = None
                try:
                    # Find the original PDF in the input_files folder
                    input_folder = f"{workflow_run_id}/input_files/"
                    pdf_files = [
                        f
                        for f in self.blob_storage_service.get_file_names_in_folder(
                            input_folder
                        )
                        if f.lower().endswith(".pdf")
                    ]

                    if pdf_files:
                        # Find the PDF that matches the current OCR file name
                        original_pdf_name = file_name.replace("_layout", "").replace(
                            "_ocr_result", ""
                        )
                        pdf_blob_path = next(
                            (p for p in pdf_files if original_pdf_name in p), None
                        )

                        if pdf_blob_path:
                            pdf_bytes = self.blob_storage_service.get_blob(
                                pdf_blob_path
                            )

                            # Save to a temporary file
                            temp_pdf_file = tempfile.NamedTemporaryFile(
                                delete=False, suffix=".pdf"
                            )
                            temp_pdf_file.write(pdf_bytes)
                            pdf_path = temp_pdf_file.name
                            temp_pdf_file.close()

                    # Use the universal utility recognition validation method
                    validation_result = (
                        self.loa_validator.validate_with_universal_utility_recognition(
                            extraction_log, workflow_run_id, pdf_path=pdf_path
                        )
                    )

                finally:
                    # Clean up the temporary file
                    if temp_pdf_file and os.path.exists(temp_pdf_file.name):
                        os.remove(temp_pdf_file.name)

                # Create simplified JSON response (only status and rejection reasons if rejected)
                simplified_response = {
                    "file_index": index,
                    "file_name": validation_result.get("file_name", file_name),
                    "status": validation_result.get("validation_status", "ERROR"),
                    "expiration_details": validation_result.get(
                        "expiration_details", None
                    ),
                }

                # Add rejection reasons only if document is rejected
                if validation_result.get("validation_status") == "REJECT":
                    simplified_response["rejection_reasons"] = validation_result.get(
                        "all_rejection_reasons", []
                    )

                # Save full response to blob storage for audit
                full_response_str = json.dumps(
                    validation_result, indent=2, sort_keys=True
                )

                # DETAILED LOGGING FOR DEBUGGING
                logging.warning(f"DETAILED VALIDATION RESULT FOR {file_name}:")
                logging.warning(full_response_str)

                openai_output_file_name = self.get_openai_output_file_name(
                    workflow_run_id, f"{file_name}_{index}"
                )

                self.blob_storage_service.upload_blob(
                    openai_output_file_name, full_response_str
                )

                response.append(simplified_response)

            except Exception as e:
                error_response = {
                    "file_index": index,
                    "file_name": file_name,
                    "status": "ERROR",
                    "error": str(e),
                }
                response.append(error_response)

        # Return simplified JSON response as requested
        return json.dumps(response, indent=2)
