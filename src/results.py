class ResultsManager:
    """
    Manages benchmark results storage and processing.
    
    This class handles saving raw API responses, parsing responses,
    and generating CSV output files.
    """
    
    @staticmethod
    def save_raw_response(response: Any, output_path: str) -> None:
        """
        Save a raw API response to a file.
        
        Args:
            response: The API response object
            output_path: Path to save the response
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                if isinstance(response, (dict, list)):
                    json.dump(response, f, indent=2)
                else:
                    f.write(str(response))
                    
            logger.debug(f"Saved raw response to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving raw response to {output_path}: {str(e)}")
    
    @staticmethod
    def save_results_to_csv(
        results: List[Dict[str, Any]], 
        output_path: str,
        headers: List[str] = CSV_HEADERS
    ) -> None:
        """
        Save benchmark results to a CSV file.
        
        Args:
            results: List of result dictionaries
            output_path: Path to save the CSV file
            headers: List of column headers
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                csv_writer = csv.writer(f)
                
                # Write header row
                csv_writer.writerow(headers)
                
                # Write data rows
                for result in results:
                    # Ensure all headers are present in the result
                    row = [result.get(header, "") for header in headers]
                    csv_writer.writerow(row)
                    
            logger.info(f"Saved {len(results)} results to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results to CSV: {str(e)}")
    
    @staticmethod
    def parse_json_response(
        response_text: str, 
        default_values: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Parse a JSON response with robust error handling.
        
        Args:
            response_text: JSON response text
            default_values: Default values to use for missing fields
            
        Returns:
            Parsed JSON dictionary
        """
        if default_values is None:
            default_values = {
                "punktzahl": -999,
                "staerken": "Error parsing response",
                "schwaechen": "Error parsing response",
                "begruendung": "Error parsing response"
            }
        
        try:
            # Try to parse the JSON response
            response_json = json.loads(response_text)
            
            # Validate expected fields
            for key in default_values:
                if key not in response_json:
                    logger.warning(f"Missing field in JSON response: {key}")
                    response_json[key] = default_values[key]
                    
            return response_json
            
        except json.JSONDecodeError as e:
            # Try to recover partial JSON
            logger.error(f"JSON decode error: {str(e)}")
            
            # Check if it's a Claude response with "{" prefix
            if response_text.startswith("{") and not response_text.endswith("}"):
                try:
                    # Try to add the closing brace
                    response_text += "}"
                    response_json = json.loads(response_text)
                    
                    # Add missing fields
                    for key in default_values:
                        if key not in response_json:
                            response_json[key] = default_values[key]
                            
                    logger.info("Successfully recovered partial JSON response")
                    return response_json
                    
                except json.JSONDecodeError:
                    pass
            
            # Return default values if recovery failed
            logger.error(f"Failed to parse JSON response: {response_text[:100]}...")
            return default_values.copy()