"""
AI Service Adapters

This module contains classes for interacting with different AI services 
(OpenAI, Claude, Mistral) to evaluate text.
"""

import logging
import os
import time
import json

import uuid
from datetime import datetime

from typing import Dict, List, Any, Optional, Tuple

import random

## Import AI API clients
try:
    from openai import OpenAI
    from anthropic import Anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    from mistralai import Mistral
    IMPORTS_SUCCESSFUL = True
except ImportError:
    IMPORTS_SUCCESSFUL = False
    print("Warning: Some AI API libraries are missing. Install with:")
    print("pip install openai anthropic mistralai requests pyyaml")

logger = logging.getLogger("awesome_scoring")

class TextRater:
    """
    Base class for text rating services.
    
    This class provides a common interface for evaluating text
    using different AI models.
    """
    
    def __init__(self, rater_type: str = "benchmark"):
        """
        Initialize the text rater.
        
        Args:
            rater_type: Type of rater to use (benchmark, simple, etc.)
        """
        self.rater_type = rater_type
        logger.info(f"Initialized TextRater with type: {rater_type}")
        
        ## Check if API libraries are available
        if not IMPORTS_SUCCESSFUL:
            logger.warning("AI API libraries not fully available. Some features may not work.")


class OpenAIRater:
    """OpenAI-based text rater adapter."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI rater.
        
        Args:
            api_key: OpenAI API key (optional, defaults to environment variable)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def rate_text(
        self, 
        text: str, 
        system_prompt: str, 
        user_prompt: str, 
        model: str = "gpt-4o",
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Rate a text using OpenAI's API.
        
        Args:
            text: Text to rate
            system_prompt: System prompt for the model
            user_prompt: User prompt template (will be formatted with text)
            model: OpenAI model to use
            temperature: Temperature for generation (optional)
            
        Returns:
            Dictionary with rating results
        """
        specific_user_prompt = user_prompt.format(student_text=text)
        
        ## Build request params
        request_params = {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": specific_user_prompt}
            ]
        }

        ## Add temperature if provided
        if temperature is not None:
            request_params["temperature"] = temperature
        
        # try: 
        response = self.client.chat.completions.create(**request_params)
        response_text = response.choices[0].message.content

        ## Parse JSON response
        # return json.loads(response_text)
        return response_text
        
        # except Exception as e:
        #     logger.error(f"Error rating text with OpenAI: {str(e)}")
        #     return {
        #         "punktzahl": -999,
        #         "staerken": f"Error: {str(e)}",
        #         "schwaechen": "Error processing response", 
        #         "begruendung": "Error processing response"
        #     }


    def rate_text_batch(
        self,
        texts: Dict[str, str],
        system_prompt: str,
        user_prompt: str,
        model: str = "gpt-4o",
        temperature: Optional[float] = None,
        batch_size: int = 50,
        output_dir: str = "./batches",
        poll_interval: int = 60
    ) -> Dict[str, str]:
        """
        Rate multiple texts using OpenAI's batch API.
        
        Args:
            texts: Dictionary mapping student IDs to text content
            system_prompt: System prompt for the model
            user_prompt: User prompt template (will be formatted with text)
            model: OpenAI model to use
            temperature: Temperature for generation (optional)
            batch_size: Maximum batch size for requests
            output_dir: Directory to save batch files
            poll_interval: Seconds between status checks
            
        Returns:
            Dictionary mapping student IDs to rating results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a unique batch ID
        batch_id = f"openai_batch_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        batch_file = os.path.join(output_dir, f"{batch_id}.jsonl")
        
        logger.info(f"Creating OpenAI batch file with {len(texts)} texts")
        
        # Create JSONL batch file
        with open(batch_file, "w", encoding="utf-8") as f:
            for student_id, text in texts.items():
                specific_user_prompt = user_prompt.format(student_text=text)
                
                # Build request for this text
                request = {
                    "custom_id": student_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "response_format": {"type": "json_object"},
                        "max_tokens": 1024,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": specific_user_prompt}
                        ]
                    }
                }
                
                if temperature is not None:
                    request["temperature"] = temperature
                
                # Write to batch file
                f.write(json.dumps(request) + "\n")
        
        logger.info(f"Created batch file at {batch_file}")

        # Submit batch request
        try:
            # Create the batch
            batch_input_file = self.client.files.create(
                file=open(batch_file, "rb"),
                purpose="batch"
            )
            batch_input_file_id = batch_input_file.id

            created_job = self.client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
          
            batch_data_id = created_job.id
            logger.info(f"Submitted OpenAI batch job with ID: {batch_data_id}")
            
            # Poll for completion
            completed = False
            results = {}
            
            while not completed:
                logger.info(f"Checking status of batch {batch_id}...")
                batch_status = self.client.batches.retrieve(batch_data_id)
                status = batch_status.status
                
                if status == "completed":
                    completed = True
                    # Get results
                    results_file = self.client.files.content(batch_status.output_file_id)
                    # Download results
                    output_file = os.path.join(output_dir, f"{batch_id}_results.jsonl")
                    with open(output_file, "w") as f:
                        f.write(results_file.text)
                
                    logger.info(f"Downloaded batch results to {output_file}")
                    
                    # Parse results
                    with open(output_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            result = json.loads(line)
                            student_id = result.get("custom_id", "missing_id")
                            response = result.get("response", {}).get("body", {})
                            response_text = response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                            
                            if student_id:
                                results[student_id] = response_text
                    
                    logger.info(f"Processed {len(results)} results from batch")
                    
                elif status == "failed":
                    logger.error(f"Batch processing failed: {batch_status.error}")
                    break
                
                else:
                    # Sleep before polling again
                    if hasattr(batch_status, 'progress') and batch_status.progress is not None:
                        progress = batch_status.progress * 100
                        logger.info(f"Batch progress: {progress:.1f}% - Waiting...")
                    else:
                        logger.info("Batch in progress - Waiting...")
                    time.sleep(poll_interval)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch with OpenAI: {str(e)}")
            # Return error results for all texts
            return {
                student_id: json.dumps({
                    "punktzahl": -999,
                    "staerken": f"Error: {str(e)}", 
                    "schwaechen": "Error processing batch",
                    "begruendung": "Error processing batch"
                })
                for student_id in texts
            }


class ClaudeRater:
    """Claude-based text rater adapter."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Claude rater.
        
        Args:
            api_key: Anthropic API key (optional, defaults to environment variable)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        self.client = Anthropic(api_key=self.api_key)
    
    def rate_text(
        self, 
        text: str, 
        system_prompt: str, 
        user_prompt: str, 
        model: str = "claude-3-opus-20240229",
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Rate a text using Claude's API.
        
        Args:
            text: Text to rate
            system_prompt: System prompt for the model
            user_prompt: User prompt template (will be formatted with text)
            model: Claude model to use
            temperature: Temperature for generation (optional)
            
        Returns:
            Dictionary with rating results
        """
        specific_user_prompt = user_prompt.format(student_text=text)
        
        ## Build request params
        request_params = {
            "model": model,
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": specific_user_prompt},
                {"role": "assistant", "content": "{"}
            ]
        }
        
        ## Add temperature if provided
        if temperature is not None:
            request_params["temperature"] = temperature
        
        try:
            response = self.client.messages.create(**request_params)
            response_text = response.content[0].text
            
            ## Parse JSON response
            if response_text.startswith("{"):
                # return json.loads(response_text)
                return response_text
            else:
                # return json.loads("{" + response_text)  ## Claude might omit the opening brace
                return "{" + response_text
        
        except Exception as e:
            logger.error(f"Error rating text with Claude: {str(e)}")
            return {
                "punktzahl": -999,
                "staerken": f"Error: {str(e)}",
                "schwaechen": "Error processing response",
                "begruendung": "Error processing response"
            }


    def rate_text_batch(
        self,
        texts: Dict[str, str],
        system_prompt: str,
        user_prompt: str,
        model: str = "claude-3-7-sonnet-20250219",
        temperature: Optional[float] = None,
        batch_size: int = 50,
        output_dir: str = "./batches",
        poll_interval: int = 60
    ) -> Dict[str, str]:
        """
        Rate multiple texts using Anthropic's batch API.
        
        Args:
            texts: Dictionary mapping student IDs to text content
            system_prompt: System prompt for the model
            user_prompt: User prompt template (will be formatted with text)
            model: Claude model to use
            temperature: Temperature for generation (optional)
            batch_size: Maximum batch size for requests
            output_dir: Directory to save batch files
            poll_interval: Seconds between status checks
            
        Returns:
            Dictionary mapping student IDs to rating results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a unique batch ID
        batch_id = f"claude_batch_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        batch_file = os.path.join(output_dir, f"{batch_id}.jsonl")
        
        logger.info(f"Creating Claude batch file with {len(texts)} texts")
        
        # Create JSONL batch file
        with open(batch_file, "w", encoding="utf-8") as f:
            for student_id, text in texts.items():
                specific_user_prompt = user_prompt.format(student_text=text)
                
                # Build request for this text
                request = {
                    "custom_id": student_id,
                    "params": {
                        #"response_format": {"type": "json_object"},
                        "model": model,
                        "max_tokens": 1024,
                        "system": [{
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"}
                        }],
                        "messages": [
                            {"role": "user", "content": specific_user_prompt},
                            {"role": "assistant", "content": "{"}
                        ]
                        # "metadata": {"student_id": student_id}
                    }
                }
                
                if temperature is not None:
                    request["params"]["temperature"] = temperature
                
                # Write to batch file
                f.write(json.dumps(request) + "\n")
        
        logger.info(f"Created batch file at {batch_file}")
        
        # Submit batch request
        try:
            # The batch API requires a list of requests
            with open(batch_file, 'rb') as f:               
                requests = [Request(json.loads(x.decode("UTF-8"))) for x in f.readlines()]
                batch_response = self.client.messages.batches.create(
                    requests = requests
                )
            
            batch_id = batch_response.id
            logger.info(f"Submitted Claude batch job with ID: {batch_id}")
            
            # Poll for completion
            completed = False
            results = {}
            
            while not completed:
                logger.info(f"Checking status of batch {batch_id}...")
                batch_status = self.client.messages.batches.retrieve(batch_id)

                status = batch_status.processing_status
                logger.info(f"Batch status: {status}")
                
                if status == "ended":
                    completed = True

                    logger.info("Claude batch job ended.")
                    
                    # Stream results file in memory-efficient chunks, processing one at a time
                    for result in self.client.messages.batches.results(batch_id):
                        student_id = result.custom_id
                        
                        #response_text = result.get("content", [{}])[0].get("text", "{}")
                        response_text = result.result.message.content[0].text

                        # Handle Claude's response format
                        if not response_text.startswith("{"):
                            response_text = "{" + response_text

                        if student_id:
                            results[student_id] = response_text
                    
                    logger.info(f"Processed {len(results)} results from batch")
                    
                elif status == "errored":
                    logger.error(f"Batch processing failed: {batch_status.error}")
                    break
                elif status == "expired":
                    logger.error(f"Batch processing failed: {batch_status.error}")
                    break
                
                else:
                    # Sleep before polling again
                    if hasattr(batch_status, 'progress') and hasattr(batch_status.progress, 'num_completed'):
                        progress = batch_status.progress.num_completed / len(texts) * 100 if len(texts) > 0 else 0
                        logger.info(f"Batch progress: {progress:.1f}% - Waiting...")
                    else:
                        logger.info("Batch in progress - Waiting...")
                    time.sleep(poll_interval)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch with Claude: {str(e)}")
            # Return error results for all texts
            return {
                student_id: json.dumps({
                    "punktzahl": -999,
                    "staerken": f"Error: {str(e)}",
                    "schwaechen": "Error processing batch", 
                    "begruendung": "Error processing batch"
                })
                for student_id in texts
            }


class MistralRater:
    """Mistral-based text rater adapter."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Mistral rater.
        
        Args:
            api_key: Mistral API key (optional, defaults to environment variable)
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        if not self.api_key:
            raise ValueError("Mistral API key is required")
        
        self.client = Mistral(api_key=self.api_key)
    
    def rate_text(
        self, 
        text: str, 
        system_prompt: str, 
        user_prompt: str, 
        model: str = "mistral-large-latest",
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Rate a text using Mistral's API.
        
        Args:
            text: Text to rate
            system_prompt: System prompt for the model
            user_prompt: User prompt template (will be formatted with text)
            model: Mistral model to use
            temperature: Temperature for generation (optional)
            
        Returns:
            Dictionary with rating results
        """
        specific_user_prompt = user_prompt.format(student_text=text)
        
        ## Build request params
        request_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": specific_user_prompt}
            ],
            "response_format": {"type": "json_object"}
        }
        
        ## Add temperature if provided
        if temperature is not None:
            request_params["temperature"] = temperature
        
        try:
            response = self.client.chat.complete(**request_params)
            response_text = response.choices[0].message.content
            
            ## Parse JSON response
            # return json.loads(response_text)
            return response_text
        
        except Exception as e:
            logger.error(f"Error rating text with Mistral: {str(e)}")
            return {
                "punktzahl": -999,
                "staerken": f"Error: {str(e)}", 
                "schwaechen": "Error processing response",
                "begruendung": "Error processing response"
            }


    def rate_text_batch(
        self,
        texts: Dict[str, str],
        system_prompt: str,
        user_prompt: str,
        model: str = "mistral-large-2411",
        temperature: Optional[float] = None,
        batch_size: int = 50,
        output_dir: str = "./batches",
        poll_interval: int = 60
    ) -> Dict[str, str]:
        """
        Rate multiple texts using Mistral's batch API.
        
        Args:
            texts: Dictionary mapping student IDs to text content
            system_prompt: System prompt for the model
            user_prompt: User prompt template (will be formatted with text)
            model: Mistral model to use
            temperature: Temperature for generation (optional)
            batch_size: Maximum batch size for requests
            output_dir: Directory to save batch files
            poll_interval: Seconds between status checks
            
        Returns:
            Dictionary mapping student IDs to rating results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a unique batch ID
        batch_id = f"mistral_batch_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        batch_file = os.path.join(output_dir, f"{batch_id}.jsonl")
        
        logger.info(f"Creating Mistral batch file with {len(texts)} texts")
        
        # Create JSONL batch file
        with open(batch_file, "w", encoding="utf-8") as f:
            for student_id, text in texts.items():
                specific_user_prompt = user_prompt.format(student_text=text)
                
                # Build request for this text
                request = {
                    "custom_id": student_id,
                    "body": {
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": specific_user_prompt}
                        ]
                        # "metadata": {"student_id": student_id}
                    }
                }
                
                if temperature is not None:
                    request["temperature"] = temperature
                
                # Write to batch file
                f.write(json.dumps(request) + "\n")
        
        logger.info(f"Created batch file at {batch_file}")
        
        # Submit batch request
        try:
            # Create the batch
            batch_data = self.client.files.upload(
                file={
                    "file_name": f"{batch_id}.jsonl",
                    "content": open(batch_file, "rb")
                },
                purpose = "batch"
            )

            created_job = self.client.batch.jobs.create(
                input_files=[batch_data.id],
                model=model,
                endpoint="/v1/chat/completions"
            )

            # retrieved_job = client.batch.jobs.get(job_id=created_job.id)
            
            # batch_data_id = batch_response.id
            batch_data_id = created_job.id
            logger.info(f"Submitted Mistral batch job with ID: {batch_data_id}")
            
            # Poll for completion
            completed = False
            results = {}
            
            while not completed:
                logger.info(f"Checking status of batch {batch_id}...")
                batch_status = self.client.batch.jobs.get(job_id=batch_data_id)
                
                status = batch_status.status
                logger.info(f"Batch status: {status}")
                
                if status == "SUCCESS":
                    completed = True
                    # Get results
                    results_file = self.client.files.download(file_id=batch_status.output_file)
                    # Download results
                    output_file = os.path.join(output_dir, f"{batch_id}_results.jsonl")
                    with open(output_file, "w") as f:
                        for chunk in results_file.stream:
                            f.write(chunk.decode("utf-8"))
                
                    logger.info(f"Downloaded batch results to {output_file}")
                    
                    # Parse results
                    with open(output_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            result = json.loads(line)
                            student_id = result.get("custom_id", "missing_id")
                            response = result.get("response", {}).get("body", {})
                            response_text = response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                            
                            if student_id:
                                results[student_id] = response_text
                    
                    logger.info(f"Processed {len(results)} results from batch")
                    
                elif status == "FAILED":
                    logger.error(f"Batch processing failed: {batch_status.error}")
                    break
                
                else:
                    # Sleep before polling again
                    if hasattr(batch_status, 'progress') and batch_status.progress is not None:
                        progress = batch_status.progress * 100
                        logger.info(f"Batch progress: {progress:.1f}% - Waiting...")
                    else:
                        logger.info("Batch in progress - Waiting...")
                    time.sleep(poll_interval)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch with Mistral: {str(e)}")
            # Return error results for all texts
            return {
                student_id: json.dumps({
                    "punktzahl": -999,
                    "staerken": f"Error: {str(e)}", 
                    "schwaechen": "Error processing batch",
                    "begruendung": "Error processing batch"
                })
                for student_id in texts
            }
