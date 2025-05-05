"""
AI Service Adapters

This module contains classes for interacting with different AI services 
(OpenAI, Claude, Mistral) to evaluate text.
"""

import logging
import os
import json
from typing import Dict, Any, Optional

## Import AI API clients
try:
    from openai import OpenAI
    from anthropic import Anthropic
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
        logger.info("Initialized OpenAI rater")
    
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
        
        try:
            response = self.client.chat.completions.create(**request_params)
            response_text = response.choices[0].message.content
            
            ## Parse JSON response
            return json.loads(response_text)
        
        except Exception as e:
            logger.error(f"Error rating text with OpenAI: {str(e)}")
            return {
                "punktzahl": -999,
                "staerken": f"Error: {str(e)}",
                "schwaechen": "Error processing response", 
                "begruendung": "Error processing response"
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
        logger.info("Initialized Claude rater")
    
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
                {"role": "user", "content": specific_user_prompt}
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
                return json.loads(response_text)
            else:
                return json.loads("{" + response_text)  ## Claude might omit the opening brace
        
        except Exception as e:
            logger.error(f"Error rating text with Claude: {str(e)}")
            return {
                "punktzahl": -999,
                "staerken": f"Error: {str(e)}",
                "schwaechen": "Error processing response",
                "begruendung": "Error processing response"
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
        logger.info("Initialized Mistral rater")
    
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
            return json.loads(response_text)
        
        except Exception as e:
            logger.error(f"Error rating text with Mistral: {str(e)}")
            return {
                "punktzahl": -999,
                "staerken": f"Error: {str(e)}", 
                "schwaechen": "Error processing response",
                "begruendung": "Error processing response"
            }
