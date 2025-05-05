import asyncio
import concurrent.futures
import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from configparser import ConfigParser
import random
import yaml

# Import AI API clients
try:
    from mistralai import File, Mistral
    from openai import OpenAI, RateLimitError, BadRequestError
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    IMPORTS_SUCCESSFUL = True
except ImportError:
    IMPORTS_SUCCESSFUL = False
    print("Warning: Some AI API libraries are missing. Install with:")
    print("pip install openai anthropic mistralai requests pyyaml")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("benchmark")

# Define expected CSV headers for output
CSV_HEADERS = ["student_id", "punktzahl", "staerken", "schwaechen", "begruendung"]

class BenchmarkRunner:
    """
    Run benchmarks using different AI services.
    
    This class orchestrates the benchmark process, including loading prompts
    and corpora, running benchmarks with different AI services, and saving results.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the benchmark runner.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Load prompts
        with open(config.system_prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()
            
        with open(config.user_prompt_path, "r", encoding="utf-8") as f:
            self.user_prompt = f.read()
            
        # Load corpus
        self.corpus = CorpusProcessor.load_corpus(
            config.input_corpus_path, config.corpus_type)
            
        # Save corpus as CSV for reference
        corpus_csv_path = os.path.join(config.output_dir, "input_corpus.csv")
        CorpusProcessor.save_corpus_to_csv(self.corpus, corpus_csv_path)
        
        # Save configuration for reference
        config_path = os.path.join(config.output_dir, "benchmark_config.yaml")
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(vars(config), f, default_flow_style=False)
                
            logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    def run(self) -> None:
        """
        Run all benchmark tests based on configuration.
        """
        logger.info("Starting benchmark runs")
        
        for run_num in range(1, self.config.num_runs + 1):
            logger.info(f"Starting RUN {run_num}/{self.config.num_runs}")
            
            # Create run-specific output directory
            run_dir = os.path.join(
                self.config.output_dir, 
                f"run_{run_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            os.makedirs(run_dir, exist_ok=True)
            
            # Run benchmarks for each enabled service
            if self.config.use_openai:
                self.run_openai_benchmark(run_dir, run_num)
                
            if self.config.use_claude:
                self.run_claude_benchmark(run_dir, run_num)
                
            if self.config.use_mistral:
                self.run_mistral_benchmark(run_dir, run_num)
                
        logger.info("All benchmark runs completed")
    
    def run_openai_benchmark(self, output_dir: str, run_num: int) -> None:
        """
        Run benchmark using OpenAI API.
        
        Args:
            output_dir: Directory to save results
            run_num: Current run number
        """
        logger.info(f"Running OpenAI benchmark (Run {run_num})")
        
        # Skip if API key is not available
        if not self.config.api_openai:
            logger.warning("Skipping OpenAI benchmark - API key not provided")
            return
        
        model = self.config.model_openai
        temperature = self.config.temperature
        comment = f"{self.config.comment_openai}_run{run_num}"
        
        # Initialize OpenAI client
        client = OpenAI(
            api_key=self.config.api_openai,
            max_retries=10
        )
        
        # Define output file path
        output_file = os.path.join(
            output_dir, 
            f"data_openai_{model}_temp{temperature}_{comment}.csv"
        )
        
        # Helper function to process a single student request
        def process_student_request(student_id: str, student_text: str) -> Tuple[str, Dict[str, Any]]:
            """
            Create request & get response for a single student using OpenAI API.
            
            Args:
                student_id: Student ID
                student_text: Student's text to evaluate
                
            Returns:
                Tuple of (student_id, response_json)
            """
            specific_user_prompt = self.user_prompt.format(student_text=student_text)
            
            logger.info(f"Processing student {student_id} with OpenAI")
            
            # Build request params with conditional temperature
            request_params = {
                "model": model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": specific_user_prompt}
                ]
            }
            
            # Only include temperature if use_temperature is True
            if self.config.use_temperature:
                request_params["temperature"] = temperature
            
            # Send the request to OpenAI with robust retry logic
            response = RetryHandler.retry_with_backoff(
                lambda: client.chat.completions.create(**request_params),
                max_retries=self.config.retry_max_attempts,
                initial_wait=self.config.retry_initial_wait,
                description=f"OpenAI API call for student {student_id}"
            )
            
            # Write raw response to file
            raw_output_file = os.path.join(
                output_dir,
                f"data_openai_{model}_temp{temperature}_{comment}_{student_id}.json"
            )
            ResultsManager.save_raw_response(response, raw_output_file)
            
            # Extract the response content
            response_text = response.choices[0].message.content
            
            # Parse JSON response with robust error handling
            default_values = {
                "punktzahl": -999,
                "staerken": "Error parsing response",
                "schwaechen": "Error parsing response",
                "begruendung": "Error parsing response"
            }
            response_json = ResultsManager.parse_json_response(response_text, default_values)
            
            logger.info(f"Completed processing for student {student_id} with OpenAI")
            return student_id, response_json
        
        # Process all students in parallel using ThreadPoolExecutor
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_student = {
                executor.submit(process_student_request, student_id, student_text): student_id
                for student_id, student_text in self.corpus.items()
            }
            
            # Track progress
            total_students = len(future_to_student)
            completed = 0
            
            # Collect results as they become available
            for future in concurrent.futures.as_completed(future_to_student):
                student_id, response = future.result()
                response["student_id"] = student_id
                results.append(response)
                
                # Update progress
                completed += 1
                logger.info(f"OpenAI progress: {completed}/{total_students} ({completed/total_students*100:.1f}%)")
        
        # Save results to CSV
        ResultsManager.save_results_to_csv(results, output_file)
        
        logger.info(f"OpenAI benchmark run {run_num} completed")