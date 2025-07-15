"""
Benchmark rating module for AWEsomeScoring.

This module contains the BenchmarkRunner class which orchestrates
the process of evaluating text corpora using different AI models.
"""

import os
from pathlib import Path

import json
import time
import logging
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

## Import local modules
from config import Config
from corpora import CorpusProcessor
from results import ResultsManager, CSV_HEADERS
from utils import RetryHandler, format_time_elapsed
from ai import TextRater, OpenAIRater, ClaudeRater, MistralRater

## Configure logging
logger = logging.getLogger("awesome_scoring")

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
        
        ## Load prompts
        with open(config.system_prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()
            
        with open(config.user_prompt_path, "r", encoding="utf-8") as f:
            self.user_prompt = f.read()
            
        ## Load corpus
        self.corpus = CorpusProcessor.load_corpus(
            config.input_corpus_path, config.corpus_type)
        
        ## Use limit paramet to reduce number of texts to score
        if self.config.limit > 0:
            self.corpus = {k: v for i, (k, v) in enumerate(self.corpus.items()) if i < config.limit}
            
        ## Save corpus as CSV for reference
        corpus_csv_path = os.path.join(config.output_dir, "input_corpus.csv")
        CorpusProcessor.save_corpus_to_csv(self.corpus, corpus_csv_path)
        
        ## Save configuration for reference
        config_path = os.path.join(config.output_dir, "benchmark_config.json")
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                ## Convert config to dict and save as JSON
                config_dict = {k: v for k, v in vars(config).items() 
                              if not k.startswith('_') and not callable(v)}
                json.dump(config_dict, f, indent=2, default=str)
                
            logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
        
        ## Initialize raters
        self._init_raters()
    
    def _init_raters(self):
        """Initialize AI raters based on configuration."""
        self.raters = {}
        
        ## Initialize OpenAI rater if enabled
        if self.config.use_openai:
            try:
                self.raters['openai'] = OpenAIRater(api_key=self.config.api_openai)
                logger.info("Initialized OpenAI rater")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI rater: {str(e)}")
        
        ## Initialize Claude rater if enabled
        if self.config.use_claude:
            try:
                self.raters['claude'] = ClaudeRater(api_key=self.config.api_claude)
                logger.info("Initialized Claude rater")
            except Exception as e:
                logger.error(f"Failed to initialize Claude rater: {str(e)}")
        
        ## Initialize Mistral rater if enabled
        if self.config.use_mistral:
            try:
                self.raters['mistral'] = MistralRater(api_key=self.config.api_mistral)
                logger.info("Initialized Mistral rater")
            except Exception as e:
                logger.error(f"Failed to initialize Mistral rater: {str(e)}")
        
        if not self.raters:
            raise ValueError("No AI raters could be initialized. Check API keys and configurations.")
    
    def run(self) -> None:
        """
        Run all benchmark tests based on configuration.
        """
        logger.info("Starting benchmark runs")
        
        start_time = time.time()
        
        for run_num in range(1, self.config.num_runs + 1):
            logger.info(f"Starting RUN {run_num}/{self.config.num_runs}")
            
            ## Create run-specific output directory
            run_dir = os.path.join(
                self.config.output_dir, 
                f"run_{run_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            os.makedirs(run_dir, exist_ok=True)
            
            ## Run benchmarks for each enabled service (batch o parallel)
            # for service, rater in self.raters.items():
            #     # Check if batch mode is enabled
            #     if self.config.use_batch_mode:
            #         logger.info("Starting RUN in batch mode.")
            #         self.run_benchmark_batch(service, rater, run_dir, run_num)
            #     else:
            #         logger.info("Starting RUN in parallel mode.")
            #         self.run_benchmark(service, rater, run_dir, run_num)


            ## Run benchmarks for all enabled services in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.raters)) as executor:
                # Check if batch mode is enabled
                if self.config.use_batch_mode:
                    logger.info("Starting RUN in batch mode.")
                    # Submit all service benchmark tasks
                    futures = {
                        executor.submit(self.run_benchmark_batch, service, rater, run_dir, run_num): service
                        for service, rater in self.raters.items()
                    }
                else:
                    logger.info("Starting RUN in parallel mode.")
                    # Submit all service benchmark tasks
                    futures = {
                        executor.submit(self.run_benchmark, service, rater, run_dir, run_num): service
                        for service, rater in self.raters.items()
                    }
                
                # Wait for all to complete and handle any exceptions
                for future in concurrent.futures.as_completed(futures):
                    service = futures[future]
                    try:
                        future.result()
                        logger.info(f"Completed {service} benchmark for run {run_num}")
                    except Exception as e:
                        logger.error(f"Error executing {service} benchmark: {str(e)}")
                        if self.config.verbose >= 1:
                            import traceback
                            logger.error(traceback.format_exc())
        
        ## Calculate total elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"All benchmark runs completed in {format_time_elapsed(elapsed_time)}")
    
    def run_benchmark(self, service: str, rater: Any, output_dir: str, run_num: int) -> None:
        """
        Run benchmark using the specified AI service.
        
        Args:
            service: Service name ('openai', 'claude', or 'mistral')
            rater: Rater instance
            output_dir: Directory to save results
            run_num: Current run number
        """
        logger.info(f"Running {service.capitalize()} benchmark (Run {run_num})")
        
        ## Get configuration for this service
        model = getattr(self.config, f"model_{service}")
        temperature = self.config.temperature if self.config.use_temperature else None

        ## Define output file path
        temp_str = f"temp{self.config.temperature}" if self.config.use_temperature else "noTemp"
        output_file = os.path.join(
            output_dir,
            f"data_{service}_{model}_{temp_str}_run{run_num}.csv"
        )
        
        ## Process function for individual texts
        def process_text(student_id: str, text: str) -> Tuple[str, Dict[str, Any]]:
            """Process a single text with the rater."""

            # try:
            logger.info(f"Processing student {student_id} with {service}")

            ## Rate the text
            result = rater.rate_text(
                text=text,
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
                model=model,
                temperature=temperature
            )

            ## Save raw response
            raw_output_file = os.path.join(
                output_dir,
                f"raw_{service}_{model}_{temp_str}_run{run_num}_{student_id}.json"
            )
            ResultsManager.save_raw_response(result, raw_output_file)

            logger.info(f"Completed processing for student {student_id} with {service}")

            return student_id, json.loads(result)
                
            # except Exception as e:               
            #     logger.error(f"Error processing student {student_id} with {service}: {str(e)}")
            #     return student_id, {
            #         "punktzahl": -999,
            #         "staerken": f"Error: {str(e)}",
            #         "schwaechen": "Error processing with AI service",
            #         "begruendung": "Error processing with AI service"
            #     }

        ## Check, if batch processing is available.
        
        
        ## Process all texts in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(process_text, student_id, text): student_id
                for student_id, text in self.corpus.items()
            }
            
            ## Track progress
            total = len(futures)
            completed = 0
            start_time = time.time()
            
            ## Collect results as they become available
            for future in concurrent.futures.as_completed(futures):
                student_id, result = future.result()
                
                ## Add student_id to result if not already present
                if "student_id" not in result:
                    result["student_id"] = student_id

                ## Ensure each value is a string
                for key, value in result.items():
                    ## Store nested structures as JSON strings
                    if isinstance(value, (dict, list)):
                        result[key] = json.dumps(value, ensure_ascii=False)
                    else:
                        result[key] = str(value)
                    
                results.append(result)
                
                ## Update progress
                completed += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / completed if completed > 0 else 0
                remaining = avg_time * (total - completed)
                
                ## Calculate ETA
                logger.info(
                    f"{service.capitalize()} progress: {completed}/{total} "
                    f"({completed/total*100:.1f}%) - "
                    f"Avg: {avg_time:.1f}s/text - "
                    f"ETA: {format_time_elapsed(remaining)}"
                )
        
        ## Save results to CSV
        ResultsManager.save_results_to_csv(results, output_file)
        
        logger.info(f"{service.capitalize()} benchmark run {run_num} completed")


    def run_benchmark_batch(self, service: str, rater: Any, output_dir: str, run_num: int) -> None:
        """
        Run benchmark using the specified AI service in batch mode.

        Args:
            service: Service name ('openai', 'claude', or 'mistral')
            rater: Rater instance
            output_dir: Directory to save results
            run_num: Current run number
        """
        logger.info(f"Running {service.capitalize()} batch benchmark (Run {run_num})")

        ## Get configuration for this service
        model = getattr(self.config, f"model_{service}")
        temperature = self.config.temperature if self.config.use_temperature else None

        ## Define output file path
        temp_str = f"temp{self.config.temperature}" if self.config.use_temperature else "noTemp"
        output_file = os.path.join(
            output_dir,
            f"data_{service}_{model}_{temp_str}_batch_run{run_num}.csv"
        )

        ## Create batch directory
        batch_dir = os.path.join(output_dir, "batches")
        os.makedirs(batch_dir, exist_ok=True)

        start_time = time.time()

        try:
            # Submit the entire corpus as a batch
            logger.info(f"Submitting batch of {len(self.corpus)} texts to {service}")

            # Process batch
            results_dict = rater.rate_text_batch(
                texts=self.corpus,
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
                model=model,
                temperature=temperature,
                batch_size=self.config.batch_size,
                output_dir=batch_dir,
                poll_interval=self.config.batch_poll_interval
            )

            # Convert results to list format expected by ResultsManager
            results = []
            for student_id, response_text in results_dict.items():
                try:
                    # Parse the JSON response
                    result = json.loads(response_text)

                    # Add student_id if not present
                    if "student_id" not in result:
                        result["student_id"] = student_id

                    # Ensure each value is a string (this was in the original code)
                    for key in result:
                        result[key] = str(json.dumps(result[key]))

                    results.append(result)

                    # Save raw response
                    raw_output_file = os.path.join(
                        output_dir,
                        f"raw_{service}_{model}_{temp_str}_batch_run{run_num}_{student_id}.json"
                    )
                    ResultsManager.save_raw_response(response_text, raw_output_file)

                except json.JSONDecodeError:
                    logger.error(f"Error parsing JSON response for student {student_id}: {response_text[:100]}...")
                    result = {
                        "student_id": student_id,
                        "punktzahl": json.dumps(-999),
                        "staerken": json.dumps("Error parsing JSON response"),
                        "schwaechen": json.dumps("Error parsing JSON response"),
                        "begruendung": json.dumps("Error parsing JSON response")
                    }
                    results.append(result)

            # Save results to CSV
            ResultsManager.save_results_to_csv(results, output_file)

            # Report completion
            elapsed_time = time.time() - start_time
            logger.info(
                f"{service.capitalize()} batch benchmark run {run_num} completed in "
                f"{format_time_elapsed(elapsed_time)} for {len(results)} texts"
            )

        except Exception as e:
            logger.error(f"Error in batch processing with {service}: {str(e)}")
            if self.config.verbose >= 1:
                import traceback
                logger.error(traceback.format_exc())
