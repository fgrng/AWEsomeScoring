"""
Benchmark command implementations for AWEsomeScoring CLI.

This module contains all benchmark-related command implementations.
"""

import logging
import yaml
from config import Config
from benchmark_rating import BenchmarkRunner

logger = logging.getLogger("awesome_scoring")


def run_benchmark(args):
    """Run benchmarks on corpus."""
    try:
        # Track use_temperature based on --no-temperature flag
        # if hasattr(args, 'no_temperature') and args.no_temperature:
        #     args.use_temperature = False
        # else:
        #     args.use_temperature = True

        # Track use_batch_mode based on command line flag
        # if hasattr(args, 'use_batch_mode') and args.batch:
        #     args.use_batch_mode = True
        # else:
        #     args.use_batch_mode = False
        
        # Set logging level based on verbosity
        if args.verbose == 1:
            logger.setLevel(logging.INFO)
        elif args.verbose >= 2:
            logger.setLevel(logging.DEBUG)
        
        # Create configuration with command context for validation
        config = Config(args, command="benchmark:run")
        
        # Display effective configuration if in verbose mode
        if args.verbose >= 1:
            logger.info("Effective configuration:")
            logger.info(yaml.dump(config.to_dict(), default_flow_style=False))
        
        # Run benchmarks
        runner = BenchmarkRunner(config)
        runner.run()
        
        logger.info("Benchmark run completed successfully")
        print("\nBenchmark run completed successfully.")
    
    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}")
        if args.verbose >= 1:
            import traceback
            logger.error(traceback.format_exc())
        print(f"\nError: {str(e)}")
