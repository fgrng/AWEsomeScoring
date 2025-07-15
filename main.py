#!/usr/bin/env python3
"""
AWEsomeScoring

A command line tool for automated writing evaluation (AWE) using different AI models.
"""

import os
import sys
import logging
from datetime import datetime

## Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the CLI parser
from cli import create_parser
from config import CONFIG_VERSION

## Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"awesome_scoring_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("awesome_scoring")

def main():
    """Main entry point for the CLI tool."""
    try:
        # Create the argument parser
        parser = create_parser()
        
        # Parse arguments
        args = parser.parse_args()
        
        # Handle version display
        if args.version:
            print(f"AWEsomeScoring version {CONFIG_VERSION}")
            print("A command line tool for automated writing evaluation (AWE) using different AI models.")
            print("MIT License - Copyright (c) 2025 Fabian Grünig")
            return
        
        # Parse the command context for configuration validation
        command_context = None
        if args.command:
            if hasattr(args, 'config_cmd') and args.config_cmd:
                command_context = f"{args.command}:{args.config_cmd}"
            elif hasattr(args, 'corpus_cmd') and args.corpus_cmd:
                command_context = f"{args.command}:{args.corpus_cmd}"
            elif hasattr(args, 'benchmark_cmd') and args.benchmark_cmd:
                command_context = f"{args.command}:{args.benchmark_cmd}"
            else:
                command_context = args.command
        
        # Run command if given
        if hasattr(args, 'func'):
            args.func(args)
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}")
        print(f"\nError: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
