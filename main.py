#!/usr/bin/env python3
"""
AWEsomeScoring

A command line tool for automated writing evaluation (AWE) using different AI models.
"""

import argparse
import os
import sys
import logging
from datetime import datetime

## Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

## Import modules
from config import ConfigManager, Config, ConfigSchema, CONFIG_VERSION
from corpora import CorpusProcessor
from benchmark_rating import BenchmarkRunner

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

def add_config_subparsers(subparsers):
    """Add configuration-related subcommands."""
    config_parser = subparsers.add_parser('config', help='Configuration commands')
    config_subparsers = config_parser.add_subparsers(dest="config_cmd", help='Configuration commands')
    
    ## Initialize config
    init_parser = config_subparsers.add_parser('init', help='Initialize default configuration file')
    init_parser.add_argument('--output', '-o', default='awesome_config.yaml', 
                            help='Output file path for configuration')
    init_parser.set_defaults(func=init_config)
    
    ## Show current config
    show_parser = config_subparsers.add_parser('show', help='Show current configuration')
    show_parser.add_argument('--config', '-c', help='Path to configuration file')
    show_parser.add_argument('--mask-keys', '-m', action='store_true', 
                            help='Mask API keys in the output')
    show_parser.set_defaults(func=show_config)

    ## Dump config
    dump_parser = config_subparsers.add_parser('dump', help='Dump the current effective configuration to a file')
    dump_parser.add_argument('--output', '-o', required=True, help='Output file path for configuration dump')
    dump_parser.add_argument('--config', '-c', help='Path to configuration file to use as base')
    dump_parser.add_argument('--format', '-f', choices=['yaml', 'json', 'ini'], default='yaml',
                           help='Output format (yaml, json, or ini)')
    dump_parser.set_defaults(func=dump_config)

    ## Validate config
    validate_parser = config_subparsers.add_parser('validate', help='Validate a configuration file')
    validate_parser.add_argument('--config', '-c', required=True, help='Path to configuration file to validate')
    validate_parser.add_argument('--command', help='Validate for a specific command (e.g., benchmark:run)')
    validate_parser.set_defaults(func=validate_config)


def add_corpus_subparsers(subparsers):
    """Add corpus-related subcommands."""
    corpus_parser = subparsers.add_parser('corpus', help='Corpus commands')
    corpus_subparsers = corpus_parser.add_subparsers(dest="corpus_cmd", help='Corpus commands')
    
    ## Convert corpus
    convert_parser = corpus_subparsers.add_parser('convert', help='Convert corpus between formats')
    convert_parser.add_argument('--input', '-i', help='Input corpus file path')
    convert_parser.add_argument('--output', '-o', help='Output file path for converted corpus')
    convert_parser.add_argument('--type', '-t', choices=['basch_narrative', 'basch_instructive'], help='Type of corpus')
    convert_parser.add_argument('--config', '-c', help='Path to configuration file')
    convert_parser.set_defaults(func=convert_corpus)
    
    ## List corpus
    list_parser = corpus_subparsers.add_parser('list', help='List all texts in corpus')
    list_parser.add_argument('--input', '-i', help='Input corpus file path')
    list_parser.add_argument('--type', '-t', choices=['basch_narrative', 'basch_instructive'], 
                            help='Type of corpus (not needed for CSV files)')
    list_parser.add_argument('--limit', '-l', type=int, default=5, help='Limit the number of texts to display')
    list_parser.add_argument('--config', '-c', help='Path to configuration file')
    list_parser.set_defaults(func=list_corpus)
    
def add_benchmark_subparsers(subparsers):
    """Add benchmark-related subcommands."""
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark commands')
    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_cmd", help='Benchmark commands')
    
    ## Run benchmark
    run_parser = benchmark_subparsers.add_parser('run', help='Run benchmarks on corpus')
    run_parser.add_argument('--input', '-i', help='Input corpus file path')
    run_parser.add_argument('--output', '-o', help='Output directory for results')
    run_parser.add_argument('--type', '-t', choices=['basch_narrative', 'basch_instructive'], help='Type of corpus')
    ## Base config
    run_parser.add_argument('--system-prompt', '-s', help='System prompt file path')
    run_parser.add_argument('--user-prompt', '-u', help='User prompt file path')
    run_parser.add_argument('--services', nargs="+", choices=["openai", "claude", "mistral"], default=[], help='AI services to use')
    run_parser.add_argument('--config', '-c', help='Path to configuration file')
    ## Config details
    run_parser.add_argument('--runs', '-r', type=int, help='Number of benchmark runs')
    run_parser.add_argument('--limit', '-l', type=int, help='Limit the number of texts to score')
    run_parser.add_argument('--temperature', type=float, help='Temperature for generation')
    run_parser.add_argument('--no-temperature', action='store_true', 
                           help="Don't include temperature in API calls (use model defaults)")
    ## Batch processing commands
    run_parser.add_argument('--batch', action='store_true', help='Use batch API mode')
    run_parser.add_argument('--batch-size', type=int, help='Batch size for batch API calls')
    run_parser.add_argument('--batch-poll-interval', type=int, help='Polling interval in seconds for batch API status checks')
    ## Other
    run_parser.add_argument('--verbose', '-v', action='count', default=0, 
                           help='Increase verbosity (can be used multiple times)')
    run_parser.set_defaults(func=run_benchmark)


## ==================================================================
## Config Commands

def init_config(args):
    """Initialize a default configuration file."""
    try:
        ## Get default configuration
        default_config = ConfigManager.get_default_config()
        
        ## Save the configuration to the specified file
        success = ConfigManager.save_config(default_config, args.output)
        
        if success:
            logger.info(f"Configuration initialized at {args.output}")
            
            ## Print environment variable settings advice
            print("\nTo set API keys as environment variables:")
            print("export OPENAI_API_KEY=your_key_here")
            print("export ANTHROPIC_API_KEY=your_key_here")
            print("export MISTRAL_API_KEY=your_key_here")
            
            ## Print next steps
            print("\nNext steps:")
            print(f"1. Edit {args.output} to configure your settings")
            print("2. Run 'awescore config validate --config your_config.yaml' to validate your configuration")
            print("3. Use your configuration with other commands via --config your_config.yaml")
        else:
            logger.error(f"Failed to initialize configuration at {args.output}")
        
    except Exception as e:
        logger.error(f"Error creating config file: {str(e)}")
        print(f"\nError: {str(e)}")


def show_config(args):
    """Show current configuration."""
    import yaml
    
    try:
        ## Create Config object that combines all configuration sources
        config_obj = Config(args, command="config:show")
        
        ## Get dictionary representation of the configuration
        config = config_obj.to_dict()
        
        ## Mask API keys if requested
        if getattr(args, 'mask_keys', False):
            for key in config:
                if 'api_' in key and config[key]:
                    config[key] = config[key][:4] + '...' + config[key][-4:] if len(config[key]) > 8 else "****"
        
        ## Get the configuration source description
        source_description = config_obj.get_source_description()
        
        ## Display configuration information
        print("\nCurrent effective configuration:")
        print(yaml.dump(config, default_flow_style=False))
        
        ## Show configuration sources
        print("\n" + source_description)
        
        ## Show information about config file if available
        if hasattr(config_obj, '_original_config') and '_config_file_path' in config_obj._original_config:
            print(f"\nConfig file used: {config_obj._original_config['_config_file_path']}")
        
        ## Show configuration version
        print(f"\nConfiguration schema version: {config_obj.config_version}")
    
    except Exception as e:
        logger.error(f"Error showing configuration: {str(e)}")
        print(f"\nError: {str(e)}")


def dump_config(args):
    """Dump the current effective configuration to a file."""
    try:
        ## Create Config object that combines all configuration sources
        config_obj = Config(args, command="config:dump")
        
        ## Get dictionary representation of the configuration
        config = config_obj.to_dict()
        
        ## Adjust output file name based on specified format
        output_path = args.output
        if args.format == 'yaml' and not output_path.endswith(('.yaml', '.yml')):
            output_path += '.yaml'
        elif args.format == 'json' and not output_path.endswith('.json'):
            output_path += '.json'
        elif args.format == 'ini' and not output_path.endswith('.ini'):
            output_path += '.ini'
        
        ## Save configuration to file
        success = ConfigManager.save_config(config, output_path)
        
        if success:
            logger.info(f"Configuration dumped to {output_path}")
            print(f"\nConfiguration successfully dumped to {output_path}")
            
            ## Show the configuration source description
            print("\n" + config_obj.get_source_description())
        else:
            logger.error(f"Failed to dump configuration to {output_path}")
            print(f"\nError: Failed to dump configuration to {output_path}")
    
    except Exception as e:
        logger.error(f"Error dumping configuration: {str(e)}")
        print(f"\nError: {str(e)}")


def validate_config(args):
    """Validate a configuration file."""
    try:
        ## Load the configuration from the specified file
        file_config = ConfigManager._load_from_file(args.config)
        
        if not file_config:
            print(f"\nError: Could not load configuration from {args.config}")
            return
        
        ## Validate the configuration
        command = args.command if args.command else None
        is_valid, errors = ConfigSchema.validate(file_config, command)
        
        ## Display validation results
        if is_valid:
            print(f"\nConfiguration file {args.config} is valid")
            if command:
                print(f"Configuration is valid for command: {command}")
            print(f"Configuration schema version: {file_config.get('config_version', 'Not specified')}")
        else:
            print(f"\nConfiguration file {args.config} has validation errors:")
            for i, error in enumerate(errors):
                print(f"{i+1}. {error}")
            
            print("\nPlease fix these errors and try again.")
    
    except Exception as e:
        logger.error(f"Error validating configuration: {str(e)}")
        print(f"\nError: {str(e)}")


## ==================================================================
## Corpus Commands

def convert_corpus(args):
    """Convert corpus between formats."""
    try:
        ## Load configuration
        config = Config(args, command="corpus:convert")
        
        ## Use configuration values if CLI arguments are not provided
        input_path = args.input or config.input_corpus_path
        corpus_type = args.type or config.corpus_type
        
        if not input_path:
            raise ValueError("Input corpus file path is required")
            
        if not os.path.exists(input_path):
            raise ValueError(f"Input corpus file does not exist: {input_path}")
            
        if input_path.endswith('.csv'):
            corpus = CorpusProcessor.load_corpus_from_csv(input_path)
            output_path = args.output or "corpus_converted.txt"
            ## No direct conversion method to text currently, just save as CSV
            CorpusProcessor.save_corpus_to_csv(corpus, output_path + ".csv")
        else:
            if not corpus_type:
                raise ValueError("Corpus type is required for text files")
                
            corpus = CorpusProcessor.load_corpus_from_txt(input_path, corpus_type)
            output_path = args.output or "corpus_converted.csv"
            CorpusProcessor.save_corpus_to_csv(corpus, output_path)
        
        logger.info(f"Conversion completed. Output saved to {output_path}")
        print(f"\nCorpus conversion completed. Output saved to {output_path}")
    
    except Exception as e:
        logger.error(f"Error converting corpus: {str(e)}")
        print(f"\nError: {str(e)}")


def list_corpus(args):
    """List texts in corpus."""
    try:
        ## Load configuration
        config = Config(args, command="corpus:list")
        
        ## Use configuration values if CLI arguments are not provided
        input_path = args.input or config.input_corpus_path
        corpus_type = args.type or config.corpus_type
        
        if not input_path:
            raise ValueError("Input corpus file path is required")
        
        if not os.path.exists(input_path):
            raise ValueError(f"Input corpus file does not exist: {input_path}")
        
        corpus = CorpusProcessor.load_corpus(input_path, corpus_type)
        
        print(f"\nLoaded {len(corpus)} texts from corpus '{input_path}'")
        print(f"Showing first {min(args.limit, len(corpus))} texts:\n")
        
        for i, (student_id, text) in enumerate(list(corpus.items())[:args.limit]):
            preview = text[:100] + "..." if len(text) > 100 else text
            preview = preview.replace('\n', ' ')
            print(f"{i+1}. Student ID: {student_id}")
            print(f"   Text: {preview}\n")

    
    except Exception as e:
        logger.error(f"Error listing corpus: {str(e)}")
        print(f"\nError: {str(e)}")


## ==================================================================
## Benchmark Commands
        
def run_benchmark(args):
    """Run benchmarks on corpus."""
    try:
        ## Track use_temperature based on --no-temperature flag
        # if hasattr(args, 'no_temperature') and args.no_temperature:
        #     args.use_temperature = False
        # else:
        #     args.use_temperature = True

        ## Track use_batch_mode based on command line flag
        # if hasattr(args, 'use_batch_mode') and args.batch:
        #     args.use_batch_mode = True
        # else:
        #     args.use_batch_mode = False
        
        ## Set logging level based on verbosity
        if args.verbose == 1:
            logger.setLevel(logging.INFO)
        elif args.verbose >= 2:
            logger.setLevel(logging.DEBUG)
        
        ## Create configuration with command context for validation
        config = Config(args, command="benchmark:run")
        
        ## Display effective configuration if in verbose mode
        if args.verbose >= 1:
            import yaml
            logger.info("Effective configuration:")
            logger.info(yaml.dump(config.to_dict(), default_flow_style=False))
        
        ## Run benchmarks
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


## ==================================================================
## MAIN Commands
        
def main():
    """Main entry point for the CLI tool."""
    try:
        parser = argparse.ArgumentParser(
            description=f"AWEsomeScoring (v{CONFIG_VERSION}) - Command line tool for automated writing evaluation (AWE) using different AI models."
        )
        subparsers = parser.add_subparsers(dest="command", help="Commands")

        ## Add commands from modules
        add_config_subparsers(subparsers)
        add_corpus_subparsers(subparsers)
        add_benchmark_subparsers(subparsers)

        ## Parser common arguments that apply to multiple commands
        parser.add_argument('--config', '-c', help='Path to configuration file')
        parser.add_argument('--version', '-v', action='store_true', help='Show version information')

        args = parser.parse_args()
        
        ## Handle version display
        if args.version:
            print(f"AWEsomeScoring version {CONFIG_VERSION}")
            print("A command line tool for automated writing evaluation (AWE) using different AI models.")
            print("MIT License - Copyright (c) 2025 Fabian Grünig")
            return
        
        ## Parse the command context for configuration validation
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
        
        ## Run command if given
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
