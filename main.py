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
from config import ConfigManager
from corpora import CorpusProcessor
from benchmark_rating import BenchmarkRunner
from ai import TextRater

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
    show_parser.set_defaults(func=show_config)


def add_corpus_subparsers(subparsers):
    """Add corpus-related subcommands."""
    corpus_parser = subparsers.add_parser('corpus', help='Corpus commands')
    corpus_subparsers = corpus_parser.add_subparsers(dest="corpus_cmd", help='Corpus commands')
    
    ## Convert corpus
    convert_parser = corpus_subparsers.add_parser('convert', help='Convert corpus between formats')
    convert_parser.add_argument('--input', '-i', required=True, help='Input corpus file path')
    convert_parser.add_argument('--output', '-o', help='Output file path for converted corpus')
    convert_parser.add_argument('--type', '-t', choices=['basch_narrative', 'basch_instructive'], 
                               required=True, help='Type of corpus')
    convert_parser.set_defaults(func=convert_corpus)
    
    ## List corpus
    list_parser = corpus_subparsers.add_parser('list', help='List all texts in corpus')
    list_parser.add_argument('--input', '-i', required=True, help='Input corpus file path')
    list_parser.add_argument('--type', '-t', choices=['basch_narrative', 'basch_instructive'], 
                            help='Type of corpus (not needed for CSV files)')
    list_parser.add_argument('--limit', '-l', type=int, default=5, help='Limit the number of texts to display')
    list_parser.set_defaults(func=list_corpus)


def add_benchmark_subparsers(subparsers):
    """Add benchmark-related subcommands."""
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark commands')
    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_cmd", help='Benchmark commands')
    
    ## Run benchmark
    run_parser = benchmark_subparsers.add_parser('run', help='Run benchmarks on corpus')
    run_parser.add_argument('--input', '-i', required=True, help='Input corpus file path')
    run_parser.add_argument('--output', '-o', help='Output directory for results')
    run_parser.add_argument('--type', '-t', choices=['basch_narrative', 'basch_instructive'], 
                           required=True, help='Type of corpus')
    run_parser.add_argument('--system-prompt', '-s', required=True, help='System prompt file path')
    run_parser.add_argument('--user-prompt', '-u', required=True, help='User prompt file path')
    run_parser.add_argument('--services', nargs="+", choices=["openai", "claude", "mistral"], 
                           default=["openai", "claude", "mistral"], help='AI services to use')
    run_parser.add_argument('--runs', '-r', type=int, default=1, help='Number of benchmark runs')
    run_parser.add_argument('--config', '-c', help='Path to configuration file')
    run_parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for generation')
    run_parser.add_argument('--no-temperature', action='store_true', 
                           help="Don't include temperature in API calls (use model defaults)")
    run_parser.add_argument('--verbose', '-v', action='count', default=0, 
                           help='Increase verbosity (can be used multiple times)')
    run_parser.set_defaults(func=run_benchmark)


def init_config(args, rater):
    """Initialize a default configuration file."""
    from config import ConfigManager
    import yaml
    
    default_config = {
        'api_openai': '',
        'api_claude': '',
        'api_mistral': '',
        'url_openai': 'https://api.openai.com/v1/chat/completions',
        'url_claude': 'https://api.anthropic.com/v1/messages',
        'temperature': 0.7,
        'model_openai': 'gpt-4o-2024-11-20',
        'model_claude': 'claude-3-7-sonnet-20250219',
        'model_mistral': 'mistral-large-2411',
        'comment_openai': 'Run',
        'comment_claude': 'Run',
        'comment_mistral': 'Run',
        'num_runs': 1,
        'max_workers': 10,
        'retry_max_attempts': 12,
        'retry_initial_wait': 2
    }
    
    try:
        with open(args.output, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        logger.info(f"Configuration initialized at {args.output}")
        
        ## Print environment variable settings advice
        print("\nTo set API keys as environment variables:")
        print("export OPENAI_API_KEY=your_key_here")
        print("export ANTHROPIC_API_KEY=your_key_here")
        print("export MISTRAL_API_KEY=your_key_here")
        
    except Exception as e:
        logger.error(f"Error creating config file: {str(e)}")


def show_config(args, rater):
    """Show current configuration."""
    import yaml
    from config import ConfigManager
    
    ## Create a dummy argparse namespace for config loading
    class DummyArgs:
        pass
    
    dummy_args = DummyArgs()
    for key, value in vars(args).items():
        setattr(dummy_args, key, value)
    
    config = ConfigManager.load_config(dummy_args)
    
    ## Mask API keys for security
    for key in config:
        if 'api_' in key and config[key]:
            config[key] = config[key][:4] + '...' + config[key][-4:]
    
    print("\nCurrent configuration:")
    print(yaml.dump(config, default_flow_style=False))
    print("\nConfiguration is loaded from (in order of precedence):")
    print("1. Command line arguments")
    print("2. Environment variables")
    print("3. Config file specified by --config")
    print("4. Default config files")


def convert_corpus(args, rater):
    """Convert corpus between formats."""
    from corpora import CorpusProcessor
    
    try:
        if args.input.endswith('.csv'):
            corpus = CorpusProcessor.load_corpus_from_csv(args.input)
            output_path = args.output or "corpus_converted.txt"
            ## No direct conversion method to text currently, just save as CSV
            CorpusProcessor.save_corpus_to_csv(corpus, output_path + ".csv")
        else:
            corpus = CorpusProcessor.load_corpus_from_txt(args.input, args.type)
            output_path = args.output or "corpus_converted.csv"
            CorpusProcessor.save_corpus_to_csv(corpus, output_path)
        
        logger.info(f"Conversion completed. Output saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error converting corpus: {str(e)}")


def list_corpus(args, rater):
    """List texts in corpus."""
    from corpora import CorpusProcessor
    
    try:
        corpus = CorpusProcessor.load_corpus(args.input, args.type)
        
        print(f"\nLoaded {len(corpus)} texts from corpus '{args.input}'")
        print(f"Showing first {min(args.limit, len(corpus))} texts:\n")
        
        for i, (student_id, text) in enumerate(list(corpus.items())[:args.limit]):
            preview = text[:100] + "..." if len(text) > 100 else text
            preview = preview.replace('\n', ' ')
            print(f"{i+1}. Student ID: {student_id}")
            print(f"   Text: {preview}\n")
            
    except Exception as e:
        logger.error(f"Error listing corpus: {str(e)}")


def run_benchmark(args, rater):
    """Run benchmarks on corpus."""
    from config import Config
    from benchmark_rating import BenchmarkRunner
    
    try:
        ## Track use_temperature based on --no-temperature flag
        if hasattr(args, 'no_temperature') and args.no_temperature:
            args.use_temperature = False
        else:
            args.use_temperature = True
        
        ## Set logging level based on verbosity
        if args.verbose == 1:
            logger.setLevel(logging.INFO)
        elif args.verbose >= 2:
            logger.setLevel(logging.DEBUG)
        
        ## Create configuration
        config = Config(args)
        
        ## Run benchmarks
        runner = BenchmarkRunner(config)
        runner.run()
        
        logger.info("Benchmark run completed successfully")
        
    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}")
        if args.verbose >= 1:
            import traceback
            logger.error(traceback.format_exc())


def load_config():
    """Load configuration from environment or config file."""
    ## This is a stub function that will be replaced by ConfigManager
    return {}


def main():
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(
        description="Command line tool for automated writing evaluation (AWE) using different AI models."
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    ## Add commands from modules
    add_config_subparsers(subparsers)
    add_corpus_subparsers(subparsers)
    add_benchmark_subparsers(subparsers)

    args = parser.parse_args()

    ## Load Rater class
    rater = TextRater()
    
    ## Run command, if given
    if hasattr(args, 'func'):
        args.func(args, rater)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
