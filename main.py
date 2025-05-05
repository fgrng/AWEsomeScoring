#!/usr/bin/env python3
"""
AWEsomeScoring

A command line tool for automated writing evaluation (AWE) using different AI models.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Command line tool for automated writing evaluation (AWE) using different AI models.")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Add commands from modules
    add_config_subparsers(subparsers)
    add_corpus_subparsers(subparsers)
    add_benchmark_subparsers(subparsers)

    args = parser.parse_args()

    # Load config
    config = load_config()
    
    # Load Rater class
    rater = TextRater(config.get("rater","benchmark"))
    
    # Run command, if given.
    if hasattr(args, 'func'):
        args.func(args, rater)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

