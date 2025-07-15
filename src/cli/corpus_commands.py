"""
Corpus command implementations for AWEsomeScoring CLI.

This module contains all corpus-related command implementations.
"""

import os
import logging
from config import Config
from corpora import CorpusProcessor

logger = logging.getLogger("awesome_scoring")


def convert_corpus(args):
    """Convert corpus between formats."""
    try:
        # Load configuration
        config = Config(args, command="corpus:convert")
        
        # Use configuration values if CLI arguments are not provided
        input_path = args.input or config.input_corpus_path
        corpus_type = args.type or config.corpus_type
        
        if not input_path:
            raise ValueError("Input corpus file path is required")
            
        if not os.path.exists(input_path):
            raise ValueError(f"Input corpus file does not exist: {input_path}")
            
        if input_path.endswith('.csv'):
            corpus = CorpusProcessor.load_corpus_from_csv(input_path)
            output_path = args.output or "corpus_converted.txt"
            # No direct conversion method to text currently, just save as CSV
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
        # Load configuration
        config = Config(args, command="corpus:list")
        
        # Use configuration values if CLI arguments are not provided
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
