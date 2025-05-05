import csv
import logging
import os
import re
from typing import Dict, List, Optional

logger = logging.getLogger("awesome_scoring")

class CorpusProcessor:
    """
    Process text corpus files and convert between formats.
    
    This class handles loading and saving text corpora in different formats,
    as well as cleaning and preprocessing the text.
    """
    
    @staticmethod
    def load_corpus(file_path: str, corpus_type: str = None) -> Dict[str, str]:
        """
        Load a text corpus from a file.
        
        Automatically detects file format based on extension and calls
        the appropriate loading method.
        
        Args:
            file_path: Path to the corpus file
            corpus_type: Type of corpus ('basch_narrative' or 'basch_instructive'), 
                         required for text files
                         
        Returns:
            Dictionary mapping student IDs to text content
        """
        if file_path.endswith('.csv'):
            return CorpusProcessor.load_corpus_from_csv(file_path)
        else:
            if not corpus_type:
                raise ValueError("Corpus type is required for text files")
            return CorpusProcessor.load_corpus_from_txt(file_path, corpus_type)
    
    @staticmethod
    def load_corpus_from_txt(file_path: str, corpus_type: str) -> Dict[str, str]:
        """
        Load a text corpus from a text file.
        
        Args:
            file_path: Path to the text file
            corpus_type: Type of corpus ('basch_narrative' or 'basch_instructive')
            
        Returns:
            Dictionary mapping student IDs to text content
        """
        logger.info(f"Reading corpus from {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        ## Perform common text cleanup
        content = CorpusProcessor._clean_text(content)
        
        ## Process based on corpus type
        if corpus_type == "basch_narrative":
            return CorpusProcessor._parse_basch_narrative_corpus(content)    
        elif corpus_type == "basch_instructive":
            return CorpusProcessor._parse_basch_instructive_corpus(content)
        else:
            raise ValueError(f"Unknown corpus type: {corpus_type}")
    
    @staticmethod
    def _clean_text(content: str) -> str:
        """
        Perform common text cleanup operations.
        
        Args:
            content: Raw text content
            
        Returns:
            Cleaned text content
        """
        ## Clean up the content
        content = re.sub(r"\r", r"", content)
        content = re.sub(r"\((\d\d\d\d)\)", r"\1", content)
        content = re.sub(r"\((\d\d\d\d\_.)\)", r"\1", content)
        content = re.sub(r"\n\s+(\d\d\d\d)", r"\n\1", content)
        content = re.sub(r"\n\s+(\d\d\d\d\_.)", r"\n\1", content)
        content = re.sub(r"(\d\d\d\d)\s+\n", r"\1\n", content)
        content = re.sub(r"(\d\d\d\d\_.)\s+\n", r"\1\n", content)
        content = re.sub("<unleserlich>", "[?]", content)
        
        return content
    
    @staticmethod
    def _parse_basch_narrative_corpus(content: str) -> Dict[str, str]:
        """
        Parse a basch_narrative corpus from cleaned text content.
        
        Args:
            content: Cleaned text content
            
        Returns:
            Dictionary mapping student IDs to text content
        """
        ## Prepare first student id for split action
        content = re.sub(r"6001\n", r"\n6001\n", content)
        
        ## Split corpus texts (results in first being the empty string)
        parts = re.split(r'\n(\d\d\d\d)\n', content)
        
        ## Structured object of texts
        corpus = {}
        for key, value in zip(parts[1::2], parts[2::2]):
            corpus[key] = value
        
        logger.info(f"Loaded {len(corpus)} texts from basch_narrative corpus")
        return corpus
    
    @staticmethod
    def _parse_basch_instructive_corpus(content: str) -> Dict[str, str]:
        """
        Parse an basch_instructive corpus from cleaned text content.
        
        Args:
            content: Cleaned text content
            
        Returns:
            Dictionary mapping student IDs to text content
        """
        ## Split corpus texts
        parts = re.split("[\n]###NEWTEXT###\n", content)
        
        ## Structured object of texts
        corpus = {}
        for t in parts:
            if not t.strip():
                continue
            key = t.partition('\n')[0]
            value = t.partition('\n')[2:][0]
            corpus[key] = value.rstrip()
        
        logger.info(f"Loaded {len(corpus)} texts from basch_instructive corpus")
        return corpus
    
    @staticmethod
    def save_corpus_to_csv(corpus: Dict[str, str], output_path: str) -> None:
        """
        Save a corpus to a CSV file.
        
        Args:
            corpus: Dictionary mapping student IDs to text content
            output_path: Path to save the CSV file
        """
        logger.info(f"Writing corpus to {output_path}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            csv_writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            
            ## Write header row
            csv_writer.writerow(["student_id", "text"])
            
            ## Write data rows
            for student_id, text in corpus.items():
                csv_writer.writerow([student_id, text])
                
        logger.info(f"Saved {len(corpus)} texts to CSV")
    
    @staticmethod
    def load_corpus_from_csv(file_path: str) -> Dict[str, str]:
        """
        Load a corpus from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary mapping student IDs to text content
        """
        logger.info(f"Reading corpus from CSV {file_path}")
        
        corpus = {}
        with open(file_path, "r", encoding="utf-8") as f:
            csv_reader = csv.reader(f)
            
            ## Check for header row
            has_header = csv.Sniffer().has_header(f.read(1024))
            f.seek(0)
            
            if has_header:
                ## Skip header row
                next(csv_reader)
            
            for row in csv_reader:
                if len(row) >= 2:
                    corpus[row[0]] = row[1]
                    
        logger.info(f"Loaded {len(corpus)} texts from CSV")
        return corpus
