#!/usr/bin/env python3
"""
ArXiv Paper Reader - CLI tool for processing arXiv papers.

Usage:
    python main.py <arxiv_url> [--parser {mineru,zai}]
    
Example:
    python main.py https://arxiv.org/pdf/1706.03762.pdf
    python main.py https://arxiv.org/pdf/1706.03762.pdf --parser zai
"""

import sys
import os
import re
import asyncio
import shutil
import argparse
from mineru_client import MinerUClient
from zai_client import ZaiClient, download_pdf
from paper_processor import PaperProcessor
from config import (
    CACHE_MINERU_DIR, CACHE_ZAI_DIR,
    OUTPUT_MINERU_DIR, OUTPUT_ZAI_DIR
)


def extract_paper_id(arxiv_url: str) -> str:
    """
    Extract paper ID from arXiv URL.
    
    Args:
        arxiv_url: URL like https://arxiv.org/pdf/1706.03762.pdf
        
    Returns:
        Paper ID like "1706.03762"
    """
    # Match patterns like 1706.03762 or 2301.12345
    match = re.search(r'(\d{4}\.\d{4,5})', arxiv_url)
    if not match:
        raise ValueError(f"Could not extract paper ID from URL: {arxiv_url}")
    return match.group(1)


def validate_arxiv_url(url: str) -> bool:
    """Validate that the URL is an arXiv PDF URL."""
    if "arxiv.org" not in url.lower():
        return False
    if ".pdf" not in url.lower():
        return False
    return True


async def process_paper(arxiv_url: str, parser: str = "mineru"):
    """
    Main processing pipeline for a paper.
    
    Args:
        arxiv_url: URL of the arXiv PDF
        parser: Parser to use ('mineru' or 'zai')
    """
    print("=" * 70)
    print(f"ArXiv Paper Reader (Parser: {parser.upper()})")
    print("=" * 70)
    print()
    
    # Validate URL
    if not validate_arxiv_url(arxiv_url):
        raise ValueError("Invalid arXiv URL. Please provide a URL like: https://arxiv.org/pdf/1706.03762.pdf")
    
    # Extract paper ID
    paper_id = extract_paper_id(arxiv_url)
    print(f"Paper ID: {paper_id}")
    print()
    
    # Select parser-specific directories
    if parser == "mineru":
        cache_dir = CACHE_MINERU_DIR
        output_dir = OUTPUT_MINERU_DIR
    else:  # zai
        cache_dir = CACHE_ZAI_DIR
        output_dir = OUTPUT_ZAI_DIR
    
    # Create cache directory for this paper
    paper_cache_dir = os.path.join(cache_dir, paper_id)
    os.makedirs(paper_cache_dir, exist_ok=True)
    
    # Initialize paper processor
    paper_processor = PaperProcessor()
    
    # Parse document based on selected parser
    if parser == "mineru":
        # MinerU workflow
        mineru_client = MinerUClient()
        
        # Step 1: Submit to MinerU
        print("Step 1: Submitting to MinerU for PDF parsing")
        print("-" * 70)
        task_id = mineru_client.submit_task(arxiv_url)
        print()
        
        # Step 2: Poll for completion
        print("Step 2: Waiting for MinerU to complete parsing")
        print("-" * 70)
        zip_url = mineru_client.poll_task_status(task_id)
        print()
        
        # Step 3: Download and extract
        print("Step 3: Downloading and extracting parsed content")
        print("-" * 70)
        markdown_path, images_dir = mineru_client.download_and_extract_zip(zip_url, paper_cache_dir)
        print()
        
    else:  # zai
        # Zai workflow
        zai_client = ZaiClient()
        
        # Step 1: Download PDF from arXiv
        print("Step 1: Downloading PDF from arXiv")
        print("-" * 70)
        pdf_filename = f"{paper_id}.pdf"
        pdf_path = os.path.join(paper_cache_dir, pdf_filename)
        download_pdf(arxiv_url, pdf_path)
        print()
        
        # Steps 2-6: Zai parsing workflow (preupload -> upload -> parse -> poll -> download -> extract)
        print("Step 2: Running Zai parsing workflow")
        print("-" * 70)
        markdown_path, images_dir = zai_client.parse_document(pdf_path, paper_cache_dir)
        print()
    
    # Step 4: Generate summaries with Claude
    print(f"Step {'4' if parser == 'mineru' else '3'}: Generating summaries with Claude Agent SDK")
    print("-" * 70)
    output_paper_dir = os.path.join(output_dir, paper_id)
    exec_summary, detailed_breakdown = await paper_processor.generate_summaries(
        markdown_path,
        images_dir,
        output_paper_dir,
        parser=parser
    )
    print()
    
    # Success!
    print("=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print()
    print(f"Paper summaries generated successfully!")
    print()
    print(f"Output directory: {output_paper_dir}")
    print(f"  - Executive Summary: {exec_summary}")
    print(f"  - Detailed Breakdown: {detailed_breakdown}")
    print(f"  - Images: {os.path.join(output_paper_dir, 'images')}")
    print()
    print(f"Cached {parser.upper()} parsed content: {paper_cache_dir}")
    print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ArXiv Paper Reader - Process arXiv papers with different parsers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py https://arxiv.org/pdf/1706.03762.pdf
  python main.py https://arxiv.org/pdf/1706.03762.pdf --parser mineru
  python main.py https://arxiv.org/pdf/1706.03762.pdf --parser zai
        """
    )
    
    parser.add_argument(
        'arxiv_url',
        type=str,
        help='URL of the arXiv PDF to process'
    )
    
    parser.add_argument(
        '--parser',
        type=str,
        choices=['mineru', 'zai'],
        default='mineru',
        help='Document parser to use (default: mineru)'
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(process_paper(args.arxiv_url, args.parser))
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

