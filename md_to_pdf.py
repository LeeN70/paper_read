#!/usr/bin/env python3
"""
Markdown to PDF Converter
Converts detailed_breakdown.md and executive_summary.md files to PDF format
with embedded images and professional styling.
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import pypandoc
except ImportError:
    logger.error("pypandoc is not installed. Please install it: pip install pypandoc")
    logger.error("Also ensure pandoc is installed on your system.")
    sys.exit(1)


# Professional CSS styling for PDFs
PDF_CSS = """
<style>
body {
    font-family: 'Segoe UI', 'Arial', sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 210mm;
    margin: 0 auto;
    padding: 20mm;
    font-size: 11pt;
}

h1 {
    color: #2c3e50;
    font-size: 28pt;
    font-weight: 700;
    margin-top: 0;
    margin-bottom: 20pt;
    padding-bottom: 10pt;
    border-bottom: 3px solid #3498db;
}

h2 {
    color: #34495e;
    font-size: 20pt;
    font-weight: 600;
    margin-top: 24pt;
    margin-bottom: 12pt;
    padding-bottom: 6pt;
    border-bottom: 2px solid #95a5a6;
}

h3 {
    color: #2c3e50;
    font-size: 16pt;
    font-weight: 600;
    margin-top: 18pt;
    margin-bottom: 10pt;
}

h4 {
    color: #34495e;
    font-size: 14pt;
    font-weight: 600;
    margin-top: 14pt;
    margin-bottom: 8pt;
}

p {
    margin-bottom: 12pt;
    text-align: justify;
}

ul, ol {
    margin-bottom: 12pt;
    padding-left: 25pt;
}

li {
    margin-bottom: 6pt;
}

code {
    background-color: #f5f5f5;
    padding: 2pt 4pt;
    border-radius: 3pt;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 10pt;
    color: #c7254e;
}

pre {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 4pt;
    padding: 12pt;
    overflow-x: auto;
    margin-bottom: 16pt;
}

pre code {
    background-color: transparent;
    padding: 0;
    color: #333;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 16pt auto;
    border: 1px solid #ddd;
    border-radius: 4pt;
    padding: 4pt;
}

blockquote {
    border-left: 4px solid #3498db;
    padding-left: 16pt;
    margin-left: 0;
    margin-right: 0;
    color: #555;
    font-style: italic;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 16pt;
}

th, td {
    border: 1px solid #ddd;
    padding: 8pt 12pt;
    text-align: left;
}

th {
    background-color: #3498db;
    color: white;
    font-weight: 600;
}

tr:nth-child(even) {
    background-color: #f8f9fa;
}

strong {
    font-weight: 600;
    color: #2c3e50;
}

em {
    font-style: italic;
    color: #555;
}

a {
    color: #3498db;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

hr {
    border: none;
    border-top: 2px solid #e0e0e0;
    margin: 24pt 0;
}
</style>
"""


def find_markdown_files(base_dirs: List[str]) -> List[Path]:
    """
    Recursively find all detailed_breakdown.md and executive_summary.md files
    in the specified base directories.
    
    Args:
        base_dirs: List of base directory paths to search
        
    Returns:
        List of Path objects for found markdown files
    """
    markdown_files = []
    target_filenames = ['detailed_breakdown.md', 'executive_summary.md']
    
    for base_dir in base_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            logger.warning(f"Directory not found: {base_dir}")
            continue
            
        for filename in target_filenames:
            # Find all matching files recursively
            found_files = list(base_path.rglob(filename))
            markdown_files.extend(found_files)
            logger.info(f"Found {len(found_files)} {filename} file(s) in {base_dir}")
    
    return sorted(markdown_files)


def resolve_image_paths(md_content: str, md_file_path: Path) -> str:
    """
    Convert relative image paths in markdown to absolute paths.
    
    Args:
        md_content: Markdown file content
        md_file_path: Path to the markdown file
        
    Returns:
        Modified markdown content with absolute image paths
    """
    md_dir = md_file_path.parent.absolute()
    
    # Pattern to match markdown image syntax: ![alt text](./images/filename.jpg)
    pattern = r'!\[([^\]]*)\]\((\./images/[^\)]+)\)'
    
    def replace_path(match):
        alt_text = match.group(1)
        rel_path = match.group(2)
        
        # Remove leading './' if present
        rel_path = rel_path.lstrip('./')
        
        # Construct absolute path
        abs_path = md_dir / rel_path
        
        # Check if image exists
        if not abs_path.exists():
            logger.warning(f"Image not found: {abs_path}")
        
        # Convert to string with forward slashes (works on Windows too)
        abs_path_str = str(abs_path).replace('\\', '/')
        
        return f'![{alt_text}]({abs_path_str})'
    
    modified_content = re.sub(pattern, replace_path, md_content)
    return modified_content


def convert_md_to_pdf(md_file: Path) -> bool:
    """
    Convert a markdown file to PDF with embedded images and styling.
    
    Args:
        md_file: Path to the markdown file
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        logger.info(f"Processing: {md_file}")
        
        # Read markdown content
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Resolve image paths to absolute paths
        md_content = resolve_image_paths(md_content, md_file)
        
        # Add CSS styling to content
        styled_content = PDF_CSS + '\n\n' + md_content
        
        # Define output PDF path (same directory as markdown file)
        pdf_file = md_file.with_suffix('.pdf')
        
        # Convert to PDF using pypandoc
        # Using --pdf-engine=wkhtmltopdf for better HTML/CSS support
        extra_args = [
            '--pdf-engine=wkhtmltopdf',
            '--css=',  # Empty CSS to use inline styles
            '--standalone',
            '--self-contained',
            '-V', 'geometry:margin=2cm',
        ]
        
        try:
            pypandoc.convert_text(
                styled_content,
                'pdf',
                format='md',
                outputfile=str(pdf_file),
                extra_args=extra_args
            )
            logger.info(f"✓ Successfully created: {pdf_file}")
            return True
            
        except RuntimeError as e:
            # If wkhtmltopdf is not available, try with default engine
            error_msg = str(e).lower()
            if 'wkhtmltopdf' in error_msg or 'pdf-engine' in error_msg:
                logger.warning("wkhtmltopdf not found, trying default PDF engine...")
                
                extra_args = [
                    '--standalone',
                    '-V', 'geometry:margin=2cm',
                    '-V', 'colorlinks=true',
                ]
                
                pypandoc.convert_text(
                    styled_content,
                    'pdf',
                    format='md',
                    outputfile=str(pdf_file),
                    extra_args=extra_args
                )
                logger.info(f"✓ Successfully created: {pdf_file} (using default engine)")
                return True
            else:
                raise
                
    except Exception as e:
        logger.error(f"✗ Failed to convert {md_file}: {str(e)}")
        return False


def main():
    """Main execution function"""
    
    # Print header
    print("=" * 70)
    print("Markdown to PDF Converter")
    print("=" * 70)
    print()
    
    # Check pandoc installation
    try:
        pandoc_version = pypandoc.get_pandoc_version()
        logger.info(f"Pandoc version: {pandoc_version}")
    except Exception as e:
        logger.error("Pandoc is not installed or not found in PATH.")
        logger.error("Please install pandoc from: https://pandoc.org/installing.html")
        sys.exit(1)
    
    # Define directories to scan
    base_dirs = ['output_mineru', 'output_zai']
    
    # Find all markdown files
    logger.info("Scanning directories for markdown files...")
    markdown_files = find_markdown_files(base_dirs)
    
    if not markdown_files:
        logger.warning("No markdown files found!")
        return
    
    print()
    logger.info(f"Found {len(markdown_files)} markdown file(s) to convert")
    print()
    
    # Convert each file
    success_count = 0
    fail_count = 0
    
    for i, md_file in enumerate(markdown_files, 1):
        print(f"[{i}/{len(markdown_files)}] Converting {md_file.name}...")
        if convert_md_to_pdf(md_file):
            success_count += 1
        else:
            fail_count += 1
        print()
    
    # Print summary
    print("=" * 70)
    print("Conversion Summary")
    print("=" * 70)
    logger.info(f"✓ Successful: {success_count}")
    if fail_count > 0:
        logger.error(f"✗ Failed: {fail_count}")
    logger.info(f"Total: {len(markdown_files)}")
    print("=" * 70)


if __name__ == '__main__':
    main()

