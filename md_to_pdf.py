#!/usr/bin/env python3
"""
Markdown to PDF Converter for Paper Reader Output

This script converts markdown files (detailed_breakdown.md, executive_summary.md, 
relevant_code.md) in output_zai directory to PDF format, preserving images.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import markdown
    from markdown.extensions import codehilite, tables, fenced_code
    from weasyprint import HTML, CSS
except ImportError as e:
    print("ERROR: Required packages not installed.")
    print("Please install: pip install markdown weasyprint pygments")
    sys.exit(1)


# CSS styles for PDF output
PDF_CSS = """
@page {
    size: A4;
    margin: 2cm 1.5cm;
}

body {
    font-family: 'DejaVu Sans', 'Arial', sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #333;
    max-width: 100%;
}

h1 {
    font-size: 24pt;
    margin-top: 1em;
    margin-bottom: 0.5em;
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 0.3em;
}

h2 {
    font-size: 18pt;
    margin-top: 1.2em;
    margin-bottom: 0.5em;
    color: #34495e;
}

h3 {
    font-size: 14pt;
    margin-top: 1em;
    margin-bottom: 0.4em;
    color: #555;
}

h4 {
    font-size: 12pt;
    margin-top: 0.8em;
    margin-bottom: 0.3em;
    color: #666;
}

p {
    margin: 0.8em 0;
    text-align: justify;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1em auto;
    border: 1px solid #ddd;
    border-radius: 4px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

code {
    font-family: 'DejaVu Sans Mono', 'Courier New', monospace;
    font-size: 10pt;
    background-color: #f5f5f5;
    padding: 2px 4px;
    border-radius: 3px;
}

pre {
    background-color: #f8f8f8;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 1em;
    overflow-x: auto;
    margin: 1em 0;
}

pre code {
    background-color: transparent;
    padding: 0;
    font-size: 9pt;
}

blockquote {
    border-left: 4px solid #3498db;
    margin: 1em 0;
    padding-left: 1em;
    color: #666;
    font-style: italic;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
}

table th,
table td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}

table th {
    background-color: #3498db;
    color: white;
    font-weight: bold;
}

table tr:nth-child(even) {
    background-color: #f9f9f9;
}

ul, ol {
    margin: 1em 0;
    padding-left: 2em;
}

li {
    margin: 0.4em 0;
}

strong {
    font-weight: bold;
    color: #2c3e50;
}

em {
    font-style: italic;
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
    border-top: 1px solid #ddd;
    margin: 2em 0;
}
"""


def convert_markdown_to_html(md_content: str, base_dir: Path) -> str:
    """
    Convert markdown content to HTML, handling image paths.
    
    Args:
        md_content: Markdown content as string
        base_dir: Base directory for resolving relative image paths
        
    Returns:
        HTML content as string
    """
    # Convert relative image paths to absolute paths
    def replace_image_path(match):
        alt_text = match.group(1)  # Alt text
        img_path = match.group(2)   # Image path
        
        # Handle relative paths
        if img_path.startswith('./'):
            img_path = img_path[2:]
        elif img_path.startswith('/'):
            # Absolute path, use as is
            abs_path = Path(img_path)
            if abs_path.exists():
                return f'![{alt_text}]({abs_path.as_uri()})'
            else:
                return match.group(0)
        
        # Convert to absolute path relative to base_dir
        abs_path = base_dir / img_path
        if abs_path.exists():
            return f'![{alt_text}]({abs_path.as_uri()})'
        else:
            # Keep original if file doesn't exist
            return match.group(0)
    
    # Replace image paths with absolute file:// URIs
    pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    md_content = re.sub(pattern, replace_image_path, md_content)
    
    # Convert markdown to HTML
    md = markdown.Markdown(
        extensions=[
            'codehilite',
            'fenced_code',
            'tables',
            'nl2br',
            'sane_lists'
        ],
        extension_configs={
            'codehilite': {
                'css_class': 'highlight',
                'use_pygments': True,
            }
        }
    )
    
    html_body = md.convert(md_content)
    
    # Wrap in full HTML document
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        {PDF_CSS}
    </style>
</head>
<body>
{html_body}
</body>
</html>"""
    
    return html


def convert_md_to_pdf(md_path: Path, pdf_path: Path) -> bool:
    """
    Convert a markdown file to PDF.
    
    Args:
        md_path: Path to markdown file
        pdf_path: Path to output PDF file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read markdown file
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Get base directory for resolving image paths
        base_dir = md_path.parent.absolute()
        
        # Convert markdown to HTML
        html_content = convert_markdown_to_html(md_content, base_dir)
        
        # Convert HTML to PDF
        HTML(string=html_content, base_url=str(base_dir)).write_pdf(
            pdf_path,
            stylesheets=[CSS(string=PDF_CSS)]
        )
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to convert {md_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_markdown_files(output_dir: Path) -> List[Tuple[Path, str]]:
    """
    Find all target markdown files in output_zai directory.
    
    Args:
        output_dir: Path to output_zai directory
        
    Returns:
        List of tuples (md_path, file_type)
    """
    target_files = [
        'detailed_breakdown.md',
        'executive_summary.md',
        'relevant_code.md'
    ]
    
    found_files = []
    
    # Walk through all subdirectories
    for paper_dir in output_dir.iterdir():
        if not paper_dir.is_dir():
            continue
        
        # Check for each target file
        for target_file in target_files:
            md_path = paper_dir / target_file
            if md_path.exists():
                found_files.append((md_path, target_file.replace('.md', '')))
    
    return found_files


def main():
    """Main function to convert markdown files to PDF."""
    print("=" * 70)
    print("Markdown to PDF Converter")
    print("=" * 70)
    print()
    
    # Get output_zai directory
    script_dir = Path(__file__).parent
    output_zai_dir = script_dir / 'output_zai'
    
    if not output_zai_dir.exists():
        print(f"ERROR: Directory not found: {output_zai_dir}")
        sys.exit(1)
    
    print(f"INFO: Scanning directory: {output_zai_dir}")
    
    # Find all markdown files
    markdown_files = find_markdown_files(output_zai_dir)
    
    if not markdown_files:
        print("INFO: No markdown files found to convert.")
        return
    
    print(f"INFO: Found {len(markdown_files)} markdown file(s) to convert")
    print()
    
    # Convert each file
    successful = 0
    failed = 0
    
    for idx, (md_path, file_type) in enumerate(markdown_files, 1):
        pdf_path = md_path.with_suffix('.pdf')
        
        print(f"[{idx}/{len(markdown_files)}] Converting {file_type}.md...")
        print(f"INFO: Processing: {md_path}")
        
        if convert_md_to_pdf(md_path, pdf_path):
            print(f"INFO: ✓ Successfully created: {pdf_path}")
            successful += 1
        else:
            print(f"INFO: ✗ Failed to convert: {md_path}")
            failed += 1
        
        print()
    
    # Print summary
    print("=" * 70)
    print("Conversion Summary")
    print("=" * 70)
    print(f"INFO: ✓ Successful: {successful}")
    if failed > 0:
        print(f"INFO: ✗ Failed: {failed}")
    print(f"INFO: Total: {len(markdown_files)}")
    print("=" * 70)


if __name__ == '__main__':
    main()

