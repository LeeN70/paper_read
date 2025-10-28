"""MinerU API client for PDF parsing."""

import time
import requests
import zipfile
import os
import shutil
from typing import Tuple
from config import (
    MINERU_TOKEN,
    MINERU_SUBMIT_URL,
    MINERU_QUERY_URL_TEMPLATE,
    POLL_INTERVAL_SECONDS,
    TIMEOUT_SECONDS
)


class MinerUClient:
    """Client for interacting with MinerU API."""
    
    def __init__(self):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MINERU_TOKEN}"
        }
    
    def submit_task(self, pdf_url: str) -> str:
        """
        Submit a PDF parsing task to MinerU.
        
        Args:
            pdf_url: URL of the PDF file to parse
            
        Returns:
            task_id: The task ID for tracking the parsing job
            
        Raises:
            Exception: If submission fails
        """
        data = {
            "url": pdf_url,
            "is_ocr": True,
            "enable_formula": False,
        }
        
        print(f"Submitting task to MinerU for: {pdf_url}")
        response = requests.post(MINERU_SUBMIT_URL, headers=self.headers, json=data)
        
        if response.status_code != 200:
            raise Exception(f"Failed to submit task: {response.status_code} - {response.text}")
        
        result = response.json()
        if result.get("code") != 0:
            raise Exception(f"MinerU API error: {result.get('msg', 'Unknown error')}")
        
        task_id = result["data"]["task_id"]
        print(f"Task submitted successfully. Task ID: {task_id}")
        return task_id
    
    def poll_task_status(self, task_id: str) -> str:
        """
        Poll task status until completion or timeout.
        
        Args:
            task_id: The task ID to poll
            
        Returns:
            full_zip_url: URL of the result ZIP file
            
        Raises:
            Exception: If task fails or times out
        """
        url = MINERU_QUERY_URL_TEMPLATE.format(task_id)
        start_time = time.time()
        
        print("Polling task status...")
        while True:
            elapsed = time.time() - start_time
            if elapsed > TIMEOUT_SECONDS:
                raise Exception(f"Task timed out after {TIMEOUT_SECONDS} seconds")
            
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                raise Exception(f"Failed to query task: {response.status_code} - {response.text}")
            
            result = response.json()
            if result.get("code") != 0:
                raise Exception(f"MinerU API error: {result.get('msg', 'Unknown error')}")
            
            data = result["data"]
            state = data.get("state")
            
            if state == "done":
                full_zip_url = data.get("full_zip_url")
                print(f"Task completed! Result URL: {full_zip_url}")
                return full_zip_url
            elif state == "failed":
                err_msg = data.get("err_msg", "Unknown error")
                raise Exception(f"Task failed: {err_msg}")
            elif state in ["pending", "running", "converting"]:
                if state == "running" and "extract_progress" in data:
                    progress = data["extract_progress"]
                    print(f"  Progress: {progress.get('extracted_pages', 0)}/{progress.get('total_pages', '?')} pages")
                else:
                    print(f"  Status: {state}")
                time.sleep(POLL_INTERVAL_SECONDS)
            else:
                print(f"  Unknown state: {state}")
                time.sleep(POLL_INTERVAL_SECONDS)
    
    def download_and_extract_zip(self, zip_url: str, extract_to: str) -> Tuple[str, str]:
        """
        Download and extract the result ZIP file.
        
        Args:
            zip_url: URL of the ZIP file
            extract_to: Directory to extract files to
            
        Returns:
            Tuple of (full_md_path, images_dir_path)
            
        Raises:
            Exception: If download or extraction fails
        """
        print(f"Downloading result ZIP from: {zip_url}")
        
        # Create temporary directory for download
        os.makedirs(extract_to, exist_ok=True)
        zip_path = os.path.join(extract_to, "temp.zip")
        
        # Download ZIP file
        response = requests.get(zip_url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download ZIP: {response.status_code}")
        
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Extracting ZIP file...")
        
        # Extract ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # Remove ZIP file
        os.remove(zip_path)
        
        # Find full.md and images directory
        # MinerU extracts files directly in the extraction directory
        print(f"Extraction directory: {extract_to}")
        print(f"Contents of extraction directory:")
        all_items = os.listdir(extract_to)
        for item in all_items:
            item_path = os.path.join(extract_to, item)
            item_type = "DIR" if os.path.isdir(item_path) else "FILE"
            print(f"  [{item_type}] {item}")
        
        # full.md and images/ are directly in the extraction directory
        full_md_path = os.path.join(extract_to, "full.md")
        images_dir = os.path.join(extract_to, "images")
        
        if not os.path.exists(full_md_path):
            raise Exception(f"full.md not found at: {full_md_path}\nExtraction directory contents: {all_items}")
        
        print(f"Extraction complete. Found full.md at: {full_md_path}")
        return full_md_path, images_dir

