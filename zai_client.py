"""Zai API client for PDF parsing."""

import time
import json
import requests
import tarfile
import os
from typing import Tuple
from config import (
    ZAI_BASE_URL,
    POLL_INTERVAL_SECONDS,
    TIMEOUT_SECONDS
)


class ZaiClient:
    """Client for interacting with Zai document parsing API."""
    
    def __init__(self):
        self.session = requests.Session()
        self.base_url = ZAI_BASE_URL
    
    def preupload(self) -> dict:
        """
        Step 1: Get pre-upload URL and UID.
        
        Returns:
            dict: Response with 'url' and 'uid' in data field
            
        Raises:
            Exception: If preupload fails
        """
        url = f"{self.base_url}/api/v2/preupload"
        response = self.session.get(url)
        
        if response.status_code != 200:
            raise Exception(f"Preupload failed: {response.status_code} - {response.text}")
        
        result = response.json()
        if result.get("code") != 0:
            raise Exception(f"Zai API error: {result.get('msg', 'Unknown error')}")
        
        return result
    
    def upload(self, upload_url: str, file_path: str):
        """
        Step 2: Upload file to the pre-upload URL.
        
        Args:
            upload_url: The URL obtained from preupload
            file_path: Local path to the PDF file
            
        Raises:
            Exception: If upload fails
        """
        with open(file_path, "rb") as f:
            response = self.session.put(upload_url, data=f)
        
        if response.status_code != 200:
            raise Exception(f"Upload failed: {response.status_code} - {response.text}")
    
    def parse(self, uid: str, doc_type: str = "pdf"):
        """
        Step 3: Trigger async parsing.
        
        Args:
            uid: The UID obtained from preupload
            doc_type: Document type (default: "pdf")
            
        Raises:
            Exception: If parse request fails
        """
        url = f"{self.base_url}/api/v2/convert/parse"
        data = {
            "uid": uid,
            "doc_type": doc_type,
        }
        response = self.session.post(url, data=json.dumps(data))
        
        if response.status_code != 200:
            raise Exception(f"Parse request failed: {response.status_code} - {response.text}")
        
        result = response.json()
        if result.get("code") != 0:
            raise Exception(f"Zai API error: {result.get('msg', 'Unknown error')}")
    
    def poll_result(self, uid: str) -> str:
        """
        Step 4: Poll for parsing result until completion.
        
        Args:
            uid: The UID of the parsing task
            
        Returns:
            str: URL of the result tar file
            
        Raises:
            Exception: If task fails or times out
        """
        url = f"{self.base_url}/api/v2/convert/result?uid={uid}"
        start_time = time.time()
        
        print("Polling Zai parsing status...")
        while True:
            elapsed = time.time() - start_time
            if elapsed > TIMEOUT_SECONDS:
                raise Exception(f"Task timed out after {TIMEOUT_SECONDS} seconds")
            
            response = self.session.get(url)
            
            if response.status_code != 200:
                raise Exception(f"Failed to query result: {response.status_code} - {response.text}")
            
            result = response.json()
            if result.get("code") != 0:
                raise Exception(f"Zai API error: {result.get('msg', 'Unknown error')}")
            
            data = result["data"]
            status = data.get("status")
            
            if status == "success":
                tar_url = data.get("url")
                page_count = data.get("page_count", "?")
                print(f"Parsing completed! Pages: {page_count}, Result URL: {tar_url}")
                return tar_url
            elif status == "failed":
                raise Exception(f"Parsing failed")
            else:
                print(f"  Status: {status}")
                time.sleep(POLL_INTERVAL_SECONDS)
    
    def download_and_extract_tar(self, tar_url: str, extract_to: str) -> Tuple[str, str]:
        """
        Download and extract the result tar file.
        
        Args:
            tar_url: URL of the tar file
            extract_to: Directory to extract files to
            
        Returns:
            Tuple of (res_md_path, imgs_dir_path)
            
        Raises:
            Exception: If download or extraction fails
        """
        print(f"Downloading result tar from: {tar_url}")
        
        # Create extraction directory
        os.makedirs(extract_to, exist_ok=True)
        tar_path = os.path.join(extract_to, "temp.tar")
        
        # Download tar file
        response = requests.get(tar_url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download tar: {response.status_code}")
        
        with open(tar_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Extracting tar file...")
        
        # Extract tar file
        with tarfile.open(tar_path, 'r') as tar_ref:
            tar_ref.extractall(extract_to)
        
        # Remove tar file
        os.remove(tar_path)
        
        # Find res.md and imgs directory
        print(f"Extraction directory: {extract_to}")
        print(f"Contents of extraction directory:")
        all_items = os.listdir(extract_to)
        for item in all_items:
            item_path = os.path.join(extract_to, item)
            item_type = "DIR" if os.path.isdir(item_path) else "FILE"
            print(f"  [{item_type}] {item}")
        
        # Zai outputs: res.md, res.txt, layout.json, and imgs/ directory
        res_md_path = os.path.join(extract_to, "res.md")
        imgs_dir = os.path.join(extract_to, "imgs")
        
        if not os.path.exists(res_md_path):
            raise Exception(f"res.md not found at: {res_md_path}\nExtraction directory contents: {all_items}")
        
        print(f"Extraction complete. Found res.md at: {res_md_path}")
        return res_md_path, imgs_dir
    
    def parse_document(self, local_pdf_path: str, output_dir: str) -> Tuple[str, str]:
        """
        Complete workflow: preupload -> upload -> parse -> poll result -> download and extract.
        
        Args:
            local_pdf_path: Path to local PDF file
            output_dir: Directory to extract results to
            
        Returns:
            Tuple of (markdown_path, images_dir_path)
        """
        print("=" * 70)
        print("Zai Parsing Workflow")
        print("=" * 70)
        
        # Step 1: Preupload
        print("\nStep 1: Getting upload URL...")
        pre_info = self.preupload()
        upload_url = pre_info["data"]["url"]
        uid = pre_info["data"]["uid"]
        print(f"UID: {uid}")
        
        # Step 2: Upload
        print("\nStep 2: Uploading PDF file...")
        self.upload(upload_url, local_pdf_path)
        print("Upload complete")
        
        # Step 3: Parse
        print("\nStep 3: Triggering async parsing...")
        self.parse(uid, doc_type="pdf")
        print("Parse request submitted")
        
        # Step 4: Poll result
        print("\nStep 4: Waiting for parsing to complete...")
        tar_url = self.poll_result(uid)
        
        # Step 5: Download and extract
        print("\nStep 5: Downloading and extracting results...")
        md_path, imgs_dir = self.download_and_extract_tar(tar_url, output_dir)
        
        print("\n" + "=" * 70)
        print("Zai Parsing Complete")
        print("=" * 70)
        
        return md_path, imgs_dir


def download_pdf(arxiv_url: str, download_path: str) -> str:
    """
    Download PDF from arXiv URL to local file.
    
    Args:
        arxiv_url: URL of the PDF on arXiv
        download_path: Local path to save the PDF
        
    Returns:
        str: Path to downloaded PDF file
        
    Raises:
        Exception: If download fails
    """
    print(f"Downloading PDF from: {arxiv_url}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    
    # Download with streaming
    response = requests.get(arxiv_url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download PDF: {response.status_code}")
    
    with open(download_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"PDF downloaded to: {download_path}")
    return download_path

