"""Configuration for the ArXiv Paper Reader project."""

import os

# MinerU API Configuration
MINERU_TOKEN = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI4MzYwMDI1OSIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc2MTUzNTA5OCwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiMWVjMTE4ZjktYzUyZi00Yjk2LWIwMWItNTM1ZTY5MWY1OGFkIiwiZW1haWwiOiIiLCJleHAiOjE3NjI3NDQ2OTh9.VajZQZZj1qlL7ZddFEcvvzR6W9YUKF7qB3_n7U3ynwlnccSxLF1fgJL9Oh1xwTinQkqvaPvVjiYGZCm2f7J3vQ"
MINERU_BASE_URL = "https://mineru.net/api/v4"
MINERU_SUBMIT_URL = f"{MINERU_BASE_URL}/extract/task"
MINERU_QUERY_URL_TEMPLATE = f"{MINERU_BASE_URL}/extract/task/{{}}"

# Zai API Configuration
ZAI_BASE_URL = "http://10.243.65.197:12004"

# Polling Configuration
POLL_INTERVAL_SECONDS = 10
TIMEOUT_SECONDS = 1800  # 30 minutes

# Directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")

# Legacy directories (for backward compatibility)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

# Parser-specific directories
CACHE_MINERU_DIR = os.path.join(PROJECT_ROOT, "cache_mineru")
CACHE_ZAI_DIR = os.path.join(PROJECT_ROOT, "cache_zai")
OUTPUT_MINERU_DIR = os.path.join(PROJECT_ROOT, "output_mineru")
OUTPUT_ZAI_DIR = os.path.join(PROJECT_ROOT, "output_zai")

# Claude SDK Configuration
CLAUDE_ALLOWED_TOOLS = ["Read", "Write", "Edit", "Grep", "Glob", "WebSearch", "Bash"]
CLAUDE_PERMISSION_MODE = "acceptEdits"

