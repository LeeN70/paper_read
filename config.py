"""Configuration for the ArXiv Paper Reader project."""

import os

# MinerU API Configuration
MINERU_TOKEN = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI4MzYwMDI1OSIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc2MTUzNTA5OCwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiMWVjMTE4ZjktYzUyZi00Yjk2LWIwMWItNTM1ZTY5MWY1OGFkIiwiZW1haWwiOiIiLCJleHAiOjE3NjI3NDQ2OTh9.VajZQZZj1qlL7ZddFEcvvzR6W9YUKF7qB3_n7U3ynwlnccSxLF1fgJL9Oh1xwTinQkqvaPvVjiYGZCm2f7J3vQ"
MINERU_BASE_URL = "https://mineru.net/api/v4"
MINERU_SUBMIT_URL = f"{MINERU_BASE_URL}/extract/task"
MINERU_QUERY_URL_TEMPLATE = f"{MINERU_BASE_URL}/extract/task/{{}}"

# Polling Configuration
POLL_INTERVAL_SECONDS = 10
TIMEOUT_SECONDS = 1800  # 30 minutes

# Directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

# Claude SDK Configuration
CLAUDE_ALLOWED_TOOLS = ["Read", "Write", "Edit", "Grep", "Glob"]
CLAUDE_PERMISSION_MODE = "acceptEdits"

