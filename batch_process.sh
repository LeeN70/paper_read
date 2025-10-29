#!/bin/bash

# Batch processing script for paper-reader
# This script processes all PDF URLs from test.txt with both zai and mineru parsers

echo "======================================================================"
echo "Batch Paper Processing Script"
echo "======================================================================"
echo ""

# Define URLs array (extracted from test.txt)
declare -a URLS=(
    "https://arxiv.org/pdf/1706.03762.pdf"  # attention is all you need
    "https://arxiv.org/pdf/1810.04805.pdf"  # bert
    "https://arxiv.org/pdf/2203.02155.pdf"  # instruct gpt
    "https://arxiv.org/pdf/2402.03300.pdf"  # grpo
    "https://arxiv.org/pdf/2503.14476.pdf"  # dapo
    "https://arxiv.org/pdf/2507.18071.pdf"  # gspo
    "https://arxiv.org/pdf/2501.12948.pdf"  # deepseek r1
    "https://arxiv.org/pdf/2507.20534.pdf"  # kimi k2
    "https://arxiv.org/pdf/2508.06471.pdf"  # glm 4.5
    "https://arxiv.org/pdf/2210.03629.pdf"  # ReAct
)

declare -a PARSERS=("zai" "mineru")

# Initialize counters
TOTAL_COUNT=0
SUCCESS_COUNT=0
FAILURE_COUNT=0

# Arrays to store results
declare -a SUCCESS_LIST
declare -a FAILURE_LIST

# Get total number of tasks
TOTAL_TASKS=$((${#URLS[@]} * ${#PARSERS[@]}))

echo "Total papers to process: ${#URLS[@]}"
echo "Parsers to use: ${PARSERS[@]}"
echo "Total tasks: $TOTAL_TASKS"
echo ""
echo "======================================================================"
echo ""

# Process each parser with all URLs
for parser in "${PARSERS[@]}"; do
    echo "======================================================================"
    echo "Processing all papers with parser: $parser"
    echo "======================================================================"
    echo ""
    
    for url in "${URLS[@]}"; do
        # Extract paper ID from URL
        paper_id=$(basename "$url" .pdf)
        
        TOTAL_COUNT=$((TOTAL_COUNT + 1))
        
        echo "----------------------------------------------------------------------"
        echo "Task $TOTAL_COUNT/$TOTAL_TASKS"
        echo "Paper ID: $paper_id"
        echo "Parser: $parser"
        echo "URL: $url"
        echo "----------------------------------------------------------------------"
        
        # Run the command and capture the exit code
        python main.py "$url" --parser "$parser"
        exit_code=$?
        
        # Check if the command was successful
        if [ $exit_code -eq 0 ]; then
            echo "✓ SUCCESS: $paper_id with parser $parser"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            SUCCESS_LIST+=("$paper_id ($parser)")
        else
            echo "✗ FAILED: $paper_id with parser $parser (exit code: $exit_code)"
            FAILURE_COUNT=$((FAILURE_COUNT + 1))
            FAILURE_LIST+=("$paper_id ($parser) - exit code: $exit_code")
        fi
        
        echo ""
    done
    
    echo ""
    echo "Completed all papers with parser: $parser"
    echo ""
done

# Print final report
echo "======================================================================"
echo "BATCH PROCESSING COMPLETE"
echo "======================================================================"
echo ""
echo "Summary:"
echo "--------"
echo "Total tasks:    $TOTAL_COUNT"
echo "Successful:     $SUCCESS_COUNT"
echo "Failed:         $FAILURE_COUNT"
echo "Success rate:   $(awk "BEGIN {printf \"%.1f\", ($SUCCESS_COUNT/$TOTAL_COUNT)*100}")%"
echo ""

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "Successful tasks:"
    echo "-----------------"
    for item in "${SUCCESS_LIST[@]}"; do
        echo "  ✓ $item"
    done
    echo ""
fi

if [ $FAILURE_COUNT -gt 0 ]; then
    echo "Failed tasks:"
    echo "-------------"
    for item in "${FAILURE_LIST[@]}"; do
        echo "  ✗ $item"
    done
    echo ""
fi

echo "======================================================================"
echo "Log file: batch_process_$(date +%Y%m%d_%H%M%S).log"
echo "======================================================================"

# Exit with error if any task failed
if [ $FAILURE_COUNT -gt 0 ]; then
    exit 1
else
    exit 0
fi

