#!/bin/bash

# Script to review Python and Markdown files one by one
# Usage: ./review_files.sh

set -e

echo "Finding all Python (.py) and Markdown (.md) files in the git repository..."

# Get all tracked Python and Markdown files from git
files=$(git ls-files | grep -E '\.(py|md)$' | sort)

if [ -z "$files" ]; then
    echo "No Python or Markdown files found in the repository."
    exit 0
fi

# Convert to array to get accurate count
files_array=()
while IFS= read -r line; do
    files_array+=("$line")
done <<< "$files"

total_files=${#files_array[@]}
current_file=0

echo "Found $total_files files to review."
echo ""

# Iterate through each file
for file in "${files_array[@]}"; do
    current_file=$((current_file + 1))

    echo "========================================="
    echo "Reviewing file $current_file of $total_files:"
    echo "$file"
    echo "========================================="

    # Check if file exists (in case it was deleted)
    if [ ! -f "$file" ]; then
        echo "File does not exist (may have been deleted). Skipping..."
        echo ""
        continue
    fi

    # Open the file in VS Code
    echo "Opening file in VS Code..."
    code "$file"

    # Wait for user confirmation
    echo ""
    echo "Please review the file in VS Code."
    echo "Press Enter when you're done reviewing this file to continue to the next one..."
    echo "Or type 'quit' to exit the review process."

    read -r response

    if [ "$response" = "quit" ] || [ "$response" = "q" ]; then
        echo "Review process stopped by user."
        exit 0
    fi

    echo "Moving to next file..."
    echo ""

done

echo "========================================="
echo "All files have been reviewed!"
echo "Review process complete."
echo "========================================="
