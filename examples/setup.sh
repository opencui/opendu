#!/bin/bash
set -x
#
# This shell can expand the tar ball, and copy the extracted dumeta to right place, and run curl to index it.
# Check if a filename is provided
if [ $# -eq 0 ]; then
    echo "Please provide, prefix (me.text/ai.bethere) and  a tar.gz filename as an argument."
    exit 1
fi

# Create a temporary directory
temp_dir=$(mktemp -d)

# Extract the tar.gz file to the temporary directory
prefix="$1"
tar -xzf "$2" -C "$temp_dir"

files=$(find "$temp_dir/en" -type f -exec grep -l "IChatbot()" {} +)

echo "Files found (simple method):"
echo "$files"

file_count=$(echo "$files" | wc -l)

# Test if there's exactly one match
if [ "$file_count" -ne 1 ]; then
    echo "Error: Number of files found is not 1."
    echo "Actual number of files found: $file_count"
    exit 1
fi

filename=$(basename "${files[0]}")
IFS='_' read -ra parts1 <<<"$filename"
last_element="${parts1[@]: -1}"
filename_without_ext="${last_element%.kt}"

path="${prefix}_${filename_without_ext}_en_746395988637257728"

mkdir -p "$path"

echo "Creating $path"

cp "$temp_dir/en/dumeta"/* "$path"

url="127.0.0.1:3001/v1/index/${path}"
echo "curl $url"
curl "$url"

# Clean up: remove the temporary directory
rm -rf "$temp_dir"

echo "Processing complete"
