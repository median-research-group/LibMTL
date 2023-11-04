#!/bin/bash

# Dropbox link provided
dropbox_link="https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0"

# Modify the Dropbox link to force download by changing dl=0 to dl=1
download_link="${dropbox_link/dl=0/dl=1}"

# Get the current directory path
current_dir="$(pwd)"

# The 'data' directory within the current directory
data_dir="${current_dir}/data"

# Check if the 'data' directory exists, if not, create it
if [ ! -d "$data_dir" ]; then
    mkdir -p "$data_dir"
fi

# Download the Dropbox content into the 'data' directory
wget -O "${data_dir}/dropbox_content.zip" "$download_link" || curl -L "$download_link" -o "${data_dir}/dropbox_content.zip"

# Extract the contents of the downloaded file
unzip -o "${data_dir}/dropbox_content.zip" -d "$data_dir"

# List the contents of the 'data' directory
ls -l "$data_dir"
