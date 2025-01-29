#!/bin/bash

# Define variables
data_dir="data"
zip_url="https://zenodo.org/records/8417503/files/ocelot2023_v1.0.1.zip?download=1"
zip_file="$data_dir/ocelot2023_v1.0.1.zip"

# Create the data directory if it doesn't exist
if [ ! -d "$data_dir" ]; then
    echo "Creating directory: $data_dir"
    mkdir "$data_dir"
else
    echo "Directory $data_dir already exists."
fi

# Download the zip file to the data directory
if [ ! -f "$zip_file" ]; then
    echo "Downloading file from $zip_url to $data_dir"
    curl -L -o "$zip_file" "$zip_url"
else
    echo "File $zip_file already exists. Skipping download."
fi

# Extract the zip file
if [ -f "$zip_file" ]; then
    echo "Extracting $zip_file into $data_dir"
    unzip -o "$zip_file" -d "$data_dir"
    echo "Extraction completed."
else
    echo "Error: Zip file $zip_file not found."
    exit 1
fi

# List filenames in the directory and save them to data.csv for easy indexing later w/ dataloader
echo "Saving filenames to csv indexing files..."
ls "$data_dir/ocelot2023_v1.0.1/images/train/cell/" > "$data_dir/ocelot2023_v1.0.1/train_data.csv"
ls "$data_dir/ocelot2023_v1.0.1/images/val/cell/" > "$data_dir/ocelot2023_v1.0.1/val_data.csv"
ls "$data_dir/ocelot2023_v1.0.1/images/test/cell/" > "$data_dir/ocelot2023_v1.0.1/test_data.csv"

echo "Filenames have been saved to csv in data/ocelot2023_v1.0.1/ directory."