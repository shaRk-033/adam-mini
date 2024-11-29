import os
from pathlib import Path
import numpy as np

def combine_bin_files(input_pattern, output_file, data_dir, chunk_size=1024*1024):
    input_files = sorted(Path(data_dir).glob(input_pattern))

    if not input_files:
        raise FileNotFoundError(f"No files found for pattern: {input_pattern}")

    print(f"Combining {len(input_files)} files into {output_file}...")

    with open(output_file, 'wb') as outfile:
        for file_path in input_files:
            print(f"  Processing {file_path.name} (skipping header)...")
            with open(file_path, 'rb') as infile:
                infile.seek(256 * 4)
                while True:
                    chunk = infile.read(chunk_size)
                    if not chunk:
                        break
                    outfile.write(chunk)
    print(f"Successfully created {output_file}.\n")

def main():
    data_dir = "./fineweb10B"

    datasets = [
        {
            "input_pattern": "fineweb_train_*.bin",
            "output_file": os.path.join(data_dir, "train.bin")
        },
        {
            "input_pattern": "fineweb_val_*.bin",
            "output_file": os.path.join(data_dir, "val.bin")
        }
    ]

    for dataset in datasets:
        combine_bin_files(
            input_pattern=dataset["input_pattern"],
            output_file=dataset["output_file"],
            data_dir=data_dir
        )

if __name__ == "__main__":
    main()
