"""
Download HBN EEG data from OpenNeuro S3 (public, no auth needed).

Usage:
  # Download first 200 subjects (~enough for initial experiments)
  python download_hbn.py --n_subjects 200 --save_dir /home/share/data_makchen/peng/datasets/hbn

  # Download all (3000+ subjects, ~1TB, takes hours)
  python download_hbn.py --n_subjects 0 --save_dir /home/share/data_makchen/peng/datasets/hbn
"""

import argparse
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str,
                        default="/home/share/data_makchen/peng/datasets/hbn")
    parser.add_argument("--n_subjects", type=int, default=200,
                        help="Number of subjects to download. 0 = all.")
    parser.add_argument("--file_types", type=str, nargs="*",
                        default=[".set", ".fdt", ".edf", ".bdf", ".vhdr", ".eeg", ".vmrk"],
                        help="EEG file extensions to download")
    args = parser.parse_args()

    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'fcp-indi'
    prefix = 'data/Projects/HBN/BIDS_EEG/'

    print("Scanning S3 for subjects...")
    paginator = s3.get_paginator('list_objects_v2')

    # Discover all subjects and their EEG files
    subject_files = defaultdict(list)
    total_size = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            size = obj['Size']

            # Only EEG data files
            if not any(key.endswith(ext) for ext in args.file_types):
                continue

            # Extract subject ID (e.g., "sub-NDARAB123")
            parts = key.split('/')
            sub_id = None
            for part in parts:
                if part.startswith('sub-'):
                    sub_id = part
                    break
            if sub_id:
                subject_files[sub_id].append((key, size))
                total_size += size

    n_total = len(subject_files)
    print(f"Found {n_total} subjects with EEG files")
    print(f"Total EEG data size: {total_size / 1e9:.1f} GB")

    # Select subjects
    subjects = sorted(subject_files.keys())
    if args.n_subjects > 0:
        subjects = subjects[:args.n_subjects]

    selected_size = sum(
        size for sub in subjects for _, size in subject_files[sub]
    )
    selected_files = sum(len(subject_files[sub]) for sub in subjects)
    print(f"\nDownloading {len(subjects)} subjects: "
          f"{selected_files} files, {selected_size / 1e9:.1f} GB")

    # Download
    downloaded = 0
    for i, sub in enumerate(subjects):
        files = subject_files[sub]
        for key, size in files:
            # Preserve directory structure
            rel_path = key[len(prefix):]
            local_path = os.path.join(args.save_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            if os.path.exists(local_path) and os.path.getsize(local_path) == size:
                continue  # Skip already downloaded

            try:
                s3.download_file(bucket, key, local_path)
                downloaded += 1
            except Exception as e:
                print(f"  Error: {key}: {e}")

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(subjects)}] {sub} done ({downloaded} files downloaded)")

    print(f"\nDone. Downloaded {downloaded} new files to {args.save_dir}")


if __name__ == "__main__":
    main()
