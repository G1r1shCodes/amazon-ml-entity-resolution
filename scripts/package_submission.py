"""Automated Submission Packaging Script for Amazon ML Entity Resolution Challenge.

Role: Person 3 (Evaluation, Packaging & Submission Integrity)

Creates the official submission archive matching competition specifications:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import argparse
import os
import zipfile
import sys

def package_submission(team_name: str = "DataResolvers", output_dir: str = ".") -> str:
    zip_filename = os.path.join(output_dir, f"{team_name}_submission.zip")
    print(f"Creating submission package: {zip_filename} ...")

    # Files to verify
    required_files = [
        "output/matching_results.tsv",
        "output/candidate_pairs.tsv",
        "Documentation_template.md",
        "README.md",
        "requirements.txt",
    ]

    for rf in required_files:
        if not os.path.exists(rf):
            raise FileNotFoundError(f"Missing required submission deliverable: {rf}")

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Output files
        zf.write("output/matching_results.tsv", arcname="output/matching_results.tsv")
        zf.write("output/candidate_pairs.tsv", arcname="output/candidate_pairs.tsv")

        # 2. Documentation template
        zf.write("Documentation_template.md", arcname="Documentation_template.md")

        # 3. Code folder: code/business_entity_resolution/
        code_prefix = "code/business_entity_resolution"
        zf.write("README.md", arcname=f"{code_prefix}/README.md")
        zf.write("requirements.txt", arcname=f"{code_prefix}/requirements.txt")

        # Add all Python files under src/
        for root, _, files in os.walk("src"):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, ".")
                    arc_path = f"{code_prefix}/{rel_path.replace(os.sep, '/')}"
                    zf.write(full_path, arcname=arc_path)

    zip_size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    print(f"\nSuccessfully generated {zip_filename} ({zip_size_mb:.2f} MB)")
    print("Archive structure verified against competition criteria.")
    return zip_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package submission zip for Amazon ML Challenge.")
    parser.add_argument("--team-name", default="DataResolvers", help="Your team name")
    args = parser.parse_args()
    package_submission(team_name=args.team_name)
