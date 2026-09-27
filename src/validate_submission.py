"""Submission Validator Wrapper for Amazon ML Entity Resolution.

Role: Person 3 (Evaluation, Validation & Submission Integrity)

Validates output/matching_results.tsv and output/candidate_pairs.tsv
against all official competition constraints before final upload.
"""

import argparse
import os
import subprocess
import sys


def validate_submission_files(
    matching_file: str = "output/matching_results.tsv",
    candidate_file: str = "output/candidate_pairs.tsv",
    test_dir: str = "dataset/test",
    check_ids: bool = False,
) -> bool:
    """Run official validation suite."""
    print("=" * 65)
    print("       AMAZON ML CHALLENGE -- SUBMISSION PRE-FLIGHT VALIDATOR       ")
    print("=" * 65)

    if not os.path.isfile(matching_file):
        print(f"[FAIL] Missing required output file: {matching_file}")
        return False

    has_candidates = os.path.isfile(candidate_file)
    if not has_candidates:
        print(f"[WARNING] Candidate file not found at: {candidate_file}")

    validator_script = os.path.join("utils", "validate_submission.py")
    if not os.path.isfile(validator_script):
        # Fallback to student_resource path
        validator_script = os.path.join("..", "6ab10eb3b23ba_student_resource", "student_resource", "utils", "validate_submission.py")

    cmd = [
        sys.executable,
        validator_script,
        "--matching", matching_file,
        "--test-dir", test_dir,
    ]
    if has_candidates:
        cmd.extend(["--candidate", candidate_file])
    if check_ids:
        cmd.append("--check-ids")

    print(f"Running validation checks via: {validator_script} ...\n")
    proc = subprocess.run(cmd)

    if proc.returncode == 0:
        print("\n" + "=" * 65)
        print("  >>> [PASS] ALL SUBMISSION CHECKS PASSED SUCCESSFULLY! <<<  ")
        print("  Your files are 100% compliant with competition constraints.")
        print("=" * 65 + "\n")
        return True
    else:
        print("\n" + "=" * 65)
        print("  >>> [FAIL] SUBMISSION VALIDATION FAILED! <<<  ")
        print("  Please fix the listed errors above before submitting.")
        print("=" * 65 + "\n")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate competition submission files.")
    parser.add_argument("--matching", "-m", default="output/matching_results.tsv", help="Path to matching_results.tsv")
    parser.add_argument("--candidate", "-c", default="output/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", "-t", default="dataset/test", help="Path to test dataset directory")
    parser.add_argument("--check-ids", action="store_true", help="Perform comprehensive ID existence check")
    args = parser.parse_args()

    success = validate_submission_files(
        matching_file=args.matching,
        candidate_file=args.candidate,
        test_dir=args.test_dir,
        check_ids=args.check_ids,
    )
    sys.exit(0 if success else 1)
