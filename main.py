"""
main.py
-------
Entry point for the full KDE extraction and analysis pipeline.

Usage:
    python3 main.py <pdf1> <pdf2> [--zip <project-yamls.zip>]

Steps:
    1. (Task 1) Extract KDEs from both PDFs using all three prompt strategies.
    2. (Task 2) Compare the two KDE YAML outputs for differences.
    3. (Task 3) Map differences to Kubescape controls and run a scan.
"""

import argparse
import os
import sys
import torch
from transformers import pipeline

from task1 import process_two_files
from task2 import (
    load_yaml_outputs,
    compare_element_names,
    compare_elements_and_requirements,
)
from task3 import (
    load_text_outputs,
    map_differences_to_kubescape_controls,
    run_kubescape,
    generate_csv,
)


OUTPUT_DIR = "kde_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KDE extraction, comparison, and Kubescape scanning pipeline."
    )
    parser.add_argument("pdf1", help="Path to the first PDF file.")
    parser.add_argument("pdf2", help="Path to the second PDF file.")
    parser.add_argument(
        "--zip",
        default="project-yamls.zip",
        metavar="ZIP",
        help="Path to the Kubernetes manifest ZIP file (default: project-yamls.zip).",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        metavar="DIR",
        help=f"Directory for all output files (default: {OUTPUT_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    for path, label in [(args.pdf1, "pdf1"), (args.pdf2, "pdf2")]:
        if not os.path.isfile(path):
            print(f"[ERROR] {label} not found: '{path}'")
            sys.exit(1)

    if not os.path.isfile(args.zip):
        print(f"[WARNING] Manifest ZIP not found: '{args.zip}'. "
              "Task 3 scan will be skipped.")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # TASK 1 — KDE extraction
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TASK 1: Extracting Key Data Elements from PDFs")
    print("=" * 60)

    print("Loading model (this may take a moment)…")
    pipe = pipeline(
        "text-generation",
        model="google/gemma-3-1b-it",
        device="cpu",
        dtype=torch.bfloat16,
    )
    print("Model loaded.\n")

    result_map = process_two_files(
        pdf_path_1=args.pdf1,
        pdf_path_2=args.pdf2,
        pipe=pipe,
        output_dir=args.output_dir,
    )

    yaml_path_1, yaml_path_2 = list(result_map.values())
    print(f"\nTask 1 complete:")
    print(f"  {args.pdf1} → {yaml_path_1}")
    print(f"  {args.pdf2} → {yaml_path_2}")

    # ------------------------------------------------------------------
    # TASK 2 — KDE comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TASK 2: Comparing KDE YAML outputs")
    print("=" * 60)

    # load_yaml_outputs discovers the files we just wrote
    discovered_1, discovered_2 = load_yaml_outputs(args.output_dir)

    name_diff_path = os.path.join(args.output_dir, "name_diff.txt")
    full_diff_path = os.path.join(args.output_dir, "full_diff.txt")

    compare_element_names(discovered_1, discovered_2, out_path=name_diff_path)
    compare_elements_and_requirements(discovered_1, discovered_2, out_path=full_diff_path)

    print(f"\nTask 2 complete:")
    print(f"  Name differences  → {name_diff_path}")
    print(f"  Full differences  → {full_diff_path}")

    # ------------------------------------------------------------------
    # TASK 3 — Kubescape mapping and scan
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TASK 3: Kubescape control mapping and scan")
    print("=" * 60)

    name_diff_path, full_diff_path = load_text_outputs(args.output_dir)

    controls_path = os.path.join(args.output_dir, "kubescape_controls.txt")
    map_differences_to_kubescape_controls(
        name_diff_path, full_diff_path, out_path=controls_path
    )
    print(f"  Controls file     → {controls_path}")

    if not os.path.isfile(args.zip):
        print(f"\n[SKIP] Kubescape scan skipped — ZIP not found: '{args.zip}'")
        print("       Provide the manifest archive with --zip <path> to run the scan.")
        return

    df = run_kubescape(controls_path, args.zip)

    csv_path = os.path.join(args.output_dir, "kubescape_results.csv")
    generate_csv(df, out_path=csv_path)

    print(f"\nTask 3 complete:")
    print(f"  Scan results      → {csv_path}")
    print(f"  Rows in report    : {len(df)}")

    print("\n" + "=" * 60)
    print("Pipeline complete. All outputs written to:", args.output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
