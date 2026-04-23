"""
task3.py
--------
Functions
---------
load_text_outputs(output_dir)
    Discover the two TEXT files produced by kde_compare.py (Task 2).

map_differences_to_kubescape_controls(name_diff_path, full_diff_path, out_path)
    Determine whether differences exist and, if so, map them to Kubescape
    control IDs.  Writes a TEXT file with either 'NO DIFFERENCES FOUND' or
    one control ID per line.

run_kubescape(controls_path, zip_path)
    Execute Kubescape from the command line.  Runs only the controls listed
    in controls_path, or all controls when the file contains
    'NO DIFFERENCES FOUND'.  Returns a pandas DataFrame of scan results.

generate_csv(df, out_path)
    Write the DataFrame to a CSV file with the agreed column headers.
"""

import os
import re
import glob
import json
import subprocess
import tempfile
import pandas as pd


# =============================================================================
# KUBESCAPE CONTROL MAPPING TABLE
# Keys are lower-case keyword fragments that may appear in KDE names or
# requirements; values are the Kubescape control IDs they map to.
# Reference: https://hub.armosec.io/docs/controls
# =============================================================================
KEYWORD_TO_CONTROLS: dict[str, list[str]] = {
    # Authentication / authorisation
    "anonymous":              ["C-0069"],
    "authorization-mode":     ["C-0069"],
    "authorization mode":     ["C-0069"],
    "alwaysallow":            ["C-0069"],
    "always allow":           ["C-0069"],
    "client ca":              ["C-0070"],
    "client-ca":              ["C-0070"],

    # Kubelet hardening
    "kubelet":                ["C-0069", "C-0070"],
    "read-only-port":         ["C-0069"],
    "read only port":         ["C-0069"],
    "streaming-connection":   ["C-0069"],
    "protect-kernel":         ["C-0069"],
    "iptables":               ["C-0069"],
    "hostname-override":      ["C-0069"],
    "eventrecordqps":         ["C-0069"],
    "rotate-certificates":    ["C-0070"],
    "rotatekubeletserver":    ["C-0070"],
    "kubeconfig":             ["C-0070"],

    # Audit logging
    "audit":                  ["C-0067"],
    "audit log":              ["C-0067"],
    "audit-log":              ["C-0067"],

    # RBAC
    "rbac":                   ["C-0035", "C-0036", "C-0058"],
    "cluster-admin":          ["C-0035"],
    "cluster admin":          ["C-0035"],
    "clusterrole":            ["C-0035", "C-0036"],
    "wildcard":               ["C-0036"],
    "create pods":            ["C-0058"],
    "create pod":             ["C-0058"],

    # Service accounts
    "service account":        ["C-0034", "C-0053"],
    "service-account":        ["C-0034", "C-0053"],
    "serviceaccount":         ["C-0034", "C-0053"],
    "automount":              ["C-0034"],
    "token":                  ["C-0034", "C-0053"],

    # Secrets
    "secret":                 ["C-0015", "C-0065"],
    "secrets":                ["C-0015", "C-0065"],
    "env variable":           ["C-0065"],
    "environment variable":   ["C-0065"],
    "etcd":                   ["C-0066"],
    "encryption":             ["C-0066"],

    # Networking / exposure
    "host network":           ["C-0041"],
    "hostnetwork":            ["C-0041"],
    "host port":              ["C-0044"],
    "hostport":               ["C-0044"],
    "host pid":               ["C-0038"],
    "host ipc":               ["C-0038"],
    "ingress":                ["C-0030"],
    "egress":                 ["C-0030"],
    "exposed":                ["C-0021"],

    # Container hardening
    "privileged":             ["C-0057"],
    "read-only":              ["C-0017"],
    "read only filesystem":   ["C-0017"],
    "immutable":              ["C-0017"],
    "capabilities":           ["C-0046"],
    "non-root":               ["C-0059"],
    "non root":               ["C-0059"],
    "run as root":            ["C-0059"],
    "seccomp":                ["C-0055"],
    "apparmor":               ["C-0055"],
    "linux hardening":        ["C-0055"],

    # Resource management
    "resource limit":         ["C-0009"],
    "resource request":       ["C-0009"],
    "cpu limit":              ["C-0009"],
    "memory limit":           ["C-0009"],

    # Workload hygiene
    "liveness":               ["C-0056"],
    "readiness":              ["C-0018"],
    "default namespace":      ["C-0061"],
    "image registry":         ["C-0001"],
    "image tag":              ["C-0075"],
    "hostpath":               ["C-0045", "C-0048"],
    "host path":              ["C-0045", "C-0048"],

    # Container-optimised OS / node
    "container-optimized":    ["C-0069"],
    "container optimized":    ["C-0069"],
}


# =============================================================================
# 1. AUTO-LOAD TEXT FILES FROM TASK 2
# =============================================================================

def load_text_outputs(output_dir: str = "kde_outputs") -> tuple[str, str]:
    """
    Discover the two TEXT files written by kde_compare.py.

    Expected filenames:
        <output_dir>/name_diff.txt
        <output_dir>/full_diff.txt

    Returns:
        (name_diff_path, full_diff_path)

    Raises:
        FileNotFoundError: If either file is missing.
    """
    name_diff = os.path.join(output_dir, "name_diff.txt")
    full_diff  = os.path.join(output_dir, "full_diff.txt")

    missing = [p for p in (name_diff, full_diff) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"Expected TEXT file(s) not found: {missing}\n"
            f"Run kde_compare.py (Task 2) first."
        )

    return name_diff, full_diff


# =============================================================================
# 2. MAP DIFFERENCES → KUBESCAPE CONTROLS
# =============================================================================

NO_DIFF_SENTINEL = "NO DIFFERENCES FOUND"


def _extract_kde_terms_from_diff(full_diff_path: str) -> list[str]:
    """
    Parse full_diff.txt and return the unique KDE names / requirement strings
    that represent actual differences (i.e. the file is not the sentinel).
    """
    with open(full_diff_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content or "NO DIFFERENCES" in content.upper():
        return []

    terms = []
    for line in content.splitlines():
        parts = line.split(",")
        if not parts:
            continue
        # Column 0 = KDE name; column 3 = requirement (or NA)
        kde_name = parts[0].strip()
        req      = parts[3].strip() if len(parts) > 3 else ""

        terms.append(kde_name)
        if req and req.upper() != "NA":
            terms.append(req)

    return list(dict.fromkeys(terms))   # deduplicate, preserve order


def _terms_to_control_ids(terms: list[str]) -> list[str]:
    """
    Map a list of KDE name / requirement strings to Kubescape control IDs
    using the KEYWORD_TO_CONTROLS table.  Falls back to a simple
    case-insensitive substring search so partial matches still work.
    """
    matched: set[str] = set()

    for term in terms:
        term_lower = term.lower()
        for keyword, control_ids in KEYWORD_TO_CONTROLS.items():
            if keyword in term_lower:
                matched.update(control_ids)

    return sorted(matched)


def map_differences_to_kubescape_controls(
    name_diff_path: str,
    full_diff_path:  str,
    out_path:        str = "kde_outputs/kubescape_controls.txt",
) -> str:
    """
    Determine whether the Task-2 diff files contain real differences.
    If yes, map the differing KDE names/requirements to Kubescape control IDs
    and write them (one per line) to out_path.
    If no differences exist, write NO_DIFF_SENTINEL instead.

    Args:
        name_diff_path: Path to name_diff.txt from Task 2.
        full_diff_path: Path to full_diff.txt from Task 2.
        out_path:       Destination TEXT file.

    Returns:
        out_path
    """
    # Treat the full_diff file as authoritative — it is a strict superset of
    # the name-only diff.
    terms = _extract_kde_terms_from_diff(full_diff_path)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    if not terms:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(NO_DIFF_SENTINEL + "\n")
        print(f"[map_differences_to_kubescape_controls] No differences → {out_path}")
        return out_path

    control_ids = _terms_to_control_ids(terms)

    if not control_ids:
        # Differences exist but none matched any known control — log and
        # fall back to running all controls so the caller is never stuck.
        print(
            "[WARN] map_differences_to_kubescape_controls: differences found but no "
            "control IDs matched.  Falling back to all controls."
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(NO_DIFF_SENTINEL + "\n")
        return out_path

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(control_ids) + "\n")

    print(
        f"[map_differences_to_kubescape_controls] "
        f"{len(control_ids)} control(s) mapped → {out_path}"
    )
    return out_path


# =============================================================================
# 3. RUN KUBESCAPE AND RETURN A DATAFRAME
# =============================================================================

def _parse_kubescape_json(json_path: str) -> pd.DataFrame:
    """
    Parse a Kubescape JSON results file into a DataFrame.

    Expected columns:
        FilePath, Severity, Control name, Failed resources,
        All Resources, Compliance score
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # -----------------------------------------------------------------------
    # Build a lookup: resourceID → list of file paths (from 'resourcesResult')
    # -----------------------------------------------------------------------
    resource_paths: dict[str, str] = {}
    for res in data.get("resources", []):
        rid  = res.get("resourceID", "")
        ref  = res.get("object", {})
        # Try to extract a meaningful file path from the resource metadata
        path = (
            ref.get("metadata", {}).get("annotations", {})
               .get("config.kubernetes.io/origin", "")
            or ref.get("metadata", {}).get("name", rid)
        )
        resource_paths[rid] = path or rid

    # -----------------------------------------------------------------------
    # Extract per-control summary from summaryDetails
    # -----------------------------------------------------------------------
    summary    = data.get("summaryDetails", {})
    controls_s = summary.get("controls", {})

    rows = []
    for ctrl_id, ctrl in controls_s.items():
        name     = ctrl.get("name", ctrl_id)
        severity = (
            ctrl.get("scoreFactor", {}) if isinstance(ctrl.get("scoreFactor"), str)
            else ctrl.get("severity", {}).get("severity", "Unknown")
        )
        counters  = ctrl.get("resourceCounters", {})
        failed    = counters.get("failedResources",  0)
        passed    = counters.get("passedResources",  0)
        skipped   = counters.get("skippedResources", 0)
        total     = failed + passed + skipped

        # Compliance score: percentage of passing resources
        score = f"{(passed / total * 100):.1f}%" if total else "N/A"

        # Collect failed resource file paths for this control from 'results'
        failed_paths: list[str] = []
        for result in data.get("results", []):
            for rc in result.get("controls", []):
                if rc.get("controlID") == ctrl_id:
                    status = rc.get("status", {})
                    if isinstance(status, dict):
                        failed_flag = status.get("status", "") == "failed"
                    else:
                        failed_flag = str(status).lower() == "failed"
                    if failed_flag:
                        rid  = result.get("resourceID", "")
                        path = resource_paths.get(rid, rid)
                        if path:
                            failed_paths.append(path)

        # One row per unique failed path; if none, emit one summary row
        unique_paths = list(dict.fromkeys(failed_paths)) or [""]
        for fp in unique_paths:
            rows.append({
                "FilePath":        fp,
                "Severity":        severity,
                "Control name":    name,
                "Failed resources": failed,
                "All Resources":   total,
                "Compliance score": score,
            })

    df = pd.DataFrame(rows, columns=[
        "FilePath", "Severity", "Control name",
        "Failed resources", "All Resources", "Compliance score",
    ])
    return df


def run_kubescape(controls_path: str, zip_path: str) -> pd.DataFrame:
    """
    Run Kubescape on zip_path using the control list in controls_path.

    If controls_path contains only NO_DIFF_SENTINEL, Kubescape is run with
    all available controls.  Otherwise, only the listed control IDs are used.

    Args:
        controls_path: Path to the TEXT file produced by
                       map_differences_to_kubescape_controls().
        zip_path:      Path to the ZIP archive of Kubernetes manifests to scan.

    Returns:
        pd.DataFrame with columns:
            FilePath, Severity, Control name, Failed resources,
            All Resources, Compliance score
    """
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Manifest ZIP not found: '{zip_path}'")

    with open(controls_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    run_all = (content == NO_DIFF_SENTINEL)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_out = os.path.join(tmpdir, "results.json")

        if run_all:
            cmd = [
                "kubescape", "scan",
                "--format", "json",
                "--output", json_out,
                zip_path,
            ]
            print("[run_kubescape] Running Kubescape on ALL controls …")
        else:
            control_ids = [ln.strip() for ln in content.splitlines() if ln.strip()]
            controls_arg = ",".join(control_ids)
            cmd = [
                "kubescape", "scan", "control", controls_arg,
                "--format", "json",
                "--output", json_out,
                zip_path,
            ]
            print(f"[run_kubescape] Running Kubescape on controls: {controls_arg} …")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,        # 5-minute safety timeout
            )
        except FileNotFoundError:
            raise RuntimeError(
                "kubescape binary not found.  "
                "Install it with: curl -s https://raw.githubusercontent.com/"
                "kubescape/kubescape/master/install.sh | /bin/bash"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Kubescape scan timed out after 5 minutes.")

        if result.returncode not in (0, 1):
            # Kubescape exits 1 when failures are found — that is expected.
            raise RuntimeError(
                f"Kubescape exited with code {result.returncode}.\n"
                f"STDOUT: {result.stdout[:2000]}\n"
                f"STDERR: {result.stderr[:2000]}"
            )

        if not os.path.isfile(json_out):
            raise RuntimeError(
                f"Kubescape did not produce output at {json_out}.\n"
                f"STDOUT: {result.stdout[:2000]}\n"
                f"STDERR: {result.stderr[:2000]}"
            )

        df = _parse_kubescape_json(json_out)

    print(f"[run_kubescape] Scan complete — {len(df)} result row(s).")
    return df


# =============================================================================
# 4. WRITE THE CSV
# =============================================================================

EXPECTED_COLUMNS = [
    "FilePath",
    "Severity",
    "Control name",
    "Failed resources",
    "All Resources",
    "Compliance score",
]


def generate_csv(df: pd.DataFrame, out_path: str = "kde_outputs/kubescape_results.csv") -> str:
    """
    Write a Kubescape results DataFrame to a CSV file.

    Ensures the output has exactly the agreed column headers, filling any
    missing columns with empty strings so the schema is always consistent.

    Args:
        df:       DataFrame returned by run_kubescape().
        out_path: Destination CSV file path.

    Returns:
        out_path
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # Guarantee column presence and order regardless of what the parser returned
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df[EXPECTED_COLUMNS].to_csv(out_path, index=False, encoding="utf-8")
    print(f"[generate_csv] {len(df)} row(s) → {out_path}")
    return out_path


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    OUTPUT_DIR = "kde_outputs"
    ZIP_PATH   = "project-yamls.zip"

    # 1. Load Task-2 text files
    name_diff_path, full_diff_path = load_text_outputs(OUTPUT_DIR)
    print(f"Loaded:\n  {name_diff_path}\n  {full_diff_path}\n")

    # 2. Map differences to Kubescape controls
    controls_path = map_differences_to_kubescape_controls(
        name_diff_path,
        full_diff_path,
        out_path=os.path.join(OUTPUT_DIR, "kubescape_controls.txt"),
    )

    # 3. Run Kubescape
    df = run_kubescape(controls_path, ZIP_PATH)
    print(df.head())

    # 4. Save CSV
    generate_csv(df, out_path=os.path.join(OUTPUT_DIR, "kubescape_results.csv"))
