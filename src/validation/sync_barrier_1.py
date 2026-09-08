import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
from src.data.schemas import ID_COL, BIOCLIM_VARS

def audit_sync_barrier_1(
    pheno_path: Path,
    env_path: Path,
    genomic_path: Path
) -> Tuple[bool, Dict[str, Any]]:
    report = {
        "status": "PASS",
        "errors": [],
        "taxa_counts": {},
        "overlap_count": 0,
    }

    if not pheno_path.exists():
        report["status"] = "FAIL"
        report["errors"].append(f"Phenotypic table not found: {pheno_path}")
    if not env_path.exists():
        report["status"] = "FAIL"
        report["errors"].append(f"Environmental table not found: {env_path}")
    if not genomic_path.exists():
        report["status"] = "FAIL"
        report["errors"].append(f"Genomic table not found: {genomic_path}")

    if report["status"] == "FAIL":
        return False, report

    df_pheno = pd.read_csv(pheno_path)
    df_env = pd.read_csv(env_path)
    df_genomic = pd.read_csv(genomic_path)

    for name, df in [("phenotypic", df_pheno), ("environmental", df_env), ("genomic", df_genomic)]:
        if ID_COL not in df.columns:
            report["status"] = "FAIL"
            report["errors"].append(f"Primary key '{ID_COL}' missing from {name} dataset.")
        elif df[ID_COL].duplicated().any():
            report["status"] = "FAIL"
            report["errors"].append(f"Duplicate taxon IDs in {name} dataset.")

    if report["status"] == "FAIL":
        return False, report

    taxa_pheno = set(df_pheno[ID_COL])
    taxa_env = set(df_env[ID_COL])
    taxa_gen = set(df_genomic[ID_COL])

    report["taxa_counts"] = {
        "phenotypic": len(taxa_pheno),
        "environmental": len(taxa_env),
        "genomic": len(taxa_gen),
    }

    common_taxa = taxa_pheno.intersection(taxa_env).intersection(taxa_gen)
    report["overlap_count"] = len(common_taxa)

    if len(common_taxa) == 0:
        report["status"] = "FAIL"
        report["errors"].append("Zero overlapping taxa across all three tables.")
        return False, report

    # Check 1:1 row index and identity
    if not (len(df_pheno) == len(df_env) == len(df_genomic) == len(common_taxa)):
        report["status"] = "FAIL"
        report["errors"].append(
            f"Table lengths do not match common overlap ({len(common_taxa)}). "
            f"Counts: pheno={len(df_pheno)}, env={len(df_env)}, gen={len(df_genomic)}"
        )

    # Check order alignment
    if not (df_pheno[ID_COL].equals(df_env[ID_COL]) and df_env[ID_COL].equals(df_genomic[ID_COL])):
        report["status"] = "FAIL"
        report["errors"].append("Row orders of canonical_taxon_id are not identical across tables.")

    # Check for NaN / Infs in environmental and genomic tables
    if df_env.isnull().any().any():
        report["status"] = "FAIL"
        nan_cols = df_env.columns[df_env.isnull().any()].tolist()
        report["errors"].append(f"Environmental table contains NaNs in: {nan_cols}")

    if df_genomic.isnull().any().any():
        report["status"] = "FAIL"
        nan_cols = df_genomic.columns[df_genomic.isnull().any()].tolist()
        report["errors"].append(f"Genomic table contains NaNs in: {nan_cols}")

    is_passed = report["status"] == "PASS"
    return is_passed, report

def main():
    parser = argparse.ArgumentParser(description="Sync Barrier 1 Alignment Audit Gate")
    parser.add_argument("--pheno", type=Path, default=Path("data/processed/phenotypic_clean.csv"))
    parser.add_argument("--env", type=Path, default=Path("data/processed/environmental_clean.csv"))
    parser.add_argument("--genomic", type=Path, default=Path("data/processed/genomic_clean.csv"))
    args = parser.parse_args()

    passed, report = audit_sync_barrier_1(args.pheno, args.env, args.genomic)
    print("=" * 60)
    print("SYNC BARRIER 1: AUDIT REPORT")
    print("=" * 60)
    print(f"STATUS: {report['status']}")
    print(f"Taxa counts: {report.get('taxa_counts', {})}")
    print(f"Three-way overlap: {report.get('overlap_count', 0)}")
    if report["errors"]:
        print("\nERRORS DETECTED:")
        for err in report["errors"]:
            print(f"  [X] {err}")
    print("=" * 60)

    if not passed:
        sys.exit(1)
    print("SUCCESS: Target & Alignment Locked.")

if __name__ == "__main__":
    main()
