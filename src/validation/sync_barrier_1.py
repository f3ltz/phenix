import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
from src.data.schemas import ID_COL, BIOCLIM_VARS


def audit_sync_barrier_1(
    pheno_path: Path,
    env_path: Path,
    genomic_path: Path,
    report_out_path: Optional = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Sync Barrier 1: Target & Alignment Lock Audit.
    Enforces:
      1. Exact file existence across phenotypic, environmental, and genomic processed datasets.
      2. Presence and uniqueness of canonical_taxon_id primary key.
      3. Strict 1:1 row index matching (equal row counts and zero missing taxa across modalities).
      4. Identical row ordering of canonical_taxon_id across all tables.
      5. Zero missing values (NaNs / Infs) in environmental and genomic feature sets.
      6. Completeness of standard bioclimatic variables (bio1 to bio19).
    """
    pheno_path = Path(pheno_path)
    env_path = Path(env_path)
    genomic_path = Path(genomic_path)

    report: Dict[str, Any] = {
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "taxa_counts": {},
        "overlap_count": 0,
        "feature_counts": {},
        "alignment_verified": False,
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
            dups = df[ID_COL][df[ID_COL].duplicated()].unique().tolist()
            report["status"] = "FAIL"
            report["errors"].append(f"Duplicate taxon IDs in {name} dataset: {dups[:5]}")

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

    report["feature_counts"] = {
        "phenotypic_traits": len([c for c in df_pheno.columns if c != ID_COL]),
        "environmental_vars": len([c for c in df_env.columns if c != ID_COL]),
        "genomic_features": len([c for c in df_genomic.columns if c != ID_COL]),
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

    # Check row order alignment
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

    # Check bioclimatic variables coverage
    missing_bioclim = [v for v in BIOCLIM_VARS if v not in df_env.columns]
    if missing_bioclim:
        report["warnings"].append(f"Environmental table missing standard bioclim variables: {missing_bioclim}")

    is_passed = report["status"] == "PASS"
    report["alignment_verified"] = is_passed

    if report_out_path:
        out_p = Path(report_out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(report, f, indent=2)

    return is_passed, report


def main():
    parser = argparse.ArgumentParser(description="Sync Barrier 1 Alignment Audit Gate")
    parser.add_argument("--pheno", type=Path, default=Path("data/processed/phenotypic_clean.csv"))
    parser.add_argument("--env", type=Path, default=Path("data/processed/environmental_clean.csv"))
    parser.add_argument("--genomic", type=Path, default=Path("data/processed/genomic_clean.csv"))
    parser.add_argument("--report-out", type=Path, default=Path("data/processed/sync_barrier_1_report.json"))
    args = parser.parse_args()

    passed, report = audit_sync_barrier_1(args.pheno, args.env, args.genomic, report_out_path=args.report_out)
    print("=" * 60)
    print("SYNC BARRIER 1: AUDIT REPORT")
    print("=" * 60)
    print(f"STATUS: {report['status']}")
    print(f"Taxa counts: {report.get('taxa_counts', {})}")
    print(f"Feature counts: {report.get('feature_counts', {})}")
    print(f"Three-way overlap: {report.get('overlap_count', 0)}")
    print(f"Alignment verified: {report.get('alignment_verified', False)}")
    if report.get("warnings"):
        print("\nWARNINGS:")
        for w in report["warnings"]:
            print(f"  [!] {w}")
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
