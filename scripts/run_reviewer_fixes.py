"""
Script to compute reviewer-requested benchmarks:
1. Continuous buffer sensitivity curve (d_buffer in [0, 20, 50, 80, 110, 140] Ma)
2. K-Pg boundary chronological slicing benchmark (T_cut = 65.0 Ma, 43 lineages)
Saves results to JSON for manuscript embedding.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import RidgeCV

from src.phylogenetics.withholding import MonophyleticWithholdingEngine
from src.phylogenetics.otol import PhyloNode
from src.phylogenetics.patristic import PatristicMatrixEngine
from src.validation.cv import PhyloCVSplitter
from src.models.diagnostics import ResidualDiagnosticsEngine

def main():
    data_dir = Path("data/processed")
    df_pheno = pd.read_csv(data_dir / "phenotypic_clean.csv")
    df_env = pd.read_csv(data_dir / "environmental_clean.csv")
    df_gen = pd.read_csv(data_dir / "genomic_clean.csv")
    D_full = np.load(data_dir / "patristic_distance_matrix.npy")
    taxa = df_pheno["canonical_taxon_id"].tolist()
    N = len(taxa)

    env_cols = [c for c in df_env.columns if c not in ["canonical_taxon_id", "latitude", "longitude", "wwf_biome_id"]]
    gen_cols = [c for c in df_gen.columns if c != "canonical_taxon_id"]
    X_base = np.hstack([df_env[env_cols].values, df_gen[gen_cols].values])
    y = np.log10(df_pheno["adult_body_mass_g"].values)

    with open(data_dir / "phylo_tree_calibrated.nwk") as f:
        root = PhyloNode.from_newick(f.read().strip())
    engine = MonophyleticWithholdingEngine(root)
    p_engine = PatristicMatrixEngine()
    diag_engine = ResidualDiagnosticsEngine(data_dir / "patristic_distance_matrix.npy", root_age_ma=177.0)

    print("=== 1. Computing Continuous Buffer Sensitivity Curve ===")
    buffers = [0.0, 20.0, 50.0, 80.0, 110.0, 140.0]
    buffer_results = []

    for d_buf in buffers:
        splitter = PhyloCVSplitter(
            all_taxa=taxa,
            patristic_matrix=D_full,
            withholding_engine=engine,
            mode="order",
            df_pheno=df_pheno,
            min_clade_size=50,
            d_buffer=d_buf,
        )
        oof_preds_rf = np.full(N, np.nan)
        oof_preds_xgb = np.full(N, np.nan)
        n_train_list = []
        n_buf_list = []

        for fold in splitter.split():
            tr = fold.train_indices
            te = fold.test_indices
            n_train_list.append(len(tr))
            n_buf_list.append(len(fold.buffer_indices))

            D_tr = D_full[np.ix_(tr, tr)]
            D_te_tr = D_full[np.ix_(te, tr)]
            B_tr = p_engine.compute_double_centered_matrix(D_tr)

            evals_raw, evecs_raw = np.linalg.eigh(B_tr)
            idx_desc = np.argsort(evals_raw)[::-1]
            k = 32
            evals_k = evals_raw[idx_desc][:k]
            evecs_k = evecs_raw[:, idx_desc][:, :k]

            X_pcoa_tr = evecs_k * np.sqrt(np.maximum(evals_k, 0))
            X_pcoa_te = p_engine.project_pcoa_eigenmaps(D_te_tr, D_tr, evals_k, evecs_k)

            X_tr = np.hstack([X_base[tr], X_pcoa_tr])
            X_te = np.hstack([X_base[te], X_pcoa_te])

            rf = RandomForestRegressor(n_estimators=25, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y[tr])
            oof_preds_rf[te] = rf.predict(X_te)

            xgb = XGBRegressor(n_estimators=25, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
            xgb.fit(X_tr, y[tr])
            oof_preds_xgb[te] = xgb.predict(X_te)

        mask = ~np.isnan(oof_preds_rf)
        yt = y[mask]
        yp_rf = oof_preds_rf[mask]
        yp_xgb = oof_preds_xgb[mask]
        ss_tot = np.sum((yt - np.mean(yt))**2)
        
        r2_rf = 1.0 - (np.sum((yt - yp_rf)**2) / ss_tot)
        mae_rf = float(np.mean(np.abs(yt - yp_rf)))
        rmse_rf = float(np.sqrt(np.mean((yt - yp_rf)**2)))
        fold_err_rf = float(10.0 ** mae_rf)

        r2_xgb = 1.0 - (np.sum((yt - yp_xgb)**2) / ss_tot)
        mae_xgb = float(np.mean(np.abs(yt - yp_xgb)))
        rmse_xgb = float(np.sqrt(np.mean((yt - yp_xgb)**2)))
        fold_err_xgb = float(10.0 ** mae_xgb)

        # Residual Blomberg's K
        valid_idx = np.where(mask)[0]
        D_sub = D_full[np.ix_(valid_idx, valid_idx)]
        resid_rf = yt - yp_rf
        K_rf = diag_engine.compute_blomberg_k(resid_rf, D_sub)

        row = {
            "d_buffer_ma": d_buf,
            "mean_n_train": int(np.mean(n_train_list)),
            "mean_n_buffer": int(np.mean(n_buf_list)),
            "r2_pooled_rf": round(float(r2_rf), 4),
            "rmse_rf": round(rmse_rf, 4),
            "mae_rf": round(mae_rf, 4),
            "fold_err_rf": round(fold_err_rf, 2),
            "r2_pooled_xgb": round(float(r2_xgb), 4),
            "rmse_xgb": round(rmse_xgb, 4),
            "mae_xgb": round(mae_xgb, 4),
            "fold_err_xgb": round(fold_err_xgb, 2),
            "resid_blomberg_k_rf": round(float(K_rf), 4),
        }
        buffer_results.append(row)
        print(f"  Buffer {d_buf:5.1f} Ma | RF R2={row['r2_pooled_rf']:7.4f}, MAE={row['mae_rf']:.4f} ({row['fold_err_rf']:.1f}x) | K={row['resid_blomberg_k_rf']:.4f}")

    with open(data_dir / "buffer_sensitivity.json", "w") as f:
        json.dump(buffer_results, f, indent=2)

    print("\n=== 2. Computing 43 K-Pg Boundary Lineages Benchmark (T_cut = 65.0 Ma) ===")
    k_pg_splitter = PhyloCVSplitter(
        all_taxa=taxa,
        patristic_matrix=D_full,
        withholding_engine=engine,
        mode="t_cut",
        t_cut_ma=65.0,
        min_clade_size=10,
        d_buffer=0.0,
    )
    print(f"Total lineages evaluated (N >= 10): {k_pg_splitter.get_n_splits()}")

    k_pg_results = {}
    models = {
        "Ridge": lambda: RidgeCV(),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=25, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": lambda: XGBRegressor(n_estimators=25, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1),
    }

    for m_name, m_factory in models.items():
        oof_preds = np.full(N, np.nan)
        for fold in k_pg_splitter.split():
            tr = fold.train_indices
            te = fold.test_indices

            D_tr = D_full[np.ix_(tr, tr)]
            D_te_tr = D_full[np.ix_(te, tr)]
            B_tr = p_engine.compute_double_centered_matrix(D_tr)

            evals_raw, evecs_raw = np.linalg.eigh(B_tr)
            idx_desc = np.argsort(evals_raw)[::-1]
            k = 32
            evals_k = evals_raw[idx_desc][:k]
            evecs_k = evecs_raw[:, idx_desc][:, :k]

            X_pcoa_tr = evecs_k * np.sqrt(np.maximum(evals_k, 0))
            X_pcoa_te = p_engine.project_pcoa_eigenmaps(D_te_tr, D_tr, evals_k, evecs_k)

            X_tr = np.hstack([X_base[tr], X_pcoa_tr])
            X_te = np.hstack([X_base[te], X_pcoa_te])

            model = m_factory()
            model.fit(X_tr, y[tr])
            oof_preds[te] = model.predict(X_te)

        mask = ~np.isnan(oof_preds)
        yt = y[mask]
        yp = oof_preds[mask]
        ss_tot = np.sum((yt - np.mean(yt))**2)
        ss_res = np.sum((yt - yp)**2)
        r2_pooled = float(1.0 - (ss_res / ss_tot))
        rmse = float(np.sqrt(np.mean((yt - yp)**2)))
        mae = float(np.mean(np.abs(yt - yp)))
        fold_err = float(10.0 ** mae)

        k_pg_results[m_name] = {
            "n_taxa": int(len(yt)),
            "r2_pooled": round(r2_pooled, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "fold_error": round(fold_err, 2),
        }
        print(f"  K-Pg {m_name:<12} | N={len(yt)} | R2_pooled={r2_pooled:7.4f} | RMSE={rmse:.4f} | MAE={mae:.4f} ({fold_err:.2f}x)")

    with open(data_dir / "k_pg_benchmark.json", "w") as f:
        json.dump(k_pg_results, f, indent=2)

    print("\nAll reviewer benchmarks completed and saved successfully!")

if __name__ == "__main__":
    main()
