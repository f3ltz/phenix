"""
Generate high-resolution, publication-grade, legible diagrams for PHENIX.
Designed with large, readable typography (10-14pt), crisp borders, and clean spacing.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ASSETS_DIR = Path("report_assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def generate_system_overview():
    """Generates large, clean Figure 1: System Overview"""
    fig, ax = plt.subplots(figsize=(11.0, 5.8), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle="", items=[], facecolor="#ffffff", border_lw=1.5):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.03",
                                      facecolor=facecolor, edgecolor="#000000", linewidth=border_lw)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.04, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color="#000000")
        offset = 0.085
        if subtitle:
            ax.text(x + w/2, y + h - offset, subtitle, ha="center", va="top", fontsize=8.8, fontstyle="italic", color="#333333")
            offset += 0.050
        for item in items:
            ax.text(x + 0.025, y + h - offset, f"• {item}", ha="left", va="top", fontsize=8.5, color="#000000")
            offset += 0.048

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#000000", lw=1.6, mutation_scale=14))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.025, label, ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.2", facecolor="#ffffff", edgecolor="none"))

    # Column 1: Multimodal Ingestion Layer
    ax.text(0.18, 0.96, "STAGE 1: MULTIMODAL INGESTION", ha="center", fontsize=11, fontweight="bold")
    draw_box(0.03, 0.68, 0.30, 0.25, "Phenomics (PanTHERIA)", "3,268 Harmonized Mammals",
             ["Target: log10 adult body mass (g)", "Life history: gestation, litter size", "8 orders of magnitude range"], facecolor="#fcfcfc")
    draw_box(0.03, 0.38, 0.30, 0.25, "Bioclimatics (WorldClim v2.1)", "19 Planetary Climate Rasters",
             ["Bio1–Bio19 thermal & precip extremes", "Range-wide quantiles (10%, 50%, 90%)", "Elevation & WWF biome categories"], facecolor="#fcfcfc")
    draw_box(0.03, 0.08, 0.30, 0.25, "Genomics (Meta ESM-2)", "Foundation Protein Embeddings",
             ["BUSCO Mammalia single-copy orthologs", "32-dim dense sequence embeddings", "Contrasted with somatic growth loci"], facecolor="#fcfcfc")

    # Arrows to Stage 2
    draw_arrow(0.33, 0.80, 0.38, 0.70)
    draw_arrow(0.33, 0.50, 0.38, 0.55)
    draw_arrow(0.33, 0.20, 0.38, 0.40)

    # Column 2: Continuous Manifold & Whitening
    ax.text(0.53, 0.96, "STAGE 2: MANIFOLD & WHITENING", ha="center", fontsize=11, fontweight="bold")
    draw_box(0.38, 0.52, 0.30, 0.41, "Spectral Timetree (Phylo-PCoA)", "TimeTree 177 Ma Calibrated Chronogram",
             ["Patristic distance matrix D (3,268 x 3,268)", "Gower centering: B = -1/2 H (D^2) H", "Top 32 orthogonal spatial eigenmaps", "Captures 89.98% metric tree variance", "Exact out-of-sample Gower projection"],
             facecolor="#f5f5f5")
    draw_box(0.38, 0.08, 0.30, 0.40, "Cholesky Feature Whitening", "Covariance Decorrelation Engine",
             ["Regularized covariance: Σ = L L^T", "Whitened features: Z* = Z (L^-1)^T", "Guaranteed Cov(Z*) = Identity", "Completely eliminates collinearity"],
             facecolor="#ededed")

    draw_arrow(0.53, 0.52, 0.53, 0.48, "Decorrelate")
    draw_arrow(0.68, 0.72, 0.73, 0.72)
    draw_arrow(0.68, 0.28, 0.73, 0.28)

    # Column 3: Validation & Benchmarking
    ax.text(0.86, 0.96, "STAGE 3: DUAL-PROTOCOL VALIDATION", ha="center", fontsize=11, fontweight="bold")
    draw_box(0.73, 0.52, 0.25, 0.41, "Predictive Architectures", "High-Capacity Tabular & Geometric",
             ["Gradient Boosting (XGBoost)", "Ensemble Bagging (Random Forest)", "Regularized Ridge Regression", "Residual PhyloGNN (PyTorch Geo)", "ELU activations & layer norm"],
             facecolor="#f9f9f9")
    draw_box(0.73, 0.08, 0.25, 0.40, "Audited Dual-Protocol", "Inflation Gap & Diagnostic Engine",
             ["Random 10-Fold CV (Imputation)", "Phylo-CV: 140 Ma Patristic Buffer", "43 K-Pg boundary lineages (65 Ma)", "Residual Blomberg's K whiteness test", "3-way variance partitioning (P/E/G)"],
             facecolor="#ffffff")

    draw_arrow(0.85, 0.52, 0.85, 0.48, "Benchmark")

    ax.set_xlim(0, 1.01)
    ax.set_ylim(0.04, 1.01)

    plt.tight_layout(pad=0.3)
    out_path = ASSETS_DIR / "fig_sys_overview_bw.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated -> {out_path}")


def generate_flowchart():
    """Generates large, clean Figure 2: End-to-End Operation Flowchart"""
    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.axis("off")

    def draw_proc(x, y, w, h, step_num, title, items=[], facecolor="#ffffff"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.03",
                                      facecolor=facecolor, edgecolor="#000000", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.045, f"{step_num}. {title}", ha="center", va="top", fontsize=10.5, fontweight="bold", color="#000000")
        offset = 0.090
        for item in items:
            ax.text(x + w/2, y + h - offset, item, ha="center", va="top", fontsize=8.6, color="#222222")
            offset += 0.048

    def draw_decision(x, y, w, h, text, subtext=""):
        poly = patches.Polygon([[x + w/2, y], [x + w, y + h/2], [x + w/2, y + h], [x, y + h/2]],
                               facecolor="#f2f2f2", edgecolor="#000000", linewidth=1.6)
        ax.add_patch(poly)
        ax.text(x + w/2, y + h/2 + (0.02 if subtext else 0), text, ha="center", va="center", fontsize=9.2, fontweight="bold")
        if subtext:
            ax.text(x + w/2, y + h/2 - 0.045, subtext, ha="center", va="center", fontsize=7.6, fontstyle="italic")

    def draw_arr(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#000000", lw=1.6, mutation_scale=14))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.025, label, ha="center", va="bottom", fontsize=8.8, fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="#ffffff", edgecolor="none"))

    # Serpentine 2-row layout with wide legible boxes
    # Top Row: Data Ingestion & Representation
    draw_proc(0.03, 0.60, 0.20, 0.32, "1", "Multimodal Ingestion",
              ["PanTHERIA (3,268 Taxa)", "WorldClim 19 Rasters", "Meta ESM-2 Orthologs", "TimeTree 177 Ma Chronogram"], facecolor="#ffffff")
    draw_arr(0.23, 0.76, 0.28, 0.76)

    draw_proc(0.28, 0.60, 0.20, 0.32, "2", "Feature Whitening",
              ["Bioclimatic Collinearity", "Ridge Tikhonov Factor", "Cholesky Σ = L L^T", "Cov(Z*) = Identity Matrix"], facecolor="#f8f8f8")
    draw_arr(0.48, 0.76, 0.53, 0.76)

    draw_proc(0.53, 0.60, 0.20, 0.32, "3", "Spectral Timetree",
              ["Patristic Distances D", "Gower B = -1/2 H D^2 H", "32 Spatial Eigenmaps", "89.98% Variance Preserved"], facecolor="#f3f3f3")
    draw_arr(0.73, 0.76, 0.78, 0.76)

    draw_proc(0.78, 0.60, 0.19, 0.32, "4", "Gower Projection",
              ["Fit Eigendecomposition on Train", "Out-of-sample Test Taxa", "Zero Transductive Leakage", "117-Dim Feature Vectors"], facecolor="#ebebeb")

    # Serpentine connector from Row 1 to Row 2
    draw_arr(0.875, 0.60, 0.875, 0.44)

    # Bottom Row: Validation Protocols & Evaluation
    draw_decision(0.76, 0.16, 0.22, 0.26, "Validation Protocol", "Evaluation Split")

    draw_arr(0.76, 0.29, 0.68, 0.29, "Phylo-CV")
    draw_arr(0.875, 0.16, 0.875, 0.08)
    draw_arr(0.875, 0.08, 0.38, 0.08)
    draw_arr(0.38, 0.08, 0.38, 0.16, "Random CV")

    draw_proc(0.45, 0.16, 0.23, 0.28, "5a", "140 Ma Patristic Buffer",
              ["Monophyletic Order Withhold", "43 K-Pg Boundary Lineages", "Quarantine Buffer d >= 140 Ma", "Audits Out-of-Clade Boundary"], facecolor="#fafafa")
    draw_arr(0.45, 0.29, 0.38, 0.29)

    draw_proc(0.24, 0.16, 0.18, 0.28, "6", "Model Execution",
              ["Random Forest & XGBoost", "Ridge Linear Regressor", "Residual PhyloGNN", "Fit on Augmented Space"], facecolor="#f0f0f0")
    draw_arr(0.24, 0.29, 0.19, 0.29)

    draw_proc(0.03, 0.16, 0.16, 0.28, "7", "Dual Reporting",
              ["R^2 impute vs R^2 extrap", "Residual Blomberg's K", "Whiteness Proof (K<0.3)", "Variance Partition (P/E/G)"], facecolor="#ffffff")

    ax.text(0.50, 0.02, "PHENIX Architecture: From Raw Biological Multimodality to Leakage-Free Generalizable Prediction",
            ha="center", fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#000000", linewidth=1.2))

    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.01, 0.98)

    plt.tight_layout(pad=0.2)
    out_path = ASSETS_DIR / "fig_flowchart_bw.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated -> {out_path}")

if __name__ == "__main__":
    generate_system_overview()
    generate_flowchart()
