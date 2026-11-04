"""
Generates high-contrast, publication-grade Black & White / Grayscale assets for
the PHENIX Endsem Dissertation. Tailored specifically for monochrome / grayscale laser printing.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ASSETS_DIR = Path("d:/Projects/Phenix/report_assets").resolve()
DATA_DIR = Path("d:/Projects/Phenix/data/processed").resolve()


def generate_results_chart_bw():
    """Generates B&W results chart with distinct hatching patterns and grayscale fills."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax1.set_facecolor("#ffffff")
    ax2.set_facecolor("#ffffff")

    models = ['Random Forest', 'XGBoost', 'Ridge Reg.', 'PhyloGNN']
    r2_means = [0.8554, 0.8597, 0.7893, -0.7726]
    r2_stds = [0.0143, 0.0194, 0.0295, 0.7381]
    extrap_rmse = [1.4254, 1.3137, 4.0708, 1.6606]

    # Grayscale fills and distinct hatch patterns
    hatches = ['///', 'xxx', '...', '\\\\\\']
    facecolors = ['#e8e8e8', '#888888', '#f4f4f4', '#2b2b2b']

    # (a) In-Domain Random CV R2
    for i, (m, r2, std) in enumerate(zip(models, r2_means, r2_stds)):
        h = max(0, r2)
        yerr = std if r2 > 0 else 0
        ax1.bar(i, h, yerr=yerr, width=0.52,
                facecolor=facecolors[i], edgecolor='#000000',
                linewidth=1.3, hatch=hatches[i], capsize=4)

    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel('In-Domain Random CV $R^2$', fontsize=10, fontweight='bold', color='#000000')
    ax1.set_title('(a) In-Domain Imputation Accuracy ($R^2$)', fontsize=11, fontweight='bold', pad=10, color='#000000')
    ax1.axhline(0.85, color='#000000', linestyle='--', linewidth=1.2, label=r'High-Precision Threshold ($R^2 \geq 0.85$)')

    for i, mean in enumerate(r2_means):
        if mean > 0:
            ax1.text(i, mean + 0.03, f'{mean:.3f}', ha='center', va='bottom', fontsize=9.2, fontweight='bold', color='#000000')
        else:
            ax1.text(i, 0.06, f'Fail\n({mean:.2f})', ha='center', va='bottom', fontsize=8.2, fontweight='bold', color='#000000')

    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, fontsize=9.5, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8.2, framealpha=1.0, edgecolor='#000000')
    ax1.grid(True, linestyle=':', alpha=0.6, color='#999999', axis='y')

    # (b) Out-of-Clade Extrapolation RMSE
    for i, (m, rmse) in enumerate(zip(models, extrap_rmse)):
        ax2.bar(i, rmse, width=0.52,
                facecolor=facecolors[i], edgecolor='#000000',
                linewidth=1.3, hatch=hatches[i])
        badge = ' [Best]' if m == 'XGBoost' else ''
        ax2.text(i, rmse + 0.12, f'{rmse:.3f}{badge}', ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#000000')

    ax2.set_ylim(0, 4.8)
    ax2.set_ylabel('Out-of-Clade Extrapolation RMSE', fontsize=10, fontweight='bold', color='#000000')
    ax2.set_title('(b) Out-of-Clade Error (140 Ma Patristic Buffer)', fontsize=11, fontweight='bold', pad=10, color='#000000')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, fontsize=9.5, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6, color='#999999', axis='y')

    plt.tight_layout(pad=1.2)
    out_path = ASSETS_DIR / "fig_results_chart_bw.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated -> {out_path}")


def generate_divergence_decay_bw():
    """Generates B&W divergence decay curve with distinct markers and linestyles."""
    json_path = DATA_DIR / "divergence_accuracy_tracking.json"
    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)

        styles = {
            "RandomForest": {"ls": "-", "marker": "o", "mfc": "#000000", "mec": "#000000", "label": "Random Forest (Augmented)"},
            "XGBoost": {"ls": "--", "marker": "s", "mfc": "#ffffff", "mec": "#000000", "label": "XGBoost (Augmented - Optimal)"},
            "Ridge": {"ls": ":", "marker": "^", "mfc": "#666666", "mec": "#000000", "label": "Ridge Regression (Linear Baseline)"},
            "PhyloGNN": {"ls": "-.", "marker": "D", "mfc": "#ffffff", "mec": "#000000", "label": "PhyloGNN (Geometric Baseline)"}
        }

        for model_name, cfg in styles.items():
            if model_name in data:
                divs = np.array(data[model_name]["mean_divergences_ma"])
                rmses = np.array(data[model_name]["rmses"])
                idx = np.argsort(divs)
                divs = divs[idx]
                rmses = rmses[idx]
                ax.plot(divs, rmses, color="#000000", linestyle=cfg["ls"],
                        linewidth=1.8, marker=cfg["marker"], markerfacecolor=cfg["mfc"],
                        markeredgecolor=cfg["mec"], markeredgewidth=1.3, markersize=6.5,
                        label=cfg["label"])

    ax.axvline(140.0, color="#000000", linestyle=":", linewidth=1.4)
    ax.text(141.5, 5.2, "140.0 Ma Quarantine Boundary\n(Strict Deep-Time Clade Isolation)",
            fontsize=8.2, fontweight="bold", color="#000000",
            bbox=dict(boxstyle="square,pad=0.3", facecolor="#ffffff", edgecolor="#000000", linewidth=1.0))

    ax.set_xlabel("Mean Patristic Divergence to Training Clades (Million Years - Ma)", fontsize=9.5, fontweight="bold", color="#000000")
    ax.set_ylabel("Out-of-Fold Extrapolation RMSE (log₁₀ mass)", fontsize=9.5, fontweight="bold", color="#000000")
    ax.set_title("Macroevolutionary Predictive Accuracy Decay Across Deep Time", fontsize=10.5, fontweight="bold", pad=10, color="#000000")
    ax.grid(True, linestyle=":", alpha=0.6, color="#999999")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=1.0, edgecolor="#000000")

    plt.tight_layout(pad=1.2)
    out_path = ASSETS_DIR / "fig_divergence_decay_bw.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated -> {out_path}")


def generate_simulation_scenarios_bw():
    """Generates B&W simulation scenario comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8.8, 4.0), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    scenarios = ["SSP1-2.6 (+1.5°C)", "SSP2-4.5 (+3.0°C)", "SSP5-8.5 (+5.0°C)"]
    x = np.arange(len(scenarios))
    w = 0.18

    ou_shifts = [-0.35, -0.19, -0.75]
    piml_shifts = [-0.28, -0.15, -0.62]
    aug_xgb_shifts = [4.35, 5.08, 5.64]
    naive_xgb_shifts = [13.02, 14.10, 12.27]

    ax.bar(x - 1.5*w, ou_shifts, w, facecolor="#1a1a1a", edgecolor="#000000", linewidth=1.2, label="Thermodynamic OU (Physics Ground Truth)")
    ax.bar(x - 0.5*w, piml_shifts, w, facecolor="#666666", hatch="xx", edgecolor="#000000", linewidth=1.2, label="Physics-Informed ML (PI-ML Bound)")
    ax.bar(x + 0.5*w, aug_xgb_shifts, w, facecolor="#cccccc", hatch="///", edgecolor="#000000", linewidth=1.2, label="Augmented ML (Phylo-PCoA XGBoost)")
    ax.bar(x + 1.5*w, naive_xgb_shifts, w, facecolor="#ffffff", hatch="...", edgecolor="#000000", linewidth=1.2, label="Naive Baseline ML (Unanchored XGBoost)")

    ax.axhline(0.0, color="#000000", linewidth=1.2)
    ax.set_ylabel("Projected Body Mass Shift (Δ% from Present)", fontsize=9.5, fontweight="bold", color="#000000")
    ax.set_title("Forward Eco-Evolutionary Trait Shift Projections Across IPCC CMIP6 Scenarios", fontsize=10.5, fontweight="bold", pad=10, color="#000000")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9.5, fontweight="bold")
    ax.set_ylim(-2.5, 17.0)
    ax.grid(True, linestyle=":", alpha=0.6, color="#999999", axis="y")

    ax.annotate("Unphysical Trait Explosion\n(Violates Bergmann's Rule)",
                xy=(2 + 1.5*w, 12.5), xytext=(1.35, 14.2),
                arrowprops=dict(arrowstyle="->", color="#000000", lw=1.2),
                fontsize=8.2, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.25", facecolor="#ffffff", edgecolor="#000000"))

    ax.annotate("Thermodynamic Cooling Bound\n(Negative Thermal Scaling ΔM < 0)",
                xy=(0 - 1.5*w, -0.4), xytext=(-0.35, -2.1),
                arrowprops=dict(arrowstyle="->", color="#000000", lw=1.2),
                fontsize=8.0, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.25", facecolor="#ffffff", edgecolor="#000000"))

    ax.legend(loc="upper left", fontsize=8.2, framealpha=1.0, edgecolor="#000000")

    plt.tight_layout(pad=1.2)
    out_path = ASSETS_DIR / "fig_simulation_scenarios_bw.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated -> {out_path}")


def generate_system_overview_bw():
    """Generates crisp vector-style B&W System Architecture Diagram (Figure 3.1) with zero text overlap."""
    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle="", items=[], facecolor="#ffffff"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.025",
                                      facecolor=facecolor, edgecolor="#000000", linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.035, title, ha="center", va="top", fontsize=8.6, fontweight="bold", color="#000000")
        offset = 0.070
        if subtitle:
            ax.text(x + w/2, y + h - offset, subtitle, ha="center", va="top", fontsize=7.2, fontstyle="italic", color="#333333")
            offset += 0.040
        for item in items:
            ax.text(x + 0.02, y + h - offset, f"• {item}", ha="left", va="top", fontsize=6.8, color="#000000")
            offset += 0.036

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#000000", lw=1.4, mutation_scale=12))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.025, label, ha="center", va="bottom", fontsize=7.2, fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="#ffffff", edgecolor="none"))

    # Tier 1: Multimodal Raw Data Layer (Left Column)
    ax.text(0.18, 0.97, "TIER 1: MULTIMODAL INGESTION", ha="center", fontsize=8.8, fontweight="bold")
    draw_box(0.03, 0.68, 0.29, 0.26, "PanTHERIA Phenomics", "N = 1,234 species",
             ["Adult body mass (g)", "Litter size & longevity", "Thermoregulatory traits"], facecolor="#ffffff")
    draw_box(0.03, 0.38, 0.29, 0.26, "WorldClim v2.1 Rasters", "19 Bioclimatic Rasters",
             ["Bio1 Mean Annual Temp", "Bio12 Annual Precip", "10-arcmin GeoTIFF rasters"], facecolor="#ffffff")
    draw_box(0.03, 0.08, 0.29, 0.26, "NCBI GenBank Genomics", "Cytochrome b CDS",
             ["1,140 bp protein coding", "GC% & Codon Usage Bias", "Transition/Transversion (Ts/Tv)"], facecolor="#ffffff")

    # Arrows to Tier 2
    draw_arrow(0.32, 0.81, 0.38, 0.68)
    draw_arrow(0.32, 0.51, 0.38, 0.55)
    draw_arrow(0.32, 0.21, 0.38, 0.40)

    # Tier 2: Continuous Manifold & Whitening
    ax.text(0.52, 0.97, "TIER 2: SPECTRAL MANIFOLD & WHITENING", ha="center", fontsize=8.8, fontweight="bold")
    draw_box(0.38, 0.53, 0.28, 0.41, "Phylo-PCoA Manifold", "TimeTree 140 Ma Calibrated Tree",
             ["Patristic distance matrix D", "Double-centering: B = -½ H D² H", "Top-10 spectral eigenvectors", "Euclidean lineage coordinates"],
             facecolor="#f6f6f6")
    draw_box(0.38, 0.08, 0.28, 0.40, "Mahalanobis Covariance Whitening", "Orthogonalized Feature Space",
             ["Cholesky Factor: Σ = L Lᵀ", "Whitened: X_white = (X - μ) L⁻ᵀ", "Cov(X_white) = Identity", "Zero phylogenetic collinearity"],
             facecolor="#ebebeb")

    draw_arrow(0.52, 0.53, 0.52, 0.48, "Decorrelate")
    draw_arrow(0.66, 0.73, 0.72, 0.73)
    draw_arrow(0.66, 0.28, 0.72, 0.30)

    # Tier 3: Downstream Models & Simulation
    ax.text(0.86, 0.97, "TIER 3: PREDICTION & BENCHMARKING", ha="center", fontsize=8.8, fontweight="bold")
    draw_box(0.72, 0.53, 0.26, 0.41, "Predictive ML & PI-ML", "Forward & Imputation Models",
             ["Random Forest & XGBoost", "Ridge Linear Regression", "PhyloGNN Graph Baseline", "Physics-Informed ML (PI-ML)", "Fourier-Kleiber regularizer"],
             facecolor="#f9f9f9")
    draw_box(0.72, 0.08, 0.26, 0.40, "Dual-Protocol Validation", "Strict Generalizability Audit",
             ["Random 5-Fold Cross-Validation", "Phylo-CV (140 Ma Buffer)", "Inflation Gap ΔR² Audit", "Extrapolation Decay Tracking"],
             facecolor="#ffffff")

    draw_arrow(0.85, 0.53, 0.85, 0.48, "Evaluate")

    ax.set_xlim(0, 1.01)
    ax.set_ylim(0.04, 1.01)

    plt.tight_layout(pad=0.4)
    out_path = ASSETS_DIR / "fig_sys_overview_bw.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated -> {out_path}")


def generate_flowchart_bw():
    """Generates crisp vector-style Flow of Operation Flowchart (Figure 4.1)."""
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.axis("off")

    def draw_proc(x, y, w, h, text, sub=""):
        rect = patches.Rectangle((x, y), w, h, facecolor="#ffffff", edgecolor="#000000", linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.02 if sub else 0), text, ha="center", va="center", fontsize=8.2, fontweight="bold")
        if sub:
            ax.text(x + w/2, y + h/2 - 0.04, sub, ha="center", va="center", fontsize=7.0, fontstyle="italic")

    def draw_decision(x, y, w, h, text):
        poly = patches.Polygon([[x + w/2, y], [x + w, y + h/2], [x + w/2, y + h], [x, y + h/2]],
                               facecolor="#f5f5f5", edgecolor="#000000", linewidth=1.3)
        ax.add_patch(poly)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=7.6, fontweight="bold")

    def draw_arr(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#000000", lw=1.3, mutation_scale=11))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.02, label, ha="center", va="bottom", fontsize=7.0, fontweight="bold")

    # Flowchart steps arranged in a clean 2-row serpentine flow
    draw_proc(0.04, 0.68, 0.18, 0.18, "1. Ingest Data", "PanTHERIA + WorldClim + NCBI")
    draw_arr(0.22, 0.77, 0.27, 0.77)

    draw_proc(0.27, 0.68, 0.18, 0.18, "2. Timetree Matrix", "TimeTree Patristic Dist D")
    draw_arr(0.45, 0.77, 0.50, 0.77)

    draw_proc(0.50, 0.68, 0.18, 0.18, "3. Spectral PCoA", "B = -½HD²H → Top-10 Eig")
    draw_arr(0.68, 0.77, 0.73, 0.77)

    draw_proc(0.73, 0.68, 0.23, 0.18, "4. Mahalanobis Whitening", "Cholesky Σ = LLᵀ, X_white")

    draw_arr(0.84, 0.68, 0.84, 0.48)

    draw_decision(0.73, 0.28, 0.22, 0.20, "140 Ma Patristic\nQuarantine Buffer?")
    draw_arr(0.73, 0.38, 0.68, 0.38, "Yes")

    draw_proc(0.48, 0.29, 0.20, 0.18, "5. Model Training", "RF, XGB, Ridge, PhyloGNN")
    draw_arr(0.48, 0.38, 0.43, 0.38)

    draw_proc(0.25, 0.29, 0.18, 0.18, "6. Dual Validation", "In-Domain vs Phylo-CV")
    draw_arr(0.25, 0.38, 0.20, 0.38)

    draw_proc(0.04, 0.29, 0.16, 0.18, "7. PI-ML Forecast", "OU & Thermal Bounds")

    ax.text(0.5, 0.10, "End-to-End Pipeline Guarantees Strict Zero-Leakage & Thermodynamic Consistency",
            ha="center", fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#000000", linewidth=1.1))

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.05, 0.95)

    plt.tight_layout(pad=0.5)
    out_path = ASSETS_DIR / "fig_flowchart_bw.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated -> {out_path}")


def generate_ui_layout_bw():
    """Generates crisp vector-style Wireframe UI Layout (Figure 5.1) with perfect zero-overlap spacing."""
    fig, ax = plt.subplots(figsize=(9.8, 5.2), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.axis("off")

    # Main outer browser window
    outer = patches.Rectangle((0.02, 0.04), 0.96, 0.90, facecolor="#ffffff", edgecolor="#000000", linewidth=1.5)
    ax.add_patch(outer)

    # Window title bar
    title_bar = patches.Rectangle((0.02, 0.88), 0.96, 0.06, facecolor="#f0f0f0", edgecolor="#000000", linewidth=1.1)
    ax.add_patch(title_bar)
    ax.text(0.05, 0.91, "●  ●  ●   PHENIX: Interactive Eco-Evolutionary Trait Forecaster & Comparative Phenomics Platform",
            va="center", fontsize=8.0, fontweight="bold", color="#000000")

    # Left Sidebar (Controls)
    sidebar = patches.Rectangle((0.04, 0.06), 0.26, 0.80, facecolor="#fafafa", edgecolor="#000000", linewidth=1.1)
    ax.add_patch(sidebar)
    ax.text(0.17, 0.83, "CONTROLS & PARAMETERS", ha="center", fontsize=8.2, fontweight="bold")
    ax.text(0.06, 0.77, "Select Species (N=1,234):", fontsize=7.2, fontweight="bold")
    ax.text(0.06, 0.73, "[ Pan troglodytes ▼ ]", fontsize=7.0, bbox=dict(boxstyle="square,pad=0.18", facecolor="#ffffff", edgecolor="#888888"))

    ax.text(0.06, 0.65, "Target Phenotypic Trait:", fontsize=7.2, fontweight="bold")
    ax.text(0.06, 0.61, "(•) Adult Body Mass (g)\n( ) Litter Size  ( ) Longevity", fontsize=6.8)

    ax.text(0.06, 0.50, "IPCC Climate Scenario:", fontsize=7.2, fontweight="bold")
    ax.text(0.06, 0.46, "Warming ΔT = +3.0°C (SSP2-4.5)", fontsize=6.8)
    ax.plot([0.06, 0.27], [0.42, 0.42], color="#000000", lw=2)
    ax.plot([0.165, 0.165], [0.40, 0.44], color="#000000", lw=3)

    ax.text(0.06, 0.33, "Validation & Constraints:", fontsize=7.2, fontweight="bold")
    ax.text(0.06, 0.28, "[x] 140 Ma Patristic Buffer\n[x] Physics-Informed ML", fontsize=6.8)

    btn = patches.Rectangle((0.06, 0.11), 0.22, 0.08, facecolor="#000000", edgecolor="#000000")
    ax.add_patch(btn)
    ax.text(0.17, 0.15, "RUN IN-SILICO FORECAST", ha="center", va="center", color="#ffffff", fontsize=7.5, fontweight="bold")

    # Top Panel: Species Info Card
    sp_card = patches.Rectangle((0.33, 0.69), 0.63, 0.17, facecolor="#ffffff", edgecolor="#000000", linewidth=1.1)
    ax.add_patch(sp_card)
    ax.text(0.35, 0.835, "TAXONOMIC & CLIMATIC PROFILE: Pan troglodytes (Chimpanzee)", va="top", fontsize=7.8, fontweight="bold")
    ax.text(0.35, 0.785, "• Order: Primates  |  Family: Hominidae  |  NCBI Accession: NC_001643.1\n• Observed Mass: 45,000 g  |  Annual Mean Temp: 24.6°C  |  Precipitation: 1,850 mm/yr\n• Minimum Patristic Divergence to Training Set: 148.5 Ma (Strict Quarantined Holdout)",
            va="top", linespacing=1.35, fontsize=6.6)

    # Middle Left: Live Forecast Comparison
    card_m1 = patches.Rectangle((0.33, 0.29), 0.30, 0.37, facecolor="#ffffff", edgecolor="#000000", linewidth=1.1)
    ax.add_patch(card_m1)
    ax.text(0.48, 0.63, "LIVE TRAIT FORECAST", ha="center", va="top", fontsize=7.8, fontweight="bold")
    ax.text(0.35, 0.56, "OU Adaptive Ground Truth:", va="top", fontsize=6.6)
    ax.barh(0.51, 0.18, 0.025, left=0.35, facecolor="#1a1a1a", edgecolor="#000000")
    ax.text(0.35, 0.45, "Physics-Informed ML (PI-ML):", va="top", fontsize=6.6)
    ax.barh(0.40, 0.17, 0.025, left=0.35, facecolor="#666666", hatch="xx", edgecolor="#000000")
    ax.text(0.35, 0.34, "Naive ML (Unanchored):", va="top", fontsize=6.6)
    ax.barh(0.29, 0.24, 0.025, left=0.35, facecolor="#ffffff", hatch="...", edgecolor="#000000")

    # Middle Right: Confidence & Divergence Meter
    card_m2 = patches.Rectangle((0.66, 0.29), 0.30, 0.37, facecolor="#ffffff", edgecolor="#000000", linewidth=1.1)
    ax.add_patch(card_m2)
    ax.text(0.81, 0.63, "DEEP-TIME RELIABILITY", ha="center", va="top", fontsize=7.8, fontweight="bold")
    ax.text(0.68, 0.56, "• Extrapolation Confidence: 84.2%\n• Patristic Buffer: 148.5 Ma\n• Inflation Risk: Low (Whitened)\n• Thermal Scaling: Respected",
            va="top", linespacing=1.35, fontsize=6.6)
    meter = patches.Rectangle((0.68, 0.35), 0.24, 0.035, facecolor="#eeeeee", edgecolor="#000000")
    ax.add_patch(meter)
    fill = patches.Rectangle((0.68, 0.35), 0.20, 0.035, facecolor="#000000", edgecolor="#000000")
    ax.add_patch(fill)
    ax.text(0.80, 0.32, "84.2% Reliability Score", ha="center", va="center", fontsize=6.8, fontweight="bold")

    # Bottom Panel: Thermodynamic Dissipation Curve
    card_b = patches.Rectangle((0.33, 0.06), 0.63, 0.20, facecolor="#ffffff", edgecolor="#000000", linewidth=1.1)
    ax.add_patch(card_b)
    ax.text(0.35, 0.23, "THERMODYNAMIC COOLING LIMIT (FOURIER & KLEIBER SCALING)", va="top", fontsize=7.6, fontweight="bold")
    ax.text(0.35, 0.17, "Heat Production (M⁰·⁷⁵) vs Heat Dissipation (M⁰·⁶⁷·ΔT). At ΔT = +3.0°C, thermal dissipation drops by 14.8%,\nstrictly requiring body shrinkage (ΔM < 0). PI-ML enforces ∂ŷ/∂T ≤ κ_temp to eliminate unphysical mass growth.",
            va="top", linespacing=1.35, fontsize=6.6)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.01, 1.0)

    plt.tight_layout(pad=0.4)
    out_path = ASSETS_DIR / "fig_ui_layout_bw.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated -> {out_path}")


if __name__ == "__main__":
    generate_results_chart_bw()
    generate_divergence_decay_bw()
    generate_simulation_scenarios_bw()
    generate_system_overview_bw()
    generate_flowchart_bw()
    generate_ui_layout_bw()
    print("ALL B&W ASSETS REGENERATED WITH CLEAN ZERO-OVERLAP SPACING.")
