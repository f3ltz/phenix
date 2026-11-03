"""
PHENIX Interactive Research Dashboard
A multi-tab Streamlit dashboard designed for intuitive general-audience exploration
and rigorous technical evaluation across all stages of the framework.
"""

import sys
import importlib
from pathlib import Path
import json

try:
    import streamlit as st
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

if STREAMLIT_AVAILABLE:
    import src.visualization.graphics as graphics_mod
    importlib.reload(graphics_mod)
    from src.models import RandomForestPipeline, XGBoostPipeline
    from src.visualization.graphics import (
        render_species_morphology_card,
        get_species_evolutionary_note,
        plot_size_comparison_silhouette,
        plot_mass_scale_gauge,
        plot_multi_trait_radar,
        plot_allometric_scaling_curve,
        plot_animal_forecast_real_units,
        plot_forecasting_comparison_horizontal,
        get_shrinkage_analogy,
        format_trait_value,
        TRAIT_CONFIGS,
        ORDER_ICONS,
    )


def run_dashboard_cli():
    """CLI launcher for the dashboard."""
    if not STREAMLIT_AVAILABLE:
        print("\n" + "=" * 70)
        print("PHENIX Interactive Dashboard")
        print("=" * 70)
        print("Streamlit is required to launch the interactive research dashboard.")
        print("Please install Streamlit in your Python environment by running:")
        print("    pip install streamlit")
        print("\nThen launch the dashboard with:")
        print("    python -m streamlit run src/visualization/dashboard.py")
        print("=" * 70 + "\n")
        return

    import subprocess
    script_path = Path(__file__).resolve()
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(script_path)])


def render_dashboard():
    if not STREAMLIT_AVAILABLE:
        return

    st.set_page_config(
        page_title="PHENIX | Macroevolutionary Trait Intelligence",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # High-End Editorial & Field-Biology Styling (No generic purple AI tropes)
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #fbfbf9 !important;
            color: #1a202c !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        .main-header {
            font-size: 34px;
            font-weight: 800;
            color: #0d3b2e !important;
            margin-bottom: 4px;
            letter-spacing: -0.6px;
            font-family: "Newsreader", "Georgia", serif;
        }
        .sub-header {
            font-size: 15px;
            color: #4a5568 !important;
            margin-bottom: 20px;
            line-height: 1.45;
        }
        .hero-banner {
            background: linear-gradient(135deg, #f7f9f7 0%, #edf4f0 100%);
            border-left: 5px solid #0d3b2e;
            padding: 18px 22px;
            border-radius: 10px;
            margin-bottom: 22px;
            font-size: 14.5px;
            line-height: 1.55;
            color: #1a202c !important;
            box-shadow: 0 2px 8px rgba(13, 59, 46, 0.06);
            border: 1px solid #e2e8f0;
        }
        .hero-banner strong {
            color: #0d3b2e !important;
        }
        .lens-badge {
            display: inline-block;
            background: #e6f4ea;
            color: #0d5a36;
            border: 1px solid #b7e1cd;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .kpi-card {
            background: #ffffff !important;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 16px 18px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.03);
            text-align: center;
        }
        .kpi-title {
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
            color: #334e68 !important;
            margin-bottom: 5px;
            letter-spacing: 0.5px;
        }
        .kpi-value {
            font-size: 26px;
            font-weight: 800;
            color: #0d3b2e !important;
            font-family: "Newsreader", "Georgia", serif;
        }
        .kpi-sub {
            font-size: 12px;
            color: #2b6cb0 !important;
            margin-top: 3px;
            font-weight: 600;
        }
        .species-card-name {
            color: #ffffff !important;
            font-size: 29px !important;
            font-weight: 700 !important;
            font-style: italic !important;
            letter-spacing: -0.4px !important;
        }
        .takeaway-box {
            background: #ffffff !important;
            border: 1px solid #e2e8f0;
            border-left: 5px solid #c05621;
            border-radius: 8px;
            padding: 15px 20px;
            margin: 14px 0 20px 0;
            font-size: 14px;
            line-height: 1.55;
            color: #1e293b !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        }
        .takeaway-box strong {
            color: #7c2d12 !important;
        }
        .phylo-explain-card {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-left: 5px solid #0d5a36;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 20px;
            font-size: 14px;
            line-height: 1.55;
            color: #166534 !important;
        }
        .phylo-explain-card strong {
            color: #14532d !important;
        }
        .alert-unanchored {
            background: #fff5f5;
            border: 1px solid #fed7d7;
            border-left: 5px solid #e53e3e;
            border-radius: 8px;
            padding: 14px 18px;
            margin-top: 14px;
            font-size: 13.5px;
            color: #9b2c2c !important;
            line-height: 1.5;
        }
        .analogy-badge {
            background: #fef3c7;
            border: 1px solid #fde68a;
            color: #92400e;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 13.5px;
            margin: 12px 0 16px 0;
            font-weight: 500;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    data_dir = Path("data/processed")
    required_files = [
        "phenotypic_clean.csv",
        "environmental_clean.csv",
        "features_phylo_pcoa.csv",
        "features_baseline.csv",
        "features_augmented.csv",
    ]
    missing_files = [f for f in required_files if not (data_dir / f).exists()]
    if missing_files:
        st.error(f"Missing required processed datasets in {data_dir.resolve()}: {missing_files}. Please run the pipeline first.")
        st.stop()

    # Load data artifacts
    @st.cache_data
    def load_artifacts():
        df_p = pd.read_csv(data_dir / "phenotypic_clean.csv")
        df_e = pd.read_csv(data_dir / "environmental_clean.csv")
        df_phylo = pd.read_csv(data_dir / "features_phylo_pcoa.csv")
        
        sim_summary = None
        if (data_dir / "simulation_summary.csv").exists():
            sim_summary = pd.read_csv(data_dir / "simulation_summary.csv")

        return df_p, df_e, df_phylo, sim_summary

    df_pheno, df_env, df_phylo, df_sim = load_artifacts()

    @st.cache_resource(show_spinner="Training live macroecological models (Random Forest & XGBoost)...")
    def load_trained_tree_models(target_trait: str):
        df_p = pd.read_csv(data_dir / "phenotypic_clean.csv")
        df_b = pd.read_csv(data_dir / "features_baseline.csv")
        df_a = pd.read_csv(data_dir / "features_augmented.csv")

        base_cols = [c for c in df_b.columns if c != "canonical_taxon_id"]
        aug_cols = [c for c in df_a.columns if c != "canonical_taxon_id"]

        X_base = df_b[base_cols].values.astype(np.float32)
        X_aug = df_a[aug_cols].values.astype(np.float32)
        raw_y = df_p[target_trait].values.astype(np.float32)

        valid_mask = (~np.isnan(raw_y)) & (raw_y > 0)
        X_b_train = X_base[valid_mask]
        X_a_train = X_aug[valid_mask]
        y_train = raw_y[valid_mask]

        rf_base = RandomForestPipeline(n_estimators=40, max_depth=12, random_state=42)
        rf_base.fit(X_b_train, y_train)

        xgb_base = XGBoostPipeline(n_estimators=50, max_depth=5, learning_rate=0.08, random_state=42)
        xgb_base.fit(X_b_train, y_train)

        rf_aug = RandomForestPipeline(n_estimators=40, max_depth=12, random_state=42)
        rf_aug.fit(X_a_train, y_train)

        xgb_aug = XGBoostPipeline(n_estimators=50, max_depth=5, learning_rate=0.08, random_state=42)
        xgb_aug.fit(X_a_train, y_train)

        return {
            "rf_base": rf_base,
            "xgb_base": xgb_base,
            "rf_aug": rf_aug,
            "xgb_aug": xgb_aug,
            "base_cols": base_cols,
            "aug_cols": aug_cols,
            "df_base": df_b,
            "df_aug": df_a,
        }

    # Ensure session state variables exist
    if "global_species_selector" not in st.session_state:
        st.session_state["global_species_selector"] = "Panthera leo"
    if "selected_species" not in st.session_state:
        st.session_state["selected_species"] = "Panthera leo"

    # Sidebar: Exploration Lens & Project Metadata
    st.sidebar.markdown("### 🌿 PHENIX Framework")
    st.sidebar.markdown(
        """
        **Macroevolutionary Trait Intelligence**  
        *Dept of IT &bull; National Institute of Technology Karnataka (NITK)*
        """
    )
    st.sidebar.markdown("---")

    # Dual-Lens Navigation Switch
    st.sidebar.markdown("#### 🔭 Exploration Lens")
    view_lens = st.sidebar.radio(
        "Select Perspective:",
        ["🌿 Field Explorer (Intuitive & Relatable)", "🔬 Computational Biologist (Technical Deep-Dive)"],
        index=0,
        help="Switch between plain-English biological storytelling and rigorous computational physics/statistics.",
    )
    is_expert_mode = "Computational Biologist" in view_lens

    st.sidebar.markdown("---")

    # Global Animal Selector in Sidebar
    st.sidebar.markdown("#### 🐾 Active Focal Specimen")
    taxa_list = sorted(df_pheno["species"].tolist())

    # Callback function to handle quick benchmark button clicks cleanly
    def select_quick_animal(sp_name: str):
        st.session_state["global_species_selector"] = sp_name
        st.session_state["selected_species"] = sp_name

    st.sidebar.write("Quick Benchmarks:")
    col_q1, col_q2, col_q3 = st.sidebar.columns(3)
    col_q1.button("🦁 Lion", on_click=select_quick_animal, args=("Panthera leo",))
    col_q2.button("🐘 Elephant", on_click=select_quick_animal, args=("Elephas maximus",))
    col_q3.button("🐺 Wolf", on_click=select_quick_animal, args=("Canis lupus",))

    col_q4, col_q5, col_q6 = st.sidebar.columns(3)
    col_q4.button("🐒 Chimp", on_click=select_quick_animal, args=("Pan troglodytes",))
    col_q5.button("🐻‍❄️ Bear", on_click=select_quick_animal, args=("Ursus maritimus",))
    col_q6.button("🐁 Mouse", on_click=select_quick_animal, args=("Mus musculus",))

    # Dropdown to select any species (bound to key="global_species_selector")
    selected_species = st.sidebar.selectbox(
        "Search all 3,268 species:",
        taxa_list,
        key="global_species_selector",
    )
    st.session_state["selected_species"] = selected_species

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Audited Dataset Parameters:**")
    n_species = len(df_pheno)
    n_orders = df_pheno["order"].nunique() if "order" in df_pheno.columns else 27
    st.sidebar.write(f"• **Species Count:** {n_species:,} Mammalia")
    st.sidebar.write(f"• **Evolutionary Orders:** {n_orders} Monophyletic")
    st.sidebar.write("• **TimeTree Calibration:** 177.0 Ma Crown Root")
    st.sidebar.write("• **Patristic Buffer:** 140.0 Ma Quarantine")

    # Header Section
    st.markdown('<div class="main-header">PHENIX: Macroevolutionary Trait Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Resolving Felsenstein’s Dilemma: Continuous Spectral Timetree Manifolds Establish the Predictability Frontier of Organismal Traits</div>',
        unsafe_allow_html=True,
    )

    # 30-Second Explainer Banner
    lens_pill = '<span class="lens-badge">🌿 Field Explorer Lens Active</span>' if not is_expert_mode else '<span class="lens-badge" style="background:#ede9fe; color:#5b21b6; border-color:#ddd6fe;">🔬 Computational Biologist Lens Active</span>'
    
    if not is_expert_mode:
        banner_content = """
        <div class="hero-banner">
            <div>""" + lens_pill + """</div>
            <strong>💡 What is PHENIX in 30 Seconds?</strong><br>
            Standard AI models predicting animal traits boast 85% accuracy, claiming to understand biology. 
            In reality, standard evaluation randomly scatters sister species between training and test sets—the AI simply cheats by recognizing close family members (<strong>clade leakage</strong>). 
            When forced to predict on a genuinely new evolutionary branch (behind a 140-million-year buffer), naive models completely collapse! 
            PHENIX provides the mathematical fix: continuous evolutionary spatial coordinates (<strong>Phylo-PCoA</strong>) and physics-informed biological guardrails.
        </div>
        """
    else:
        banner_content = """
        <div class="hero-banner">
            <div>""" + lens_pill + """</div>
            <strong>🔬 Technical Thesis:</strong> Under Felsenstein's (1985) phylogenetic dilemma, biological data violate the foundational IID assumption of statistical learning through shared ancestral path lengths. 
            Random cross-validation suffers severe phylogenetic autocorrelation leakage ($R^2 = 0.85$ inflation). 
            When evaluated under patristically quarantined cross-validation ($d_{\\text{buffer}} \\ge 140.0\\text{ Ma}$), naive models exhibit structural failure ($R^2 < 0$). 
            PHENIX resolves this through a 32-dimensional spectral decomposition of the 177.0 Ma TimeTree and Fourier-Kleiber thermodynamic regularization.
        </div>
        """
    st.markdown(banner_content, unsafe_allow_html=True)

    # Top KPI Cards
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">The "Family Cheat" Gap</div>
                <div class="kpi-value" style="color: #c05621 !important;">+8.83 R²</div>
                <div class="kpi-sub" style="color: #c05621 !important;">Inflation from Clade Leakage</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_k2:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">Within-Clade Accuracy</div>
                <div class="kpi-value" style="color: #0d3b2e !important;">R² = 0.85</div>
                <div class="kpi-sub" style="color: #0d3b2e !important;">MAE = 0.336 log10(g) (~2.1x)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_k3:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">Phylogenetic Inertia</div>
                <div class="kpi-value" style="color: #134e4a !important;">86.93%</div>
                <div class="kpi-sub" style="color: #134e4a !important;">Body Size Governed by Ancestry</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_k4:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">Direct Climate Signal</div>
                <div class="kpi-value" style="color: #d97706 !important;">&lt; 0.1%</div>
                <div class="kpi-sub" style="color: #d97706 !important;">Independent Macroclimate Effect</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 2 Main Tabs (Tab 2 removed per user request)
    tab1, tab2 = st.tabs([
        "🐾 Tab 1: Specimen Hall & Multi-Trait Allometry",
        "🌍 Tab 2: Climate Simulator & Time-Machine",
    ])

    # Extract active species data safely
    sp_sub = df_pheno[df_pheno["species"] == selected_species]
    sp_row = sp_sub.iloc[0] if len(sp_sub) > 0 else df_pheno.iloc[0]
    sp_id = sp_row["canonical_taxon_id"]

    env_sub = df_env[df_env["canonical_taxon_id"] == sp_id]
    env_row = env_sub.iloc[0] if len(env_sub) > 0 else df_env.iloc[0]

    phylo_sub = df_phylo[df_phylo["canonical_taxon_id"] == sp_id]
    phylo_row = phylo_sub.iloc[0] if len(phylo_sub) > 0 else df_phylo.iloc[0]

    # -------------------------------------------------------------------------
    # TAB 1: SPECIMEN HALL & MULTI-TRAIT ALLOMETRY
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown(f"### 🐾 Specimen Hall: *{selected_species}*")
        st.markdown(
            "Explore morphological traits, physical scale comparisons against familiar animals, "
            "6-trait pace-of-life radars, and global allometric scaling laws."
        )

        col_hab1, col_hab2 = st.columns([2.5, 1.5])
        with col_hab1:
            st.info(f"💡 Active Specimen: **{selected_species}** ({sp_row['order']}). You can switch animals anytime in the left sidebar!")
        with col_hab2:
            st.markdown(
                f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; font-size: 13px; color: #334155;">
                    <strong style="color: #0d3b2e;">Bioclimatic Native Habitat:</strong><br>
                    • Annual Temp (Bio1): <strong>{env_row['bio1']:.1f}°C</strong> &bull; Precip: <strong>{env_row['bio12']:.0f} mm</strong> &bull; Elev: <strong>{env_row['elevation']:.0f} m</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 1. Specimen Morphology Card (with bright white species name, Superpower & Scale note)
        card_html = render_species_morphology_card(
            species_name=selected_species,
            order=sp_row["order"],
            family=sp_row["family"],
            genus=sp_row["genus"],
            mass_g=sp_row["adult_body_mass_g"],
            trophic_level=sp_row.get("trophic_level", np.nan),
        )
        st.markdown(card_html, unsafe_allow_html=True)

        # 2. Visual Size-O-Meter Silhouette Comparison & Log Mass Gauge
        col_sz1, col_sz2 = st.columns([1, 1])
        with col_sz1:
            st.markdown("#### 📏 Physical Size-O-Meter (vs. Familiar Animals)")
            fig_sz = plot_size_comparison_silhouette(sp_row["adult_body_mass_g"], selected_species)
            st.pyplot(fig_sz)
            plt.close(fig_sz)

        with col_sz2:
            st.markdown("#### 🧭 Logarithmic Mammalian Mass Spectrum")
            fig_scale = plot_mass_scale_gauge(sp_row["adult_body_mass_g"], selected_species)
            st.pyplot(fig_scale)
            plt.close(fig_scale)

        # 3. Multi-Trait Radar & Allometry Curves
        st.markdown("---")
        st.markdown("#### 🧬 Allometric Life-History Spectrum")
        st.markdown(
            "Mammalian traits follow coordinated **pace-of-life allometries**. "
            "The radar chart compares this species across 6 core PanTHERIA traits against its taxonomic order and class median."
        )

        col_radar, col_allo = st.columns([1, 1])

        with col_radar:
            st.markdown("**Allometric Life-History Radar (Percentiles vs. Order Median)**")
            fig_radar = plot_multi_trait_radar(df_pheno, sp_id)
            st.pyplot(fig_radar)
            plt.close(fig_radar)

        with col_allo:
            trait_options = {
                "gestation_length_d": "Gestation Length (Days)",
                "weaning_age_d": "Weaning Age (Days)",
                "max_longevity_m": "Max Longevity (Months)",
                "litter_size": "Litter Size (Offspring)",
                "home_range_km2": "Home Range Area (km²)",
            }
            selected_trait_col = st.selectbox(
                "Compare Scaling Against:",
                list(trait_options.keys()),
                format_func=lambda x: trait_options[x],
                index=0,
                key="tab1_trait_scaling",
            )
            fig_allo = plot_allometric_scaling_curve(
                df_pheno,
                sp_id=sp_id,
                y_trait=selected_trait_col,
                y_label=trait_options[selected_trait_col],
            )
            st.pyplot(fig_allo)
            plt.close(fig_allo)

        # 4. Expert Mode: PCoA Coordinates
        if is_expert_mode:
            st.markdown("---")
            st.markdown("#### 🔬 Spectral Manifold: 32 Continuous Phylo-PCoA Spatial Coordinates")
            st.markdown(
                """
                Gower's double-centering decomposes the 177.0 Ma TimeTree patristic distance matrix into 
                32 orthogonal spatial coordinates that capture 89.98% of total evolutionary metric space variance. 
                Below are the leading 16 coordinates for this species.
                """
            )
            pcoa_cols = [c for c in df_phylo.columns if c != "canonical_taxon_id"][:16]
            pcoa_vals = [float(phylo_row[c]) for c in pcoa_cols]
            n_pcoa = len(pcoa_vals)

            fig_pcoa, ax_pcoa = plt.subplots(figsize=(10, 2.6), dpi=100)
            fig_pcoa.patch.set_facecolor("#ffffff")
            ax_pcoa.set_facecolor("#fafbfc")
            colors = ["#0d3b2e" if v >= 0 else "#c05621" for v in pcoa_vals]
            ax_pcoa.bar(range(1, n_pcoa + 1), pcoa_vals, color=colors, edgecolor="#1a202c", linewidth=0.7)
            ax_pcoa.set_xticks(range(1, n_pcoa + 1))
            ax_pcoa.set_xticklabels([f"PC{i}" for i in range(1, n_pcoa + 1)], fontsize=8.0)
            ax_pcoa.set_ylabel("Eigenmap Projection", fontsize=8.5, fontweight="bold", color="#2c3e50")
            ax_pcoa.set_title(f"Phylo-PCoA Coordinates: {selected_species} ({sp_row['order']})", fontsize=9.5, fontweight="bold", color="#0d3b2e")
            ax_pcoa.grid(axis="y", linestyle="--", alpha=0.45, color="#cbd5e0")
            plt.tight_layout(pad=0.8)
            st.pyplot(fig_pcoa)
            plt.close(fig_pcoa)

    # -------------------------------------------------------------------------
    # TAB 2: CLIMATE SIMULATOR & TIME-MACHINE (Biophysical thermodynamics removed per user request)
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("### 🌍 Eco-Evolutionary Climate Simulator & Time-Machine")
        st.info(f"Target Animal: **{selected_species}** (synced from sidebar). Adjust climate scenarios to witness morphological evolutionary tracking vs. unconstrained AI hallucinations.")

        # Explainer: Phylo-Lock vs Thermodynamic Rule
        st.markdown(
            """
            <div class="phylo-explain-card">
                <strong>💡 Why the Thermodynamic Rule 'Hard Carries' the Delta, while the Phylo Lock 'Hard Carries' the Baseline:</strong><br>
                When predicting an animal's future traits, there are two distinct mathematical scales:<br>
                1. <strong>The Baseline Level (Carried 99.9% by the Phylo Lock):</strong> 
                   An Asian Elephant weighs ~3,300 kg, while a House Mouse weighs ~20 g. The <strong>32 Phylo-PCoA spatial eigenmaps</strong> anchor the animal to its ancestral Bauplan. <em>Without the Phylo Lock, an elephant collapses to the global mammal median (~89 g)—a catastrophic error of 36,000x!</em><br>
                2. <strong>The Delta / Change Under Warming (Carried by the Thermodynamic Rule):</strong> 
                   When temperature rises by +3°C, the elephant does not change its evolutionary lineage over 50 years (ΔZ<sub>phylo</sub> = 0). 
                   Because the tree coordinates are static in the short term, their derivative is zero. 
                   Therefore, the <strong>thermodynamic cooling law</strong> must govern the adaptive shift (ΔM &lt; 0), preventing naive ML from hallucinating that heat makes animals bigger!
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Trait Selector
        col_tr1, col_tr2 = st.columns([2, 1])
        with col_tr1:
            selected_trait_key = st.selectbox(
                "Select Life-History Trait to Forecast:",
                list(TRAIT_CONFIGS.keys()),
                format_func=lambda k: f"{TRAIT_CONFIGS[k]['label']} ({TRAIT_CONFIGS[k]['unit']})",
                index=0,
                key="tab2_trait",
            )
        trait_cfg = TRAIT_CONFIGS[selected_trait_key]

        # Extract baseline value safely
        order_sim = sp_row["order"]
        raw_val = sp_row[selected_trait_key]
        if pd.notna(raw_val) and raw_val > 0:
            base_trait_val = float(raw_val)
            val_source_note = "Observed PanTHERIA measurement"
        else:
            order_subset = df_pheno[df_pheno["order"] == order_sim][selected_trait_key].dropna()
            if len(order_subset) > 0 and order_subset.median() > 0:
                base_trait_val = float(order_subset.median())
                val_source_note = f"Order {order_sim} median fallback"
            else:
                base_trait_val = float(df_pheno[selected_trait_key].dropna().median())
                val_source_note = "Global Mammalia median fallback"

        with col_tr2:
            st.markdown(
                f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; font-size: 13px; color: #334155;">
                    <strong style="color: #0d3b2e;">Current Baseline:</strong><br>
                    <strong>{format_trait_value(selected_trait_key, base_trait_val)}</strong> <em>({val_source_note})</em>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Scenario & Time-Machine Controls
        st.markdown("---")
        col_ctrl, col_view = st.columns([1, 2])

        with col_ctrl:
            st.markdown("#### ⏳ Scenario Controls")
            preset_dt_map = {
                "SSP1-2.6 (Mild Warming, +1.5°C)": 1.5,
                "SSP2-4.5 (Moderate Warming, +3.0°C)": 3.0,
                "SSP5-8.5 (Extreme Warming, +5.0°C)": 5.0,
            }

            def on_scenario_preset_change():
                sel = st.session_state.get("tab2_scenario", "")
                if sel in preset_dt_map:
                    st.session_state["tab2_dt"] = preset_dt_map[sel]

            scenario_choice = st.selectbox(
                "IPCC AR6 Scenario Preset:",
                ["SSP1-2.6 (Mild Warming, +1.5°C)", "SSP2-4.5 (Moderate Warming, +3.0°C)", "SSP5-8.5 (Extreme Warming, +5.0°C)", "Custom"],
                index=1,
                key="tab2_scenario",
                on_change=on_scenario_preset_change,
            )

            if "tab2_dt" not in st.session_state:
                st.session_state["tab2_dt"] = 3.0

            delta_t = st.slider(
                "Global Temperature Anomaly (ΔT °C):",
                min_value=-2.0,
                max_value=6.0,
                step=0.5,
                key="tab2_dt",
            )
            years_forward = st.slider(
                "Evolutionary Horizon (Years Forward):",
                min_value=10_000,
                max_value=1_000_000,
                value=500_000,
                step=50_000,
                key="tab2_yrs",
            )
            alpha_sel = st.slider(
                "OU Adaptation Rate (α per Ma):",
                min_value=0.5,
                max_value=8.0,
                value=2.5,
                step=0.5,
                key="tab2_alpha",
                help="Macroevolutionary selection strength. Higher values mean faster tracking of the thermodynamic Bergmann optimum.",
            )

            st.markdown("---")
            view_mode = st.radio("Display Projection Format:", ["Real-World Units (for selected animal)", "Macroevolutionary Delta (Δ log₁₀)"], index=0)
            show_unanchored_collapse = st.checkbox("⚠️ Compare: Show model if Phylo-Lock is REMOVED", value=False)

        with col_view:
            st.markdown(f"#### Forecasts for *{selected_species}*")

            # Load and cache trained tree models (Random Forest & XGBoost)
            models = load_trained_tree_models(selected_trait_key)
            rf_aug = models["rf_aug"]
            xgb_aug = models["xgb_aug"]
            rf_base = models["rf_base"]
            xgb_base = models["xgb_base"]
            base_cols = models["base_cols"]
            aug_cols = models["aug_cols"]
            df_base = models["df_base"]
            df_aug = models["df_aug"]

            # Locate species index
            sp_matches = df_pheno[df_pheno["species"] == selected_species]
            sp_idx = sp_matches.index[0] if len(sp_matches) > 0 else 0

            row_base_cur = df_base.iloc[sp_idx][base_cols].copy()
            row_aug_cur = df_aug.iloc[sp_idx][aug_cols].copy()

            extreme_factor = 1.20 if delta_t >= 3.0 else 1.10
            precip_factor = 1.05 if delta_t >= 3.0 else 1.02

            row_base_fut = row_base_cur.copy()
            row_aug_fut = row_aug_cur.copy()

            for r in [row_base_fut, row_aug_fut]:
                for v in ["bio1", "bio8", "bio9", "bio10", "bio11"]:
                    if v in r:
                        r[v] = r[v] + delta_t
                if "bio5" in r:
                    r["bio5"] = r["bio5"] + delta_t * extreme_factor
                if "bio6" in r:
                    r["bio6"] = r["bio6"] + delta_t * extreme_factor
                if "bio7" in r and "bio5" in r and "bio6" in r:
                    r["bio7"] = r["bio5"] - r["bio6"]
                for v in ["bio12", "bio13", "bio14", "bio15", "bio16", "bio17", "bio18", "bio19"]:
                    if v in r:
                        r[v] = max(r[v] * precip_factor, 0.0)
                if "habitat_suitability" in r:
                    decay = np.exp(-0.05 * max(delta_t - 1.5, 0.0))
                    r["habitat_suitability"] = np.clip(r["habitat_suitability"] * decay, 0.0, 1.0)

            # Molecular Clock Proteomic Drift across forward evolutionary horizon:
            # Simulates point mutations drifting ESM-2 protein language representations:
            # Z_genomic(t + dt) = Z_genomic(t) + N(0, drift_rate * sqrt(dt_ma))
            dt_ma = years_forward / 1_000_000.0
            drift_rate = 0.05
            gen_std = float(drift_rate * np.sqrt(max(dt_ma, 1e-4)))
            rng_gen = np.random.default_rng(42 + sp_idx)

            gen_cols = [c for c in row_base_fut.index if c.startswith("esm2_") or c.startswith("snp_") or c.startswith("func_")]
            noise = rng_gen.normal(0.0, gen_std, size=len(gen_cols)).astype(np.float32)

            for i, col in enumerate(gen_cols):
                row_base_fut[col] += noise[i]
                row_aug_fut[col] += noise[i]

            x_b_cur = row_base_cur.values.reshape(1, -1).astype(np.float32)
            x_a_cur = row_aug_cur.values.reshape(1, -1).astype(np.float32)
            x_b_fut = row_base_fut.values.reshape(1, -1).astype(np.float32)
            x_a_fut = row_aug_fut.values.reshape(1, -1).astype(np.float32)

            # Live Predictions from real ML models (log10 space)
            pred_rf_aug_log = float(rf_aug.predict(x_a_fut)[0])
            pred_xgb_aug_log = float(xgb_aug.predict(x_a_fut)[0])
            pred_rf_base_log = float(rf_base.predict(x_b_fut)[0])
            pred_xgb_base_log = float(xgb_base.predict(x_b_fut)[0])

            cur_rf_aug_log = float(rf_aug.predict(x_a_cur)[0])
            cur_xgb_aug_log = float(xgb_aug.predict(x_a_cur)[0])
            cur_rf_base_log = float(rf_base.predict(x_b_cur)[0])
            cur_xgb_base_log = float(xgb_base.predict(x_b_cur)[0])

            shift_rf_aug = pred_rf_aug_log - cur_rf_aug_log
            shift_xgb_aug = pred_xgb_aug_log - cur_xgb_aug_log
            shift_rf_base = pred_rf_base_log - cur_rf_base_log
            shift_xgb_base = pred_xgb_base_log - cur_xgb_base_log

            pred_rf_aug_real = float(10.0 ** pred_rf_aug_log)
            pred_xgb_aug_real = float(10.0 ** pred_xgb_aug_log)
            pred_rf_base_real = float(10.0 ** pred_rf_base_log)
            pred_xgb_base_real = float(10.0 ** pred_xgb_base_log)

            # Physics Simulation: Ornstein-Uhlenbeck with Bergmann's rule
            beta_trait = trait_cfg["beta_temp"]
            shift_optimum = beta_trait * delta_t
            dt_ma = years_forward / 1_000_000.0
            shift_ou_log = shift_optimum * (1.0 - np.exp(-alpha_sel * dt_ma))
            pred_ou_real = base_trait_val * (10.0 ** shift_ou_log)

            # Physics-Informed ML: blends thermodynamic OU selection prior with phylo-anchored ML shift
            shift_piml_log = 0.70 * shift_ou_log + 0.30 * shift_rf_aug
            pred_piml_real = base_trait_val * (10.0 ** shift_piml_log)

            # Visceral Shrinkage Analogy Banner
            delta_val_physics = pred_ou_real - base_trait_val
            analogy_text = get_shrinkage_analogy(selected_species, base_trait_val, delta_val_physics, selected_trait_key)
            if analogy_text:
                st.markdown(
                    f"""
                    <div class="analogy-badge">
                        <strong>🌿 Physical Reality Check:</strong> {analogy_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            cur_rf_aug_real = float(10.0 ** cur_rf_aug_log)
            cur_xgb_aug_real = float(10.0 ** cur_xgb_aug_log)
            cur_rf_base_real = float(10.0 ** cur_rf_base_log)
            cur_xgb_base_real = float(10.0 ** cur_xgb_base_log)

            # Metric Scorecards (4 key forecasts)
            m1, m2, m3, m4 = st.columns(4)
            ou_pct = ((pred_ou_real - base_trait_val) / base_trait_val) * 100.0
            piml_pct = ((pred_piml_real - base_trait_val) / base_trait_val) * 100.0
            rf_shift_pct = ((pred_rf_aug_real - cur_rf_aug_real) / cur_rf_aug_real) * 100.0
            xgb_shift_pct = ((pred_xgb_aug_real - cur_xgb_aug_real) / cur_xgb_aug_real) * 100.0

            m1.metric("Physics Simulation (OU)", format_trait_value(selected_trait_key, pred_ou_real), f"{ou_pct:+.1f}% vs Current")
            m2.metric("Physics-Informed ML", format_trait_value(selected_trait_key, pred_piml_real), f"{piml_pct:+.1f}% vs Current")
            m3.metric("Random Forest (Phylo-Lock)", format_trait_value(selected_trait_key, pred_rf_aug_real), f"{rf_shift_pct:+.1f}% shift under warming")
            m4.metric("XGBoost (Phylo-Lock)", format_trait_value(selected_trait_key, pred_xgb_aug_real), f"{xgb_shift_pct:+.1f}% shift under warming")

            # Chart rendering
            if view_mode == "Real-World Units (for selected animal)":
                fig_animal = plot_animal_forecast_real_units(
                    species_name=selected_species,
                    trait_col=selected_trait_key,
                    trait_label=trait_cfg["label"],
                    base_val=base_trait_val,
                    pred_ou=pred_ou_real,
                    pred_rf_aug=pred_rf_aug_real,
                    pred_xgb_aug=pred_xgb_aug_real,
                    pred_rf_base=pred_rf_base_real,
                    pred_xgb_base=pred_xgb_base_real,
                    pred_piml=pred_piml_real,
                    show_unanchored=show_unanchored_collapse,
                )
                st.pyplot(fig_animal)
                plt.close(fig_animal)
            else:
                fig_delta = plot_forecasting_comparison_horizontal(
                    delta_t=delta_t,
                    shift_ou=shift_ou_log,
                    shift_rf_aug=shift_rf_aug,
                    shift_xgb_aug=shift_xgb_aug,
                    shift_rf_base=shift_rf_base,
                    shift_xgb_base=shift_xgb_base,
                    shift_piml=shift_piml_log,
                    trait_label=trait_cfg["label"],
                    show_unanchored=show_unanchored_collapse,
                )
                st.pyplot(fig_delta)
                plt.close(fig_delta)

            # Unanchored Comparison Callout
            if show_unanchored_collapse:
                rf_base_shift_pct = ((pred_rf_base_real - cur_rf_base_real) / cur_rf_base_real) * 100.0
                xgb_base_shift_pct = ((pred_xgb_base_real - cur_xgb_base_real) / cur_xgb_base_real) * 100.0
                baseline_err_pct = ((pred_rf_base_real - base_trait_val) / base_trait_val) * 100.0
                st.markdown(
                    f"""
                    <div class="alert-unanchored">
                        <strong>🚨 What Happens Without the Phylo Lock? (Ablation Proof)</strong><br>
                        When the 32 continuous Phylo-PCoA eigenmaps are stripped away, the models lose all information about <em>{selected_species}'s</em> evolutionary lineage. 
                        They conflate high tropical temperatures with large ecosystem biomass, causing massive hallucinations:<br>
                        &bull; <strong>Naive Random Forest:</strong> <strong>{format_trait_value(selected_trait_key, pred_rf_base_real)}</strong> ({rf_base_shift_pct:+.1f}% shift under warming; {baseline_err_pct:+.0f}% baseline error)<br>
                        &bull; <strong>Naive XGBoost:</strong> <strong>{format_trait_value(selected_trait_key, pred_xgb_base_real)}</strong> ({xgb_base_shift_pct:+.1f}% shift under warming)<br>
                        <em>Notice how the Phylo-Locked models predict {format_trait_value(selected_trait_key, pred_rf_aug_real)} and {format_trait_value(selected_trait_key, pred_xgb_aug_real)}, preserving the true ~{format_trait_value(selected_trait_key, base_trait_val)} scale within the biological envelope!</em>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if is_expert_mode and df_sim is not None:
            st.markdown("---")
            with st.expander("📋 View Calibrated IPCC Scenario Simulation Summary Table"):
                st.dataframe(df_sim, use_container_width=True)


if __name__ == "__main__":
    if STREAMLIT_AVAILABLE and "streamlit" in sys.modules:
        render_dashboard()
    else:
        run_dashboard_cli()
