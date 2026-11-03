"""
PHENIX Graphics & Species Visualization Module
Provides:
  1. Morphological silhouettes and order badges for mammalian taxa.
  2. Overlap-free logarithmic mass spectrum gauge.
  3. Multi-trait allometric radar/spider charts (Mass, Gestation, Litter, Longevity, Home Range).
  4. Macroecological allometric scaling curves.
  5. Macroevolutionary Clade Buffer Quarantine diagram (Tab 2).
  6. Evolutionary Divergence vs. Accuracy Decay tracking plot (Tab 2).
  7. Thermodynamic Heat Dissipation Biophysics diagram (Tab 3).
  8. Real-world animal trait forecasting bar chart (Tab 3).
  9. Overlap-free horizontal forward forecasting comparison bar chart (Tab 3).
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Morphological order icons
ORDER_ICONS = {
    "Carnivora": "🦁",
    "Cetartiodactyla": "🦌",
    "Artiodactyla": "🦌",
    "Chiroptera": "🦇",
    "Rodentia": "🐿️",
    "Primates": "🐒",
    "Proboscidea": "🐘",
    "Perissodactyla": "🦏",
    "Diprotodontia": "🦘",
    "Lagomorpha": "🐇",
    "Soricomorpha": "🦔",
    "Eulipotyphla": "🦔",
    "Cingulata": "🛡️",
    "Pilosa": "🦥",
    "Afrosoricida": "🦔",
    "Didelphimorphia": "🐀",
    "Dasyuromorphia": "🐾",
    "Monotremata": "🦆",
    "Scandentia": "🐿️",
    "Dermoptera": "🦇",
    "Hyracoidea": "🦡",
    "Sirenia": "🧜",
    "Tubulidentata": "🐜",
    "Pholidota": "🦎",
    "Macroscelidea": "🐭",
    "Paucituberculata": "🐭",
    "Microbiotheria": "🐭",
    "Notoryctemorphia": "⛏️",
}

# 6 Well-spaced standard reference benchmarks for body mass (in grams)
SPACED_BENCHMARKS = [
    ("Etruscan Shrew", 1.8, "0.3"),
    ("House Mouse", 20.0, "1.3"),
    ("Domestic Cat", 4_000.0, "3.6"),
    ("Human", 70_000.0, "4.8"),
    ("African Elephant", 4_500_000.0, "6.7"),
    ("Blue Whale", 100_000_000.0, "8.0"),
]

# Supported phenotypic traits configuration for the Multi-Trait Sandbox
TRAIT_CONFIGS = {
    "adult_body_mass_g": {
        "label": "Adult Body Mass",
        "unit": "Mass (g, kg, tonnes)",
        "beta_temp": -0.015,  # -3.4% per deg C
        "description": "Thermodynamic cooling limit (Bergmann's rule): surface-area-to-volume ratio must increase under thermal stress, requiring smaller body size.",
    },
    "gestation_length_d": {
        "label": "Gestation Length",
        "unit": "Days",
        "beta_temp": -0.008,  # -1.8% per deg C
        "description": "Metabolic turnover & embryogenesis: elevated temperature accelerates biochemical kinetics, shortening gestation time.",
    },
    "max_longevity_m": {
        "label": "Maximum Longevity",
        "unit": "Months / Years",
        "beta_temp": -0.010,  # -2.3% per deg C
        "description": "Metabolic rate of living theory: higher kinetic turnover accelerates cellular oxidative stress ('live fast, die young'), reducing lifespan.",
    },
    "litter_size": {
        "label": "Litter Size",
        "unit": "Offspring / Litter",
        "beta_temp": +0.012,  # +2.8% per deg C
        "description": "r/K reproductive strategy: thermal instability and higher adult mortality favor higher reproductive fecundity.",
    },
    "weaning_age_d": {
        "label": "Weaning Age",
        "unit": "Days",
        "beta_temp": -0.007,  # -1.6% per deg C
        "description": "Maternal lactation budget: juvenile growth rates accelerate under higher basal temperature, shortening weaning age.",
    },
    "home_range_km2": {
        "label": "Home Range Area",
        "unit": "km²",
        "beta_temp": -0.014,  # -3.2% per deg C
        "description": "Ecosystem productivity & foraging cost: warmer regions support higher primary biomass density, contracting required territory size.",
    },
}


def format_trait_value(trait_col: str, val: float) -> str:
    """Formats numeric trait values into clean human-readable units."""
    if pd.isna(val) or val <= 0:
        return "N/A"

    if trait_col == "adult_body_mass_g":
        if val >= 1_000_000:
            return f"{val / 1_000_000:.2f} tonnes"
        elif val >= 1_000:
            return f"{val / 1_000:.2f} kg"
        else:
            return f"{val:.1f} g"
    elif trait_col == "gestation_length_d":
        if val >= 60:
            return f"{val:.0f} days ({val / 30.4:.1f} mos)"
        return f"{val:.1f} days"
    elif trait_col == "max_longevity_m":
        years = val / 12.0
        return f"{years:.1f} yrs ({val:.0f} mos)"
    elif trait_col == "litter_size":
        return f"{val:.2f} offspring"
    elif trait_col == "weaning_age_d":
        return f"{val:.0f} days"
    elif trait_col == "home_range_km2":
        if val < 0.1:
            return f"{val * 100:.1f} ha"
        return f"{val:.2f} km²"
    return f"{val:.2f}"


def render_species_morphology_card(
    species_name: str,
    order: str,
    family: str,
    genus: str,
    mass_g: float,
    trophic_level: Optional[float] = None,
) -> str:
    """Generates a high-contrast, editorial botanical/field-biology card for the species."""
    icon = ORDER_ICONS.get(order, "🐾")
    
    # Trophic badge
    if pd.notna(trophic_level) and int(round(trophic_level)) == 1:
        trophic_str = '<span style="background: #e6f4ea; color: #0d5a36; border: 1px solid #b7e1cd; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 11.5px;">🌿 Herbivore</span>'
    elif pd.notna(trophic_level) and int(round(trophic_level)) == 2:
        trophic_str = '<span style="background: #fef3c7; color: #92400e; border: 1px solid #fde68a; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 11.5px;">🍎 Omnivore</span>'
    elif pd.notna(trophic_level) and int(round(trophic_level)) == 3:
        trophic_str = '<span style="background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 11.5px;">🥩 Carnivore</span>'
    else:
        trophic_str = '<span style="background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 11.5px;">❓ Trophic Unknown</span>'

    # Human-readable weight
    mass_display = format_trait_value("adult_body_mass_g", mass_g)
    field_note = get_species_evolutionary_note(species_name, order, mass_g, trophic_level)

    html = f"""
    <div style="background: linear-gradient(135deg, #0a3622 0%, #134e4a 65%, #1b3d2f 100%); color: #ffffff; padding: 22px 26px; border-radius: 12px; margin-bottom: 16px; box-shadow: 0 4px 16px rgba(10,54,34,0.18); border-left: 6px solid #d97706;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px;">
            <div style="display: flex; align-items: center; gap: 18px;">
                <div style="font-size: 46px; background: rgba(255,255,255,0.15); border-radius: 14px; width: 72px; height: 72px; display: flex; align-items: center; justify-content: center; box-shadow: inset 0 0 10px rgba(0,0,0,0.12); border: 1px solid rgba(255,255,255,0.2);">
                    {icon}
                </div>
                <div>
                    <div class="species-card-name" style="margin: 0; font-size: 29px; font-weight: 700; font-style: italic; color: #ffffff !important; letter-spacing: -0.4px; font-family: 'Newsreader', 'Georgia', serif;">{species_name}</div>
                    <div style="font-size: 13.5px; color: #ccfbf1 !important; margin-top: 4px;">
                        Order: <strong style="color: #ffffff !important;">{order}</strong> &bull; Family: <strong style="color: #ffffff !important;">{family}</strong> &bull; Genus: <strong style="color: #ffffff !important;">{genus}</strong>
                    </div>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 26px; font-weight: 800; color: #fde047; text-shadow: 0 1px 4px rgba(0,0,0,0.4);">{mass_display}</div>
                <div style="margin-top: 6px;">{trophic_str}</div>
            </div>
        </div>
        <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.15); font-size: 13px; color: #f0fdf4; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
            <div>
                <strong style="color: #fde047;">🌟 Evolutionary Superpower:</strong> {field_note['title']} — <span style="color: #d1fae5;">{field_note['description']}</span>
            </div>
            <div style="background: rgba(0,0,0,0.22); padding: 3px 9px; border-radius: 6px; font-size: 12px; color: #fde047;">
                📏 {field_note['scale_text']}
            </div>
        </div>
    </div>
    """
    return html


def get_species_evolutionary_note(
    species_name: str,
    order: str,
    mass_g: float,
    trophic_level: Optional[float] = None,
) -> Dict[str, str]:
    """Generates an engaging, human-readable field note and scale comparison."""
    if mass_g >= 1_000_000:
        multiplier = mass_g / 70_000.0
        scale_txt = f"Weighs as much as ~{multiplier:.0f} adult humans"
    elif mass_g >= 70_000:
        multiplier = mass_g / 70_000.0
        scale_txt = f"~{multiplier:.1f}× the weight of an adult human"
    elif mass_g >= 4_000:
        multiplier = mass_g / 4_000.0
        scale_txt = f"Equivalent to ~{multiplier:.1f} domestic cats"
    elif mass_g >= 100:
        multiplier = mass_g / 20.0
        scale_txt = f"Weighs about {multiplier:.0f} house mice"
    else:
        scale_txt = f"Lighter than a standard chicken egg ({mass_g:.1f} g)"

    order_superpowers = {
        "Proboscidea": (
            "The Giant Convective Radiator",
            "At over 3 tonnes, surface-area-to-volume ratio is minimal. To avoid hyperthermia without sweating, they use massive vascularized ear pinnae as convective thermal cooling fans.",
        ),
        "Chiroptera": (
            "Aerodynamic Hyper-Metabolism",
            "The only true flying mammals. Powering sustained flapping flight requires the highest mass-specific metabolic rate among mammals, demanding rapid embryogenesis and tight thermal dissipation.",
        ),
        "Cetartiodactyla": (
            "The Allometric Titans",
            "Spanning from 2 kg chevrotains to 100-tonne blue whales, this lineage demonstrates that macroecological allometric power laws remain unbroken across 8 orders of magnitude.",
        ),
        "Artiodactyla": (
            "Foregut Fermentation Engine",
            "Multi-chambered rumen digestive architecture allows massive biomass extraction from high-cellulose forage across global grasslands.",
        ),
        "Carnivora": (
            "Apex Energetics & Macro-Ranges",
            "Carnivores require expansive home ranges scaling to M^1.1 to sustain high trophic intake. Ambient warming accelerates basal metabolism, forcing territorial range expansion.",
        ),
        "Rodentia": (
            "The Evolutionary Blitzkrieg",
            "Representing over 40% of mammalian species, rodents leverage rapid developmental turnover and large litters to survive rapid macroclimatic oscillations.",
        ),
        "Primates": (
            "The Slow-Paced Encephalizers",
            "High brain-to-body ratios demand long maternal lactation and extended lifespans, shifting life history to the ultra-slow end of the mammalian pace-of-life continuum.",
        ),
        "Diprotodontia": (
            "Hypometabolic Marsupial Resilience",
            "Evolving in arid Australasian climates, marsupials maintain basal metabolic rates ~30% below placentals, enabling survival through multi-year resource droughts.",
        ),
        "Lagomorpha": (
            "Double-Pass Digestion & Springing Kinetic",
            "Caecotrophy allows rabbits and pikas to extract double the nutrition from sparse tundra vegetation, supporting explosive reproductive fecundity.",
        ),
        "Soricomorpha": (
            "The 1,200 BPM Knife-Edge",
            "With extreme surface-area-to-volume ratios, shrews lose heat so rapidly they must consume their own body weight daily or starve within hours (Kleiber's boundary).",
        ),
        "Eulipotyphla": (
            "Ancient Subterranean Specialization",
            "Adapted to low-oxygen tunnels with specialized hemoglobin that binds oxygen even at elevated burrow temperatures.",
        ),
        "Perissodactyla": (
            "High-Volume Hindgut Grazers",
            "Odd-toed ungulates (rhinos, equids, tapirs) process massive fibrous forage through an enlarged caecum, thriving on bulk fiber where ruminants starve.",
        ),
        "Cingulata": (
            "The Osteoderm Shield",
            "The only living mammals possessing protective dermal bone armor, trading high metabolic flexibility for predator defense and fossorial burrowing.",
        ),
        "Pilosa": (
            "Hypometabolic Arboreal Canopy Dwellers",
            "Sloths maintain one of the lowest body temperatures and metabolic rates of any mammal, moving minimally to conserve solar energy.",
        ),
        "Monotremata": (
            "The Ancient Egg-Laying Mosaic",
            "Possessing electroreceptive bills and egg-laying physiology, monotremes retain ancestral amniote traits dating back over 170 million years.",
        ),
    }

    if order in order_superpowers:
        title, desc = order_superpowers[order]
    else:
        title = f"The {order} Bauplan"
        desc = (
            f"Evolving within {order}, this species exhibits specialized physiological and allometric adaptations "
            f"calibrated along 177 million years of mammalian TimeTree divergence."
        )

    return {
        "title": title,
        "description": desc,
        "scale_text": scale_txt,
    }


def plot_size_comparison_silhouette(mass_g: float, species_name: str) -> plt.Figure:
    """
    Renders an intuitive size comparison chart comparing the focal species against
    familiar references (House Mouse, Domestic Cat, Adult Human).
    """
    if isinstance(mass_g, str) and not isinstance(species_name, str):
        species_name, mass_g = mass_g, float(species_name)
    else:
        try:
            mass_g = float(mass_g)
        except (ValueError, TypeError):
            mass_g = 100.0
        species_name = str(species_name)
    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfc")

    benchmarks = [
        ("House Mouse", 20.0, "🐁", "#718096"),
        ("Domestic Cat", 4_000.0, "🐈", "#4a5568"),
        ("Adult Human", 70_000.0, "🧍", "#2d3748"),
        (species_name, mass_g, "★", "#c05621"),
    ]

    labels = [b[0] for b in benchmarks]
    masses = [b[1] for b in benchmarks]
    colors = [b[3] for b in benchmarks]

    log_masses = [np.log10(max(m, 1.0)) for m in masses]
    y_pos = np.arange(len(labels))

    bars = ax.barh(y_pos, log_masses, color=colors, edgecolor="#1a202c", height=0.48, linewidth=0.8)

    for i, bar in enumerate(bars):
        w = bar.get_width()
        val = masses[i]
        formatted = format_trait_value("adult_body_mass_g", val)
        # Ratio comparison relative to human (70 kg)
        if i == 3:
            ratio = val / 70_000.0
            if ratio >= 1.0:
                rel_str = f" ({ratio:.1f}× Human)"
            elif ratio >= 0.05:
                rel_str = f" ({val/4000.0:.1f}× Cat)"
            else:
                rel_str = f" ({val/20.0:.0f}× Mouse)"
        elif i == 2:
            rel_str = " [Reference: 70 kg]"
        else:
            rel_str = ""

        ax.annotate(
            f" {formatted}{rel_str}",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#1a202c",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.0, fontweight="bold", color="#1a202c")
    ax.invert_yaxis()
    ax.set_xlabel("log₁₀(Body Mass [g])", fontsize=9.0, fontweight="bold", color="#2c3e50")
    ax.set_title("Visual Scale Comparison (vs. Familiar Anchors)", fontsize=10.0, fontweight="bold", color="#0d3b2e")
    ax.grid(axis="x", linestyle="--", alpha=0.45, color="#cbd5e0")
    ax.set_xlim(0, max(log_masses) * 1.35 + 0.5)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout(pad=1.0)
    return fig


def get_shrinkage_analogy(species_name: str, base_val: float, delta_val: float, trait_key: str) -> str:
    """Generates visceral physical shrinkage analogies for non-experts."""
    if trait_key == "adult_body_mass_g":
        mass_loss_g = abs(delta_val)
        if delta_val < 0:
            if mass_loss_g >= 1_000_000:
                t_loss = mass_loss_g / 1_000_000.0
                humans = mass_loss_g / 70_000.0
                return f"Loses {t_loss:.2f} tonnes of mass (~the weight of {humans:.0f} adult humans) to increase cooling surface area."
            elif mass_loss_g >= 50_000:
                kg_loss = mass_loss_g / 1_000.0
                humans = mass_loss_g / 70_000.0
                return f"Sheds {kg_loss:.1f} kg (~{humans:.1f} humans) to dissipate metabolic heat without hyperthermia."
            elif mass_loss_g >= 1_000:
                kg_loss = mass_loss_g / 1_000.0
                cats = mass_loss_g / 4_000.0
                return f"Loses {kg_loss:.2f} kg (~{cats:.1f} domestic cats of tissue) due to increased thermal dissipation constraints."
            else:
                mice = max(mass_loss_g / 20.0, 0.1)
                return f"Shrinks by {mass_loss_g:.1f} g (~{mice:.1f} mice worth of tissue) under thermal stress."
        else:
            return f"Expands mass by {format_trait_value(trait_key, abs(delta_val))} (unconstrained ecological split)."

    elif trait_key == "gestation_length_d":
        if delta_val < 0:
            return f"Embryogenesis accelerates by {abs(delta_val):.1f} days due to temperature-driven biochemical kinetics."
        return f"Developmental gestation extends by {abs(delta_val):.1f} days."

    elif trait_key == "max_longevity_m":
        years_lost = abs(delta_val) / 12.0
        if delta_val < 0:
            return f"Lifespan shortens by {years_lost:.1f} years ({abs(delta_val):.0f} months) via the metabolic rate-of-living effect."
        return f"Lifespan extends by {years_lost:.1f} years."

    elif trait_key == "litter_size":
        if delta_val > 0:
            return f"Reproductive clutch increases by +{delta_val:.2f} offspring, shifting toward an r-selection survival strategy."
        return f"Litter size contracts by {abs(delta_val):.2f} offspring."

    elif trait_key == "weaning_age_d":
        return f"Juvenile weaning accelerates by {abs(delta_val):.1f} days as maternal lactation period shortens."

    elif trait_key == "home_range_km2":
        if delta_val < 0:
            return f"Territory contracts by {abs(delta_val):.2f} km² due to higher tropical plant biomass density."
        return f"Foraging range expands by +{delta_val:.2f} km² to meet elevated metabolic energy budgets."

    return ""


def plot_mass_scale_gauge(mass_g: float, species_name: str) -> plt.Figure:
    """
    Renders an overlap-free logarithmic scale comparing the species mass
    against 6 evenly spaced benchmark taxa from shrew to whale.
    """
    log_mass = float(np.log10(max(mass_g, 1.0)))
    fig, ax = plt.subplots(figsize=(10, 2.0), dpi=100)
    fig.patch.set_facecolor("#fbfcfd")
    ax.set_facecolor("#fbfcfd")

    # Main axis line
    ax.plot([-0.2, 8.5], [0, 0], color="#95a5a6", linewidth=2.5, zorder=1)

    # Plot 6 benchmarks with alternating label offsets to completely eliminate overlap
    for i, (ref_name, ref_mass, _) in enumerate(SPACED_BENCHMARKS):
        ref_log = float(np.log10(ref_mass))
        ax.scatter(ref_log, 0, color="#576574", s=32, zorder=2)
        
        y_offset = -20 if i % 2 == 0 else -34
        ax.annotate(
            ref_name,
            xy=(ref_log, 0),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.0,
            fontweight="bold",
            color="#2c3e50",
        )

    # Highlight focal species with clear top annotation
    ax.scatter(log_mass, 0, color="#e74c3c", s=180, edgecolors="#ffffff", linewidth=2.5, zorder=4)
    ax.annotate(
        f"{species_name}\n({format_trait_value('adult_body_mass_g', mass_g)})",
        xy=(log_mass, 0),
        xytext=(0, 26),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color="#b71c1c",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f5", edgecolor="#e53935", linewidth=1.2),
        arrowprops=dict(arrowstyle="->", color="#e53935", lw=1.5),
    )

    ax.set_xlim(-0.5, 8.8)
    ax.set_ylim(-0.9, 0.9)
    ax.set_yticks([])
    ax.set_xticks(range(9))
    ax.set_xticklabels(["1g", "10g", "100g", "1kg", "10kg", "100kg", "1t", "10t", "100t"], fontsize=8.5, color="#555555")
    ax.set_xlabel("Logarithmic Body Mass Spectrum", fontsize=9.0, labelpad=2, fontweight="bold", color="#333333")

    for spine in ["top", "left", "right", "bottom"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout(pad=0.8)
    return fig


def plot_multi_trait_radar(df_pheno: pd.DataFrame, sp_id: str) -> plt.Figure:
    """
    Renders an allometric radar chart evaluating the species across
    6 core life-history traits with padded labels to prevent clipping.
    """
    sp_row = df_pheno[df_pheno["canonical_taxon_id"] == sp_id].iloc[0]
    order_name = sp_row["order"]
    order_df = df_pheno[df_pheno["order"] == order_name]

    traits = [
        ("adult_body_mass_g", "Body Mass"),
        ("gestation_length_d", "Gestation"),
        ("weaning_age_d", "Weaning Age"),
        ("max_longevity_m", "Longevity"),
        ("home_range_km2", "Home Range"),
        ("litter_size", "Litter Size"),
    ]

    categories = [t[1] for t in traits]
    N_vars = len(categories)

    sp_percentiles = []
    order_percentiles = []

    for col, _ in traits:
        valid_all = df_pheno[col].dropna()
        sp_val = sp_row[col]

        if pd.notna(sp_val) and len(valid_all) > 0:
            sp_pct = float(np.mean(valid_all <= sp_val) * 100.0)
        else:
            sp_pct = 50.0

        valid_order = order_df[col].dropna()
        if len(valid_order) > 0:
            order_pct = float(np.mean(valid_all <= valid_order.median()) * 100.0)
        else:
            order_pct = 50.0

        sp_percentiles.append(sp_pct)
        order_percentiles.append(order_pct)

    angles = np.linspace(0, 2 * np.pi, N_vars, endpoint=False).tolist()
    sp_percentiles += sp_percentiles[:1]
    order_percentiles += order_percentiles[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 4.6), subplot_kw=dict(polar=True), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfc")

    # Order polygon
    ax.plot(angles, order_percentiles, color="#718096", linewidth=1.5, linestyle="--", label=f"Order {order_name} Median")
    ax.fill(angles, order_percentiles, color="#718096", alpha=0.15)

    # Selected species polygon
    ax.plot(angles, sp_percentiles, color="#0d3b2e", linewidth=2.5, marker="o", markersize=5, label=f"{sp_row['species']}")
    ax.fill(angles, sp_percentiles, color="#134e4a", alpha=0.35)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9.0, fontweight="bold", color="#1c2d42")
    ax.tick_params(pad=12)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7.5, color="#888888")
    ax.set_ylim(0, 100)
    ax.grid(color="#dcdde1", linestyle="--", linewidth=0.8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8.5)

    plt.tight_layout(pad=1.5)
    return fig


def plot_allometric_scaling_curve(
    df_pheno: pd.DataFrame,
    sp_id: str,
    y_trait: str = "gestation_length_d",
    y_label: str = "Gestation Length (Days)",
) -> plt.Figure:
    """Renders allometric scaling scatter curve with non-overlapping legend."""
    sp_row = df_pheno[df_pheno["canonical_taxon_id"] == sp_id].iloc[0]
    
    sub_df = df_pheno[["canonical_taxon_id", "species", "order", "adult_body_mass_g", y_trait]].dropna()
    sub_df = sub_df[(sub_df["adult_body_mass_g"] > 0) & (sub_df[y_trait] > 0)]

    x_log = np.log10(sub_df["adult_body_mass_g"].values)
    y_log = np.log10(sub_df[y_trait].values)

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfc")

    ax.scatter(x_log, y_log, color="#bdc3c7", alpha=0.40, s=14, label="PanTHERIA Mammalia", zorder=1)

    order_sub = sub_df[sub_df["order"] == sp_row["order"]]
    if len(order_sub) > 0:
        ax.scatter(
            np.log10(order_sub["adult_body_mass_g"].values),
            np.log10(order_sub[y_trait].values),
            color="#2b6cb0",
            alpha=0.65,
            s=28,
            label=f"Order {sp_row['order']}",
            zorder=2,
        )

    sp_mass = sp_row["adult_body_mass_g"]
    sp_y = sp_row[y_trait]
    if pd.notna(sp_mass) and pd.notna(sp_y) and sp_mass > 0 and sp_y > 0:
        ax.scatter(
            np.log10(sp_mass),
            np.log10(sp_y),
            color="#c05621",
            s=140,
            edgecolors="#1a202c",
            linewidth=1.8,
            label=f"★ {sp_row['species']}",
            zorder=4,
        )

    if len(x_log) > 1:
        slope, intercept = np.polyfit(x_log, y_log, 1)
        x_grid = np.linspace(x_log.min(), x_log.max(), 100)
        ax.plot(x_grid, slope * x_grid + intercept, color="#0d3b2e", linestyle="--", linewidth=1.6, label=f"Slope = {slope:.2f}")

    ax.set_xlabel("log₁₀(Adult Body Mass [g])", fontsize=9.5, fontweight="bold", color="#2c3e50")
    ax.set_ylabel(f"log₁₀({y_label})", fontsize=9.5, fontweight="bold", color="#2c3e50")
    ax.set_title(f"Allometric Scaling: Mass vs. {y_label}", fontsize=10.5, fontweight="bold", color="#1a202c")
    ax.grid(True, linestyle="--", alpha=0.45, color="#cbd5e0")
    ax.legend(fontsize=8.0, loc="lower right", framealpha=0.92)

    plt.tight_layout(pad=1.0)
    return fig


# =============================================================================
# TAB 2 GRAPHICS: CLADE BUFFER DIAGRAM & DIVERGENCE DECAY
# =============================================================================

def plot_clade_buffer_diagram() -> plt.Figure:
    """
    Renders an infographic diagram illustrating the 140.0 Ma Patristic Buffer
    Quarantine Protocol and why naive cross-validation leaks sister species.
    """
    fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfc")

    ax.axvspan(37, 140, color="#fff3cd", alpha=0.55, label="140.0 Ma Buffer Quarantine (Exclusion Zone)")

    lines = [
        ([0, 20], [50, 50], "#2c3e50", 2.5),
        ([20, 50], [50, 75], "#27ae60", 2.2),
        ([50, 177], [75, 85], "#27ae60", 2.0),
        ([50, 177], [75, 65], "#27ae60", 2.0),
        ([20, 37], [50, 25], "#2c3e50", 2.2),
        ([37, 100], [25, 32], "#e67e22", 2.0),
        ([100, 177], [32, 38], "#d35400", 1.8),
        ([37, 177], [25, 12], "#c0392b", 2.5),
    ]

    for xs, ys, color, lw in lines:
        ax.plot(xs, ys, color=color, linewidth=lw)

    ax.text(180, 85, "Training Clade A (e.g., Carnivora)", fontsize=8.5, color="#1e7e34", fontweight="bold", va="center")
    ax.text(180, 65, "Training Clade B (e.g., Rodentia)", fontsize=8.5, color="#1e7e34", fontweight="bold", va="center")
    ax.text(180, 38, "Quarantined Sister Lineages (EXCLUDED)", fontsize=8.5, color="#d35400", fontweight="bold", va="center")
    ax.text(180, 12, "Test Clade (Withheld Order, N ≥ 50)", fontsize=8.5, color="#c0392b", fontweight="bold", va="center")

    ax.set_xlim(-5, 235)
    ax.set_ylim(0, 100)
    ax.set_yticks([])
    ax.set_xticks([0, 37, 140, 177])
    ax.set_xticklabels(["Root\n(177 Ma)", "d_buffer cutoff\n(140 Ma)", "Recent Splits\n(37 Ma)", "Present Day\n(0 Ma)"], fontsize=8.0)
    ax.set_xlabel("Evolutionary Divergence Time (Million Years Ago [Ma])", fontsize=9.0, fontweight="bold", color="#2c3e50")
    ax.set_title("The Patristic Quarantine Buffer (Sync Barrier 3)", fontsize=10.5, fontweight="bold", color="#1a202c")

    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#bdc3c7")
    ax.legend(loc="upper left", fontsize=8.0, framealpha=0.92)

    plt.tight_layout(pad=1.0)
    return fig


def plot_divergence_decay_curve(data_dir: Path) -> plt.Figure:
    """Plots accuracy degradation (RMSE) as divergence distance increases."""
    json_path = data_dir / "divergence_accuracy_tracking.json"
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfc")

    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)

        colors_map = {"RandomForest": "#27ae60", "XGBoost": "#2980b9", "Ridge": "#e67e22", "PhyloGNN": "#8e44ad"}
        for model_name, info in data.items():
            if "mean_divergences_ma" in info and "rmses" in info:
                divs = info["mean_divergences_ma"]
                rmses = info["rmses"]
                color = colors_map.get(model_name, "#555555")
                ax.scatter(divs, rmses, color=color, s=35, alpha=0.75)
                order_idx = np.argsort(divs)
                ax.plot(np.array(divs)[order_idx], np.array(rmses)[order_idx], color=color, linewidth=1.8, label=model_name)

    ax.set_xlabel("Mean Patristic Divergence to Training Set (Ma)", fontsize=9.0, fontweight="bold", color="#2c3e50")
    ax.set_ylabel("Out-of-Fold Extrapolation RMSE", fontsize=9.0, fontweight="bold", color="#2c3e50")
    ax.set_title("Predictive Accuracy Decay Across Deep Time", fontsize=10.5, fontweight="bold", color="#1a202c")
    ax.grid(True, linestyle="--", alpha=0.45, color="#cbd5e0")
    ax.legend(fontsize=8.0, loc="upper left")

    plt.tight_layout(pad=1.0)
    return fig


# =============================================================================
# TAB 3 GRAPHICS: THERMODYNAMICS & MULTI-TRAIT ANIMAL FORECASTING
# =============================================================================

def plot_thermodynamic_allometry_diagram() -> plt.Figure:
    """
    Infographic diagram illustrating the Fourier/Kleiber heat-balance physics:
    Heat Production (M^0.75) vs Heat Dissipation (M^0.67 * delta_T).
    """
    fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfc")

    mass_grid = np.linspace(10, 100_000, 200)
    log_m = np.log10(mass_grid)

    heat_prod = 0.75 * log_m
    heat_loss_baseline = 0.67 * log_m + 0.25
    heat_loss_warmed = 0.67 * log_m + 0.10

    ax.plot(log_m, heat_prod, color="#c0392b", linewidth=2.2, label="Metabolic Heat Production: Q_in ∝ M⁰·⁷⁵ (Kleiber's Law)")
    ax.plot(log_m, heat_loss_baseline, color="#2980b9", linewidth=2.0, linestyle="--", label="Heat Dissipation at 15°C: Q_out ∝ M⁰·⁶⁷·(T_body - T_amb)")
    ax.plot(log_m, heat_loss_warmed, color="#e67e22", linewidth=2.0, linestyle="-.", label="Heat Dissipation at 20°C (Warming shrinks thermal gradient)")

    ax.annotate(
        "Thermodynamic Cooling Limit\n(Forces Body Mass Reduction: ΔM < 0)",
        xy=(3.5, 2.72),
        xytext=(1.8, 3.1),
        arrowprops=dict(facecolor="#c0392b", shrink=0.08, width=1.5, headwidth=6),
        fontsize=8.5,
        fontweight="bold",
        color="#b71c1c",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffebee", edgecolor="#c0392b", alpha=0.9),
    )

    ax.set_xlabel("log₁₀(Body Mass [g])", fontsize=9.0, fontweight="bold", color="#2c3e50")
    ax.set_ylabel("log₁₀(Heat Rate)", fontsize=9.0, fontweight="bold", color="#2c3e50")
    ax.set_title("Biophysical Thermodynamics: Why Warming Mandates Smaller Size (Bergmann's Rule)", fontsize=10.0, fontweight="bold", color="#1a202c")
    ax.grid(True, linestyle="--", alpha=0.45, color="#cbd5e0")
    ax.legend(fontsize=7.8, loc="lower right")

    plt.tight_layout(pad=1.0)
    return fig


def plot_animal_forecast_real_units(
    species_name: str,
    trait_col: str,
    trait_label: str,
    base_val: float,
    pred_ou: float,
    pred_rf_aug: Optional[float] = None,
    pred_xgb_aug: Optional[float] = None,
    pred_rf_base: Optional[float] = None,
    pred_xgb_base: Optional[float] = None,
    pred_piml: Optional[float] = None,
    show_unanchored: bool = False,
    **kwargs,
) -> plt.Figure:
    """
    Renders an overlap-free horizontal comparison bar chart displaying
    the actual real-world values for a specific animal across real ML models (RF & XGBoost).
    """
    if pred_rf_aug is None:
        pred_rf_aug = kwargs.get("pred_aug", base_val)
    if pred_xgb_aug is None:
        pred_xgb_aug = kwargs.get("pred_aug", base_val)
    if pred_rf_base is None:
        pred_rf_base = kwargs.get("pred_naive", None)
    if pred_xgb_base is None:
        pred_xgb_base = kwargs.get("pred_naive", None)

    fig, ax = plt.subplots(figsize=(8.2, 4.5), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfc")

    formatted_base = format_trait_value(trait_col, base_val)

    labels = [
        f"Baseline (Current: {formatted_base})",
        "Physics Simulation (Thermodynamic OU)",
    ]
    vals = [base_val, pred_ou]
    colors = ["#718096", "#0d3b2e"]

    if pred_piml is not None:
        labels.append("Physics-Informed ML (PI-ML)")
        vals.append(pred_piml)
        colors.append("#047857")

    labels.append("Random Forest (Phylo-Lock Active)")
    vals.append(pred_rf_aug)
    colors.append("#155e42")

    labels.append("XGBoost (Phylo-Lock Active)")
    vals.append(pred_xgb_aug)
    colors.append("#2b6cb0")

    if show_unanchored and pred_rf_base is not None and pred_xgb_base is not None:
        labels.append("Naive Random Forest (No Phylo-Lock)")
        vals.append(pred_rf_base)
        colors.append("#c05621")

        labels.append("Naive XGBoost (No Phylo-Lock)")
        vals.append(pred_xgb_base)
        colors.append("#d97706")

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, vals, color=colors, edgecolor="#1a202c", height=0.55, linewidth=0.8)

    # Reference line at current baseline value
    ax.axvline(base_val, color="#718096", linestyle="--", linewidth=1.2, alpha=0.7)

    max_val = max(vals) if len(vals) > 0 and max(vals) > 0 else 1.0

    for i, bar in enumerate(bars):
        w = bar.get_width()
        pct_change = ((w - base_val) / base_val) * 100.0 if base_val > 0 else 0.0
        pct_str = f" ({pct_change:+.1f}%)" if i > 0 else " [Current]"
        formatted_w = format_trait_value(trait_col, w)
        ann_text = f" {formatted_w}{pct_str}"

        ax.annotate(
            ann_text,
            xy=(w, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#1a202c",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5, fontweight="bold", color="#1a202c")
    ax.invert_yaxis()
    ax.set_xlabel(f"Projected Trait Value ({trait_label})", fontsize=9.0, fontweight="bold", color="#2c3e50")
    ax.set_title(f"Live Machine Learning Forecast for {species_name} — {trait_label}", fontsize=10.5, fontweight="bold", color="#0d3b2e")
    ax.grid(axis="x", linestyle="--", alpha=0.45, color="#cbd5e0")

    # Extra margin on right so labels don't clip
    ax.set_xlim(0, max_val * 1.38)

    plt.tight_layout(pad=1.0)
    return fig


def plot_forecasting_comparison_horizontal(
    delta_t: float,
    shift_ou: float,
    shift_rf_aug: Optional[float] = None,
    shift_xgb_aug: Optional[float] = None,
    shift_rf_base: Optional[float] = None,
    shift_xgb_base: Optional[float] = None,
    shift_piml: Optional[float] = None,
    trait_label: str = "Adult Body Mass",
    show_unanchored: bool = False,
    **kwargs,
) -> plt.Figure:
    """
    Renders an overlap-free horizontal bar chart comparing forward trait shifts (log scale)
    across real machine learning models (Random Forest & XGBoost).
    """
    if shift_rf_aug is None:
        shift_rf_aug = kwargs.get("shift_aug_ml", 0.0)
    if shift_xgb_aug is None:
        shift_xgb_aug = kwargs.get("shift_aug_ml", 0.0)
    if shift_rf_base is None:
        shift_rf_base = kwargs.get("shift_naive_ml", None)
    if shift_xgb_base is None:
        shift_xgb_base = kwargs.get("shift_naive_ml", None)

    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfc")

    labels = [
        "Physics Simulation (Thermodynamic OU)",
    ]
    vals = [shift_ou]
    colors = ["#0d3b2e"]

    if shift_piml is not None:
        labels.append("Physics-Informed ML (PI-ML)")
        vals.append(shift_piml)
        colors.append("#047857")

    labels.append("Random Forest (Phylo-Lock Active)")
    vals.append(shift_rf_aug)
    colors.append("#155e42")

    labels.append("XGBoost (Phylo-Lock Active)")
    vals.append(shift_xgb_aug)
    colors.append("#2b6cb0")

    if show_unanchored and shift_rf_base is not None and shift_xgb_base is not None:
        labels.append("Naive Random Forest (No Phylo-Lock)")
        vals.append(shift_rf_base)
        colors.append("#c05621")

        labels.append("Naive XGBoost (No Phylo-Lock)")
        vals.append(shift_xgb_base)
        colors.append("#d97706")

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, vals, color=colors, edgecolor="#1a202c", height=0.55, linewidth=0.8)
    ax.axvline(0, color="#111111", linestyle="--", linewidth=0.8)

    for bar in bars:
        width = bar.get_width()
        x_text = width + (0.003 if width >= 0 else -0.003)
        ha = "left" if width >= 0 else "right"
        ax.annotate(
            f"{width:+.4f}",
            xy=(x_text, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha=ha,
            fontsize=8.5,
            fontweight="bold",
            color="#1a202c",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5, fontweight="bold", color="#1a202c")
    ax.invert_yaxis()
    ax.set_xlabel(f"Predicted Phenotypic Trait Shift (Δ log₁₀ {trait_label})", fontsize=9.0, fontweight="bold", color="#2c3e50")
    ax.set_title(f"Live ML Model Shifts under ΔT = +{delta_t:.1f}°C ({trait_label})", fontsize=10.5, fontweight="bold", color="#0d3b2e")
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="#cbd5e0")

    min_val, max_val = min(vals), max(vals)
    span = max(abs(min_val), abs(max_val), 0.05)
    ax.set_xlim(-span * 1.4, span * 1.4)

    plt.tight_layout(pad=1.0)
    return fig
