import os
import subprocess
from pathlib import Path
import fitz

def build_perfect_report():
    assets_dir = Path("d:/Projects/Phenix/report_assets").resolve()
    nitk_logo_uri = (assets_dir / "nitk_logo.png").as_uri()
    fig3_1_uri = (assets_dir / "fig_feature_pipeline.jpg").as_uri()
    fig3_2_uri = (assets_dir / "fig_validation_flow.jpg").as_uri()
    fig4_1_uri = (assets_dir / "fig_results_chart.png").as_uri()

    # MathML equations
    eq1_mathml = '''<math display="block" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>B</mi><mo>=</mo><mo>−</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mi>H</mi><mo stretchy="false">(</mo><msup><mi>D</mi><mrow><mo>∘</mo><mn>2</mn></mrow></msup><mo stretchy="false">)</mo><mi>H</mi><mo>,</mo><mspace width="1.5em"></mspace><mi>H</mi><mo>=</mo><mi>I</mi><mo>−</mo><mfrac><mn>1</mn><mi>N</mi></mfrac><mn mathvariant="bold">1</mn><msup><mn mathvariant="bold">1</mn><mi>T</mi></msup></mrow></semantics></math>'''
    eq2_mathml = '''<math display="block" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi mathvariant="normal">Σ</mi><mo>=</mo><mi>L</mi><msup><mi>L</mi><mi>T</mi></msup><mo>,</mo><mspace width="1.5em"></mspace><msub><mi>X</mi><mtext>whitened</mtext></msub><mo>=</mo><mo stretchy="false">(</mo><mi>X</mi><mo>−</mo><mi>μ</mi><mo stretchy="false">)</mo><msup><mi>L</mi><mrow><mo>−</mo><mi>T</mi></mrow></msup></mrow></semantics></math>'''
    eq3_mathml = '''<math display="block" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msup><mi>R</mi><mn>2</mn></msup><mo>=</mo><mn>1</mn><mo>−</mo><mfrac><mrow><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><mo stretchy="false">(</mo><msub><mi>y</mi><mi>i</mi></msub><mo>−</mo><msub><mover><mi>y</mi><mo accent="true">̂</mo></mover><mi>i</mi></msub><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow><mrow><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><mo stretchy="false">(</mo><msub><mi>y</mi><mi>i</mi></msub><mo>−</mo><mover><mi>y</mi><mo accent="true">‾</mo></mover><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow></mfrac><mo>,</mo><mspace width="1.5em"></mspace><mtext>MAE</mtext><mo>=</mo><mfrac><mn>1</mn><mi>N</mi></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><mo stretchy="false">|</mo><msub><mi>y</mi><mi>i</mi></msub><mo>−</mo><msub><mover><mi>y</mi><mo accent="true">̂</mo></mover><mi>i</mi></msub><mo stretchy="false">|</mo></mrow></semantics></math>'''
    eq4_mathml = '''<math display="block" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi mathvariant="normal">Δ</mi><msup><mi>R</mi><mn>2</mn></msup><mo>=</mo><msubsup><mi>R</mi><mtext>Random&nbsp;CV</mtext><mn>2</mn></msubsup><mo>−</mo><msubsup><mi>R</mi><mtext>Phylo-CV</mtext><mn>2</mn></msubsup></mrow></semantics></math>'''

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PHENIX Evaluation Report</title>
<style>
@page {{
    size: 210mm 297mm;
    margin: 0;
}}
* {{
    box-sizing: border-box;
    -webkit-print-color-adjust: exact;
}}
body {{
    margin: 0;
    padding: 0;
    background-color: #ffffff;
    font-family: 'Times New Roman', Times, serif;
    font-size: 12pt;
    line-height: 1.45;
    color: #000000;
}}
.page {{
    width: 210mm;
    height: 297mm;
    box-sizing: border-box;
    padding-left: 30.5mm;
    padding-right: 25.4mm;
    position: relative;
    page-break-after: always;
    break-after: page;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    overflow: hidden;
}}
.chapter-page {{
    padding-top: 48mm;
}}
.cont-page {{
    padding-top: 25.4mm;
}}
.page-footer {{
    position: absolute;
    bottom: 18mm;
    left: 0;
    width: 100%;
    text-align: center;
    font-size: 12pt;
    font-family: 'Times New Roman', Times, serif;
}}
p {{
    margin-top: 0;
    margin-bottom: 3.5mm;
    text-align: justify;
    text-justify: inter-word;
}}
.chapter-num {{
    font-size: 24.8pt;
    font-weight: bold;
    line-height: 1.2;
    margin-bottom: 4mm;
}}
.chapter-title {{
    font-size: 24.8pt;
    font-weight: bold;
    line-height: 1.2;
    margin-bottom: 7mm;
}}
.section-title {{
    font-size: 17.2pt;
    font-weight: bold;
    line-height: 1.25;
    margin-top: 4mm;
    margin-bottom: 3.5mm;
}}
.subsection-title {{
    font-size: 13.5pt;
    font-weight: bold;
    line-height: 1.3;
    margin-top: 3.5mm;
    margin-bottom: 2.5mm;
}}
.fig-container {{
    text-align: center;
    margin: 2mm 0 1.5mm 0;
}}
.fig-container img {{
    max-width: 100%;
    height: auto;
}}
.caption {{
    font-size: 10.9pt;
    text-align: center;
    margin-top: 1.5mm;
    margin-bottom: 3mm;
    line-height: 1.35;
}}
.eq-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 2.5mm 0;
    padding: 0 4mm;
}}
.eq-math {{
    flex: 1;
    text-align: center;
}}
.eq-num {{
    font-size: 12pt;
    font-family: 'Times New Roman', Times, serif;
    font-style: normal;
    padding-left: 5mm;
}}
table.booktabs {{
    width: 100%;
    border-collapse: collapse;
    margin: 2.5mm 0 3.5mm 0;
    font-size: 9.2pt;
    line-height: 1.28;
}}
table.booktabs th, table.booktabs td {{
    padding: 5px 6px;
    text-align: left;
    vertical-align: top;
}}
table.booktabs th {{
    border-top: 1.5pt solid #000;
    border-bottom: 0.75pt solid #000;
    font-weight: bold;
}}
table.booktabs tr:last-child td {{
    border-bottom: 1.5pt solid #000;
}}
.toc-item {{
    display: flex;
    align-items: baseline;
    margin-bottom: 2.8mm;
    font-size: 12pt;
}}
.toc-dots {{
    flex: 1;
    border-bottom: 1px dotted #222;
    margin: 0 4px;
    height: 0;
    transform: translateY(-4px);
}}
.toc-sub {{
    padding-left: 18px;
}}
.ref-item {{
    text-indent: -24px;
    padding-left: 24px;
    margin-bottom: 3mm;
    text-align: justify;
    text-justify: inter-word;
    font-size: 11pt;
    line-height: 1.38;
}}
ul.bullet-list {{
    margin: 0 0 3mm 0;
    padding-left: 18px;
    list-style-type: disc;
}}
ul.bullet-list li {{
    margin-bottom: 2.2mm;
    text-align: justify;
    text-justify: inter-word;
}}
ol.num-list {{
    margin: 0 0 3mm 0;
    padding-left: 20px;
}}
ol.num-list li {{
    margin-bottom: 2.5mm;
    text-align: justify;
    text-justify: inter-word;
}}
math {{
    font-family: 'Cambria Math', 'Times New Roman', serif;
}}
</style>
</head>
<body>

<!-- ================= PAGE 1: COVER PAGE ================= -->
<div class="page" style="padding-top: 40mm; align-items: center; text-align: center;">
    <div style="font-size: 12pt; font-weight: bold; letter-spacing: 1.5px; margin-bottom: 22mm;">
        INFORMATION TECHNOLOGY PROJECT
    </div>
    
    <div style="font-size: 21pt; font-weight: bold; line-height: 1.35; max-width: 90%; margin-bottom: 5mm;">
        PHENIX: Phylogenetically Informed Machine Learning in Phenotypic Trait Prediction
    </div>
    
    <div style="font-size: 13.5pt; font-weight: normal; line-height: 1.4; max-width: 86%; margin-bottom: 16mm;">
        Predicting Traits of Mammal Species from Climate, Genes, and Evolutionary History, Without Fooling Ourselves About Accuracy
    </div>
    
    <div style="font-size: 12pt; margin-bottom: 3.5mm;">
        Submitted to Professor Deekshitha
    </div>
    
    <div style="font-size: 12pt; margin-bottom: 3.5mm;">
        by
    </div>
    
    <div style="font-size: 12pt; font-weight: bold; margin-bottom: 15mm;">
        Asrith Singampalli (231EC110), Samruddhi Deshpande (231EC117)
    </div>
    
    <div style="margin-bottom: 15mm;">
        <img src="{nitk_logo_uri}" alt="NITK Crest" style="width: 105px; height: auto;">
    </div>
    
    <div style="font-size: 11.5pt; line-height: 1.45;">
        Department of Information Technology<br>
        NATIONAL INSTITUTE OF TECHNOLOGY KARNATAKA (NITK)<br>
        SURATHKAL, MANGALORE - 575 025<br>
        October 2026
    </div>
</div>

<!-- ================= PAGE 2: TABLE OF CONTENTS ================= -->
<div class="page chapter-page">
    <div class="chapter-title" style="margin-bottom: 10mm;">Contents</div>
    
    <div class="toc-item" style="font-weight: bold;">
        <span>Abstract</span><span class="toc-dots"></span><span>ii</span>
    </div>
    
    <div style="height: 3mm;"></div>
    
    <div class="toc-item" style="font-weight: bold;">
        <span>1 &nbsp; Introduction</span><span class="toc-dots"></span><span>1</span>
    </div>
    
    <div style="height: 3mm;"></div>
    
    <div class="toc-item" style="font-weight: bold;">
        <span>2 &nbsp; Objectives</span><span class="toc-dots"></span><span>2</span>
    </div>
    
    <div style="height: 3mm;"></div>
    
    <div class="toc-item" style="font-weight: bold;">
        <span>3 &nbsp; System Block Diagram</span><span class="toc-dots"></span><span>3</span>
    </div>
    <div class="toc-item toc-sub">
        <span>3.1 &nbsp; Multimodal Feature Extraction and Phylo-PCoA Pipeline</span><span class="toc-dots"></span><span>3</span>
    </div>
    <div class="toc-item toc-sub">
        <span>3.2 &nbsp; Leak-Free Validation and Benchmark Flow</span><span class="toc-dots"></span><span>5</span>
    </div>
    
    <div style="height: 3mm;"></div>
    
    <div class="toc-item" style="font-weight: bold;">
        <span>4 &nbsp; Results and Discussion</span><span class="toc-dots"></span><span>6</span>
    </div>
    <div class="toc-item toc-sub">
        <span>4.1 &nbsp; In-Domain Imputation versus Out-of-Clade Extrapolation</span><span class="toc-dots"></span><span>6</span>
    </div>
    <div class="toc-item toc-sub">
        <span>4.2 &nbsp; Biological Accuracy and Extrapolation Drift</span><span class="toc-dots"></span><span>7</span>
    </div>
    <div class="toc-item toc-sub">
        <span>4.3 &nbsp; Clade Withholding and Inflation Gap Analysis</span><span class="toc-dots"></span><span>7</span>
    </div>
    <div class="toc-item toc-sub">
        <span>4.4 &nbsp; Architectural Benchmark Verdicts and Discussion</span><span class="toc-dots"></span><span>8</span>
    </div>
    
    <div style="height: 3mm;"></div>
    
    <div class="toc-item" style="font-weight: bold;">
        <span>5 &nbsp; Conclusion and Future Work</span><span class="toc-dots"></span><span>9</span>
    </div>
    <div class="toc-item toc-sub">
        <span>5.1 &nbsp; Conclusion</span><span class="toc-dots"></span><span>9</span>
    </div>
    <div class="toc-item toc-sub">
        <span>5.2 &nbsp; Future Work</span><span class="toc-dots"></span><span>10</span>
    </div>
    
    <div style="height: 3mm;"></div>
    
    <div class="toc-item" style="font-weight: bold;">
        <span>References</span><span class="toc-dots"></span><span>11</span>
    </div>
    
    <div class="page-footer">i</div>
</div>

<!-- ================= PAGE 3: ABSTRACT ================= -->
<div class="page chapter-page">
    <div class="chapter-title" style="margin-bottom: 7mm;">Abstract</div>
    
    <p>
        Accurately predicting missing phenotypic traits across species is essential for biodiversity science, yet limited field surveys leave basic life-history measurements, such as adult body mass, absent for thousands of living mammals. Machine learning is increasingly deployed to fill these gaps from climate and genomic covariates. However, standard machine learning assumes independent and identically distributed observations. In organismal biology, species share common ancestors across a branching evolutionary tree—a fundamental challenge formulated as Felsenstein’s Dilemma (1985). Under naive random cross-validation, sister species enter both training and testing sets, causing clade leakage: algorithms memorize lineage proximity, artificially inflating <i>R</i><sup>2</sup> above 0.90, yet failing when tested on unseen lineages. An <i>R</i><sup>2</sup> below zero indicates predictions worse than guessing the sample mean.
    </p>
    <p>
        This project, PHENIX, introduces a phylogenetically informed framework to evaluate phenotypic trait prediction across 3,268 mammal species. We extracted 32 orthogonal tree coordinates via Gower double-centering (Phylo-PCoA) on the 3,268 &times; 3,268 cophenetic distance matrix (Open Tree of Life and TimeTree, root age 177.0 Ma), capturing 89.98% of phylogenetic variance, and combined them with Cholesky-whitened climate (WorldClim) and genomic features (ESM-2 embeddings, SNP, metabolic scores). We benchmarked 16 setups across four architectures under random 10-fold CV and phylogenetic CV with a 140 Ma temporal buffer. Baseline features alone yield near-zero <i>R</i><sup>2</sup> (Ridge: 0.001, Random Forest: -0.025, XGBoost: -0.034). Adding Phylo-PCoA boosts within-clade <i>R</i><sup>2</sup> to 0.849 for Random Forest and 0.848 for XGBoost (&plusmn;2.16&times; error factor). However, under phylogenetic CV, all models produce negative <i>R</i><sup>2</sup>, with XGBoost proving the most stable extrapolator (RMSE 1.274). Variance partitioning indicates that 86.93% of predictable variation reflects phylogenetic inertia, while climate contributes under 0.1%. These findings demonstrate that random cross-validation measures relatives rather than biology, establishing the necessity of the Dual-Protocol Standard.
    </p>
    
    <div class="page-footer">ii</div>
</div>

<!-- ================= PAGE 4: CHAPTER 1 INTRODUCTION ================= -->
<div class="page chapter-page">
    <div class="chapter-num">Chapter 1</div>
    <div class="chapter-title">Introduction</div>
    
    <p>
        Accelerating environmental change has intensified the need for global species trait databases, yet limited field surveys leave fundamental measurements, such as adult body mass, absent for thousands of living mammals. Machine learning is increasingly deployed to impute these gaps from environmental and genomic covariates. However, standard machine learning assumes that observations are independent and identically distributed. In organismal biology, this assumption is fundamentally violated: species share common ancestors across a branching evolutionary tree. This challenge was formally articulated by Joseph Felsenstein in 1985 as Felsenstein’s Dilemma, recognizing that shared ancestry creates profound phylogenetic autocorrelation.
    </p>
    <p>
        When standard random cross-validation is applied, sister species and close relatives enter both the training and test folds. Rather than learning true functional biological mechanisms, models memorize lineage proximity—a pathology termed <em>clade leakage</em>. Memorizing relatives can inflate cross-validation <i>R</i><sup>2</sup> scores above 0.90, giving an illusion of high predictive accuracy, only to fail catastrophically when evaluated on unseen lineages. In statistical evaluation, an <i>R</i><sup>2</sup> below zero indicates that the model produces worse predictions than simply guessing the sample mean.
    </p>
    <p>
        Prior comparative methods suffer from severe computational and representational limits. Classical Phylogenetic Generalized Least Squares (PGLS) and independent contrasts require <i>O</i>(<i>N</i><sup>3</sup>) matrix inversions and linear assumptions (Felsenstein 1985; Martins and Hansen 1997). Phylogenetic eigenvector regression with Gower double-centering was historically applied only to small clades under 200 species (Diniz-Filho et al. 1998). While spatial block cross-validation addresses geospatial leakage (Roberts et al. 2017; Meyer et al. 2018), phylogenetic cross-validation remains rare in trait prediction. Furthermore, foundation protein embeddings (ESM-2) and geometric deep learning had not been benchmarked on out-of-clade extrapolation (Lin et al. 2023). This project quantifies how much evolutionary relatedness governs trait predictability, asking: <em>How well can machine learning predict traits of species from lineages it has never seen?</em>
    </p>
    
    <div class="page-footer">1</div>
</div>

<!-- ================= PAGE 5: CHAPTER 2 OBJECTIVES ================= -->
<div class="page chapter-page">
    <div class="chapter-num">Chapter 2</div>
    <div class="chapter-title">Objectives</div>
    
    <p>
        The aim of this project is to find out how evolutionary non-independence limits machine learning models in predicting phenotypic traits, assessing both in-domain interpolation and out-of-clade extrapolation across deep time. The work is divided into the following five objectives:
    </p>
    
    <ol class="num-list">
        <li>
            <strong>Build a multimodal dataset:</strong> Assemble a standardized dataset of 3,268 mammal species across 27 taxonomic orders, integrating adult body mass from PanTHERIA, 21 bioclimatic variables from WorldClim, and 64 genomic features (ESM-2 protein embeddings, SNP variants, and metabolic scores).
        </li>
        <li>
            <strong>Generate spectral tree embeddings (Phylo-PCoA):</strong> Convert the global mammalian evolutionary tree (calibrated from TimeTree and Open Tree of Life, root divergence 177.0 million years ago) into 32 numeric features via Gower double-centering on the 3,268 &times; 3,268 distance matrix, capturing 89.98% of tree variance.
        </li>
        <li>
            <strong>Withhold whole clades for leak-free validation:</strong> Establish a phylogenetic cross-validation scheme that withholds whole clades from training, augmented by a 140 million year (Ma) temporal buffer that removes training species closer than 140 Ma to the test clade.
        </li>
        <li>
            <strong>Benchmark 16 experimental setups:</strong> Systematically evaluate 16 configurations combining 4 model architectures (Ridge Regression, Random Forest, XGBoost Regressor, and PhyloGNN) &times; 2 feature representations (Baseline vs. Augmented) &times; 2 validation schemes (Random 10-fold CV vs. Phylogenetic CV).
        </li>
        <li>
            <strong>Inspect residual phylogenetic autocorrelation:</strong> Quantify remaining phylogenetic signal in prediction errors across all models using Blomberg’s <i>K</i> and Pagel’s <i>&lambda;</i> to test whether spectral coordinates eliminate lineage autocorrelation.
        </li>
    </ol>
    
    <p>
        The study is conducted in Python simulation, predicting log<sub>10</sub> adult body mass across 3,268 mammal species, explicitly separating in-domain phylogenetic imputation from out-of-clade evolutionary extrapolation.
    </p>
    
    <div class="page-footer">2</div>
</div>

<!-- ================= PAGE 6: CHAPTER 3 METHODOLOGY - PART 1 ================= -->
<div class="page chapter-page">
    <div class="chapter-num">Chapter 3</div>
    <div class="chapter-title">System Block Diagram</div>
    
    <p>
        This chapter describes the system modelled in this project. The architecture is presented as a multimodal feature extraction pipeline and a leak-free validation framework designed to rigorously benchmark biological trait prediction.
    </p>
    
    <div class="section-title">3.1 Multimodal Feature Extraction and Phylo-PCoA Pipeline</div>
    <p>
        Figure 3.1 shows the feature extraction pipeline. It integrates taxonomic, phylogenetic, bioclimatic, and genomic data across 3,268 mammalian species. The prediction target is log<sub>10</sub> adult body mass obtained from PanTHERIA.
    </p>
    
    <div class="fig-container">
        <img src="{fig3_1_uri}" alt="Multimodal Feature Extraction Pipeline" style="max-height: 44mm;">
        <div class="caption">Figure 3.1: Block diagram of the multimodal feature extraction and Phylo-PCoA pipeline.</div>
    </div>
    
    <p>
        The phylogenetic tree is constructed from the Open Tree of Life topology calibrated with divergence times from TimeTree, rooted at 177.0 million years ago (Ma). The resulting 3,268 &times; 3,268 cophenetic distance matrix is processed using Gower double-centering to extract 32 continuous principal coordinates (Phylo-PCoA), capturing 89.98% of total tree variance. In parallel, 21 bioclimatic variables from WorldClim and 64 genomic features are decorrelated via Cholesky whitening to yield an 85-dimensional baseline, which is concatenated with the 32 tree coordinates to produce 117 features.
    </p>
    <p>
        The individual blocks are described in detail on the following page.
    </p>
    
    <div class="page-footer">3</div>
</div>

<!-- ================= PAGE 7: CHAPTER 3 METHODOLOGY - PART 2 ================= -->
<div class="page cont-page">
    <ul class="bullet-list">
        <li>
            <strong>Phylogenetic Distance Matrix and Phylo-PCoA:</strong> Let <i>D</i> &isin; &reals;<sup><i>N</i>&times;<i>N</i></sup> represent the pairwise cophenetic distance matrix across <i>N</i> = 3,268 mammal species. Element-wise squared distances <i>D</i><sup>&compfn;2</sup> are transformed into a centered inner product matrix <i>B</i> using Gower double-centering:
        </li>
    </ul>
    
    <div class="eq-row">
        <div class="eq-math">{eq1_mathml}</div>
        <div class="eq-num">(3.1)</div>
    </div>
    
    <p>
        Spectral decomposition of <i>B</i> = <i>V</i>&Lambda;<i>V</i><sup><i>T</i></sup> yields the leading 32 eigenvectors scaled by their eigenvalues, &Phi; = <i>V</i><sub>32</sub>&Lambda;<sub>32</sub><sup>1/2</sup>. These 32 orthogonal coordinates capture 89.98% of the total evolutionary tree variance.
    </p>
    
    <ul class="bullet-list">
        <li>
            <strong>Baseline Features and Cholesky Whitening:</strong> The 85 baseline features comprise 21 WorldClim climate/elevation layers and 64 genomic attributes (ESM-2 protein embeddings, SNP variants, and metabolic scores). Collinearity is removed via Cholesky whitening:
        </li>
    </ul>
    
    <div class="eq-row">
        <div class="eq-math">{eq2_mathml}</div>
        <div class="eq-num">(3.2)</div>
    </div>
    
    <ul class="bullet-list">
        <li>
            <strong>Augmented Representation:</strong> Whitened baseline features and Phylo-PCoA coordinates are concatenated to form the 117-dimensional input vector <i>X</i><sub>aug</sub> = [<i>X</i><sub>whitened</sub> &parallel; &Phi;].
        </li>
        <li>
            <strong>Prediction Target and Metrics:</strong> The target is <i>y</i> = log<sub>10</sub>(body mass). Predictions &ycirc; are evaluated using the coefficient of determination (<i>R</i><sup>2</sup>) and Mean Absolute Error (MAE):
        </li>
    </ul>
    
    <div class="eq-row">
        <div class="eq-math">{eq3_mathml}</div>
        <div class="eq-num">(3.3)</div>
    </div>
    
    <ul class="bullet-list">
        <li>
            <strong>Inflation Gap (&Delta;<i>R</i><sup>2</sup>):</strong> Measures accuracy inflation caused by clade leakage in naive cross-validation:
        </li>
    </ul>
    
    <div class="eq-row">
        <div class="eq-math">{eq4_mathml}</div>
        <div class="eq-num">(3.4)</div>
    </div>
    
    <div class="page-footer">4</div>
</div>

<!-- ================= PAGE 8: CHAPTER 3 METHODOLOGY - PART 3 ================= -->
<div class="page cont-page">
    <div class="section-title" style="margin-top: 0; margin-bottom: 3.5mm;">3.2 Leak-Free Validation and Benchmark Flow</div>
    <p>
        The benchmark procedure is evaluated numerically in Python. The flow of the leak-free validation framework is shown in Figure 3.2.
    </p>
    
    <div class="fig-container">
        <img src="{fig3_2_uri}" alt="Validation and Benchmark Flow" style="max-height: 62mm;">
        <div class="caption">Figure 3.2: Flow diagram of the leak-free validation and benchmark framework.</div>
    </div>
    
    <p>
        The validation flow proceeds across two fundamental evaluation protocols:
    </p>
    <p>
        <strong>1. Random 10-Fold Cross-Validation:</strong> Standard naive cross-validation where 3,268 species are randomly assigned across 10 folds. Because sister species co-occur across folds, models are prone to clade leakage.
    </p>
    <p>
        <strong>2. Phylogenetic Cross-Validation (Phylo-CV):</strong> Entire monophyletic clades are held out as test sets (e.g., clades of 1,215 species, 678 species, 211 species, and 194 species). To eliminate leakage from proximate relatives, a 140 million year (Ma) temporal buffer removes all training species sharing a common ancestor with the test clade within 140 Ma.
    </p>
    <p>
        Four distinct model architectures are benchmarked: Ridge Regression (L2-regularized linear model), Random Forest (bagged decision trees), XGBoost Regressor (gradient boosted trees), and PhyloGNN (a 2-layer Graph Convolutional Network implemented in PyG over 3,442 tree nodes). Each model is trained on both baseline (85-d) and augmented (117-d) features, yielding 16 full experimental setups.
    </p>
    
    <div class="page-footer">5</div>
</div>

<!-- ================= PAGE 9: CHAPTER 4 RESULTS - PART 1 ================= -->
<div class="page chapter-page">
    <div class="chapter-num">Chapter 4</div>
    <div class="chapter-title">Results and Discussion</div>
    
    <p>
        This chapter presents empirical results across the 16 benchmark configurations evaluated on 3,268 mammal species, using the methodology established in Chapter 3.
    </p>
    
    <div class="section-title">4.1 In-Domain Imputation versus Out-of-Clade Extrapolation</div>
    <p>
        Figure 4.1 shows the performance comparison between in-domain Random CV <i>R</i><sup>2</sup> and out-of-clade Phylo-CV <i>R</i><sup>2</sup> under the 140 Ma buffer across all four machine learning architectures.
    </p>
    
    <div class="fig-container">
        <img src="{fig4_1_uri}" alt="Results Chart" style="max-height: 44mm;">
        <div class="caption">Figure 4.1: Model performance under Random 10-fold CV versus Phylogenetic CV with 140 Ma buffer.</div>
    </div>
    
    <p>
        When trained on baseline features alone (climate and genomics), all models exhibit near-zero predictive power under random cross-validation: Ridge yields <i>R</i><sup>2</sup> = 0.001, Random Forest yields <i>R</i><sup>2</sup> = -0.025, and XGBoost yields <i>R</i><sup>2</sup> = -0.034. Incorporating 32 Phylo-PCoA tree coordinates transforms within-clade accuracy, elevating <i>R</i><sup>2</sup> to 0.849 &plusmn; 0.02 for Random Forest and 0.848 &plusmn; 0.02 for XGBoost. However, under phylogenetic cross-validation (withholding entire clades with the 140 Ma buffer), all models produce negative <i>R</i><sup>2</sup> values, demonstrating that out-of-clade extrapolation represents a fundamentally distinct regime.
    </p>
    
    <div class="page-footer">6</div>
</div>

<!-- ================= PAGE 10: CHAPTER 4 RESULTS - PART 2 ================= -->
<div class="page cont-page">
    <div class="section-title" style="margin-top: 0; margin-bottom: 3.5mm;">4.2 Biological Accuracy and Extrapolation Drift</div>
    <p>
        The contrast between in-domain interpolation and out-of-clade extrapolation highlights the structural limitations of different model families when navigating phylogenetic manifolds:
    </p>
    <p>
        Under in-domain random cross-validation, tree-based models (Random Forest and XGBoost) achieve tight biological accuracy, predicting mammalian body mass within a factor of &plusmn;2.16&times; to &plusmn;2.18&times; of true values (MAE of 0.336 and 0.338). Within known clades, Phylo-PCoA coordinates act as effective spatial coordinates, allowing decision trees to partition phenotypic variance along continuous evolutionary trajectories. Ridge Regression achieves <i>R</i><sup>2</sup> = 0.781 &plusmn; 0.03 with an MAE of 0.390 (&plusmn;2.45&times; factor error).
    </p>
    <p>
        However, under out-of-clade extrapolation with the 140 Ma buffer, test clades occupy distant regions in the 32-dimensional spectral coordinate space that were never observed during training. Ridge Regression collapses catastrophically to <i>R</i><sup>2</sup> &approx; -55.07 and an RMSE of 3.187. Because linear hyperplanes are unbounded, evaluating distant coordinates produces extreme drift. In contrast, XGBoost maintains bounded predictions due to gradient shrinkage (&eta; = 0.05), achieving the lowest extrapolation RMSE (1.274) and MAE (1.132) among all evaluated models. Random Forest achieves an extrapolation RMSE of 1.423.
    </p>
    
    <div class="section-title" style="margin-top: 5mm; margin-bottom: 3.5mm;">4.3 Clade Withholding and Inflation Gap Analysis</div>
    <p>
        Evaluating models under both validation schemes reveals massive performance inflation in standard benchmarks. The inflation gap &Delta;<i>R</i><sup>2</sup> = <i>R</i><sup>2</sup><sub>Random CV</sub> - <i>R</i><sup>2</sup><sub>Phylo-CV</sub> ranges from +5.02 to +5.11 on baseline features and expands up to +8.83 (and &Delta;<i>R</i><sup>2</sup> &ge; +5.56 across tree models) on augmented features.
    </p>
    <p>
        PhyloGNN (the 2-layer Graph Convolutional Network) fails under both validation protocols, achieving <i>R</i><sup>2</sup> = -0.545 &plusmn; 0.28 on Random CV (&plusmn;13.2&times; error factor, MAE = 1.121) and an RMSE of 1.968 under Phylo-CV. Graph convolution across 3,442 hierarchical tree nodes causes severe over-smoothing, diffusing fine-grained species signals toward the ancestral root.
    </p>
    
    <div class="page-footer">7</div>
</div>

<!-- ================= PAGE 11: CHAPTER 4 RESULTS - PART 3 ================= -->
<div class="page cont-page">
    <div class="section-title" style="margin-top: 0; margin-bottom: 2mm;">4.4 Architectural Benchmark Verdicts and Discussion</div>
    <p>
        The performance metrics and architectural verdicts across the four benchmarked models are summarized in Table 4.1.
    </p>
    
    <div style="font-weight: bold; font-size: 10pt; text-align: center; margin-bottom: 1.5mm;">
        Table 4.1: Benchmark model ranking, interpolation accuracy, biological error, and extrapolation verdicts.
    </div>
    
    <table class="booktabs">
        <thead>
            <tr>
                <th style="width: 15%;">Rank / Model</th>
                <th style="width: 17%;">Architecture</th>
                <th style="width: 15%;">Random CV <i>R</i><sup>2</sup></th>
                <th style="width: 16%;">Biological Accuracy</th>
                <th style="width: 13%;">Phylo-CV RMSE</th>
                <th style="width: 24%;">Architectural Verdict</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>1. Random Forest</strong></td>
                <td>Ensemble (Bagged Trees)</td>
                <td>0.849 &plusmn; 0.02</td>
                <td>&plusmn;2.16&times; (0.336)</td>
                <td>1.423</td>
                <td><strong>Top Precision:</strong> Captures nonlinear phylogenetic gradients cleanly without hyperparameter tuning.</td>
            </tr>
            <tr>
                <td><strong>2. XGBoost Regressor</strong></td>
                <td>Gradient Boosted Trees</td>
                <td>0.848 &plusmn; 0.02</td>
                <td>&plusmn;2.18&times; (0.338)</td>
                <td><strong>1.274 ★</strong></td>
                <td><strong>Best Extrapolator:</strong> Gradient shrinkage (&eta;=0.05) prevents extreme drift on unseen clades.</td>
            </tr>
            <tr>
                <td><strong>3. Ridge Regression</strong></td>
                <td>Cross-Validated Linear (L2)</td>
                <td>0.781 &plusmn; 0.03</td>
                <td>&plusmn;2.45&times; (0.390)</td>
                <td>3.187</td>
                <td><strong>Linear Baseline:</strong> High in-domain fit, but unbounded hyperplanes produce unstable extrapolation.</td>
            </tr>
            <tr>
                <td><strong>4. PhyloGNN (PyG)</strong></td>
                <td>2-Layer Graph Conv (GCN)</td>
                <td>-0.545 &plusmn; 0.28</td>
                <td>&plusmn;13.2&times; (1.121)</td>
                <td>1.968</td>
                <td><strong>Over-Smoothing:</strong> Message passing across 3,442 tree nodes diffuses fine-grained species signals toward the root.</td>
            </tr>
        </tbody>
    </table>
    
    <p>
        The most critical insight is that <em>random cross-validation measures relative relatedness, not biological transferability</em>. When close relatives are present in training, Phylo-PCoA reaches <i>R</i><sup>2</sup> &approx; 0.85 and cuts residual phylogenetic autocorrelation by 85.9% (as measured by Blomberg’s <i>K</i> and Pagel’s <i>&lambda;</i>). However, variance partitioning indicates that 86.93% of predictable body-mass variation represents pure phylogenetic inertia; bioclimatic covariates contribute under 0.1%.
    </p>
    <p>
        Consequently, when tested on an unseen evolutionary clade, the learned coordinates cannot be interpolated, driving out-of-clade <i>R</i><sup>2</sup> below zero. These findings demand the implementation of the <em>Dual-Protocol Standard</em>: studies predicting phenotypic traits must report both in-domain imputation accuracy and out-of-clade extrapolation performance.
    </p>
    
    <div class="page-footer">8</div>
</div>

<!-- ================= PAGE 12: CHAPTER 5 CONCLUSION ================= -->
<div class="page chapter-page">
    <div class="chapter-num">Chapter 5</div>
    <div class="chapter-title">Conclusion and Future Work</div>
    
    <div class="section-title">5.1 Conclusion</div>
    <p>
        This project set out to evaluate how evolutionary history and phylogenetic relatedness constrain machine learning predictions of phenotypic traits across 3,268 mammal species. By benchmarking Ridge Regression, Random Forest, XGBoost, and PhyloGNN under both naive random cross-validation and rigorous clade-withholding phylogenetic cross-validation (Phylo-CV) with a 140 Ma temporal buffer, this study resolves Felsenstein’s dilemma in modern biological machine learning.
    </p>
    <p>
        The results show that the two evaluation regimes behave very differently. Within known clades, Phylogenetic Principal Coordinate Analysis (Phylo-PCoA) successfully compresses the global mammalian tree of life into 32 continuous spectral coordinates, capturing 89.98% of phylogenetic variance. Combining these coordinates with whitened climate and genomic covariates enables high-precision trait imputation: Random Forest and XGBoost achieve <i>R</i><sup>2</sup> &approx; 0.85 with a biological error factor of &plusmn;2.16&times; to &plusmn;2.18&times;, while cutting residual phylogenetic autocorrelation by 85.9%. In contrast, baseline environmental and genomic features alone achieve zero predictive capability (<i>R</i><sup>2</sup> &approx; 0).
    </p>
    <p>
        However, when evaluated on genuinely unseen lineages via Phylo-CV, all models exhibit negative <i>R</i><sup>2</sup>, demonstrating that out-of-clade extrapolation represents a fundamentally distinct and harder learning problem. Standard random cross-validation creates pervasive clade leakage, inflating perceived performance by &Delta;<i>R</i><sup>2</sup> &ge; +5.56 (up to +8.83). XGBoost proves to be the most robust architecture under lineage withholding, maintaining bounded error (RMSE = 1.274) through gradient shrinkage, whereas linear models diverge (<i>R</i><sup>2</sup> &approx; -55.07). Variance decomposition proves that 86.93% of predictable body-mass variation is phylogenetic inertia, while climate adds less than 0.1%. These findings establish that naive cross-validation fools researchers by evaluating memorization of relatives, establishing the imperative for the Dual-Protocol Standard in comparative biology.
    </p>
    
    <div class="page-footer">9</div>
</div>

<!-- ================= PAGE 13: CHAPTER 5 FUTURE WORK ================= -->
<div class="page cont-page">
    <div class="section-title" style="margin-top: 0; margin-bottom: 4mm;">5.2 Future Work</div>
    <p>
        The findings of this project lay the empirical groundwork for several vital research extensions:
    </p>
    
    <ul class="bullet-list">
        <li>
            <strong>Extension to Plant Phenomics and Global Floras:</strong> Mammalian lineages exhibit deep conservation of body mass. Future investigations will test whether Phylo-PCoA and clade withholding yield similar predictability frontiers in plant kingdoms, where phenotypic plasticity, hybridization, and polyploidy present distinct evolutionary dynamics.
        </li>
        <li>
            <strong>Paleontological Integration and Fossil Lineages:</strong> Incorporating morphological measurements and divergence timings from fossil taxa can bridge deep phylogenetic gaps, providing intermediate transitional nodes that reduce extrapolation distance across 140 Ma evolutionary divergences.
        </li>
        <li>
            <strong>Non-Euclidean Hyperbolic Embeddings:</strong> Euclidean coordinate spaces inherently introduce metric distortion when embedding exponentially branching tree hierarchies. Projecting phylogenetic distance matrices onto continuous hyperbolic manifolds (such as the Poincaré ball or Lorentz model) represents a promising direction to capture branching hierarchies without metric distortion.
        </li>
        <li>
            <strong>Adaptive Lineage-Aware Regularization:</strong> Standard gradient boosting and neural regularization techniques do not account for phylogenetic branch lengths. Developing loss functions that dynamically penalize extrapolation drift based on evolutionary distance could improve model stability on deep, isolated clades.
        </li>
    </ul>
    
    <div class="page-footer">10</div>
</div>

<!-- ================= PAGE 14: REFERENCES ================= -->
<div class="page chapter-page">
    <div class="chapter-title" style="margin-bottom: 8mm;">References</div>
    
    <div class="ref-item">
        [1] Blomberg, S. P., Garland, T., and Ives, A. R. (2003). Testing for phylogenetic signal in comparative data: behavioral traits are more labile. <em>Evolution</em>, 57(4):717–745.
    </div>
    <div class="ref-item">
        [2] Diniz-Filho, J. A. F., Ramos de Sant'Ana, C. E., and Bini, L. M. (1998). An eigenvector method for estimating phylogenetic inertia. <em>Evolution</em>, 52(5):1247–1262.
    </div>
    <div class="ref-item">
        [3] Felsenstein, J. (1985). Phylogenies and the comparative method. <em>The American Naturalist</em>, 125(1):1–15.
    </div>
    <div class="ref-item">
        [4] Fey, M. and Lenssen, J. E. (2019). Fast graph representation learning with PyG. <em>ICLR Workshop on Representation Learning on Graphs and Manifolds</em>.
    </div>
    <div class="ref-item">
        [5] Fick, S. E. and Hijmans, R. J. (2017). WorldClim 2: new 1-km spatial resolution climate surfaces for global land areas. <em>International Journal of Climatology</em>, 37(12):4302–4315.
    </div>
    <div class="ref-item">
        [6] Gower, J. C. (1966). Some distance properties of latent root and vector methods used in multivariate analysis. <em>Biometrika</em>, 53(3-4):325–338.
    </div>
    <div class="ref-item">
        [7] Jones, K. E., Bielby, J., Cardillo, M., Fritz, S. A., O'Dell, J., Orme, C. D. L., Safi, K., Sechrest, W., Boakes, E. H., Carbone, C., et al. (2009). PanTHERIA: a species-level database of life history, ecology, and geography of extant and recently extinct mammals. <em>Ecology</em>, 90(9):2648.
    </div>
    <div class="ref-item">
        [8] Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N., Verkuil, R., Kabeli, O., Shmueli, Y., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. <em>Science</em>, 379(6637):1123–1130.
    </div>
    <div class="ref-item">
        [9] Martins, E. P. and Hansen, T. F. (1997). Phylogenies and the comparative method: a general approach to incorporating phylogenetic information into the analysis of interspecific data. <em>The American Naturalist</em>, 149(4):646–667.
    </div>
    <div class="ref-item">
        [10] Meyer, H., Reudenbach, C., Wöllauer, S., and Nauss, T. (2018). Importance of spatial predictor variable selection in machine learning applications - Moving from data reproduction to spatial prediction. <em>Ecological Modelling</em>, 387:144–154.
    </div>
    <div class="ref-item">
        [11] Pagel, M. (1999). Inferring the historical patterns of biological evolution. <em>Nature</em>, 401:877–884.
    </div>
    <div class="ref-item">
        [12] Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., et al. (2017). Cross-validation strategies for data with temporal, spatial, or phylogenetic structure. <em>Ecography</em>, 40(8):913–929.
    </div>
    
    <div class="page-footer">11</div>
</div>

</body>
</html>
"""
    out_html = Path("d:/Projects/Phenix/phenix_report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated HTML -> {out_html}")

    pdf_out = Path("d:/Projects/Phenix/phenix_midsem_eval_report.pdf").resolve()
    html_abs = Path(out_html).resolve()
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    cmd = [
        edge_path,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--virtual-time-budget=6000",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_out}",
        f"{html_abs.as_uri()}"
    ]

    print("Compiling PDF with Edge headless...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if pdf_out.exists() and pdf_out.stat().st_size > 10000:
        print(f"SUCCESS: Compiled PDF -> {pdf_out} ({pdf_out.stat().st_size} bytes)")
    else:
        print(f"FAILED to generate PDF: {res.stderr}")

if __name__ == "__main__":
    build_perfect_report()
