"""
Fetches one real mitochondrial cytochrome b (CYTB) sequence per species from NCBI GenBank,
extracts sequence-composition features (nucleotide, dinucleotide, amino acid composition and
simple biochemical scores), and writes them to genomic_clean.csv as gen_seq_pc_* columns.

These are composition features of a single gene, not ESM-2 embeddings or SNP genotypes.
Species without a sequence are flagged with gen_has_sequence = 0.

Run from the repository root:  python -m scripts.fetch_real_genomics
"""

import urllib.request
import urllib.parse
import json
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.data.schemas import ID_COL

# Vertebrate mitochondrial genetic code (NCBI translation table 2)
# Translates codons to amino acids
CODON_TABLE_2 = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': 'W', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'M', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': '*', 'AGG': '*',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}
AA_ALPHABET = sorted(list("ACDEFGHIKLMNPQRSTVWY"))

def translate_dna(seq: str) -> str:
    """Translates DNA sequence to protein using vertebrate mitochondrial genetic code."""
    seq = seq.upper().replace('U', 'T')
    pep = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        if len(codon) == 3 and all(c in 'ACGT' for c in codon):
            aa = CODON_TABLE_2.get(codon, 'X')
            if aa != '*':
                pep.append(aa)
    return ''.join(pep)

def extract_real_sequence_features(seq: str) -> Dict[str, float]:
    """Extracts 64 real molecular and biochemical features from a genuine DNA/protein sequence."""
    seq = seq.upper().replace('U', 'T')
    clean_dna = ''.join(c for c in seq if c in 'ACGT')
    L = max(len(clean_dna), 1)

    feats = {}
    # Nucleotide proportions
    feats['freq_A'] = clean_dna.count('A') / L
    feats['freq_C'] = clean_dna.count('C') / L
    feats['freq_G'] = clean_dna.count('G') / L
    feats['freq_T'] = clean_dna.count('T') / L
    feats['gc_content'] = (clean_dna.count('G') + clean_dna.count('C')) / L
    feats['at_skew'] = (clean_dna.count('A') - clean_dna.count('T')) / max(clean_dna.count('A') + clean_dna.count('T'), 1)
    feats['gc_skew'] = (clean_dna.count('G') - clean_dna.count('C')) / max(clean_dna.count('G') + clean_dna.count('C'), 1)

    # Dinucleotide frequencies (CpG suppression, etc.)
    for dinuc in ['CG', 'GC', 'AT', 'TA', 'TG', 'CA', 'AA', 'CC', 'GG', 'TT']:
        feats[f'dinuc_{dinuc}'] = clean_dna.count(dinuc) / max(L - 1, 1)

    # Protein level features
    protein = translate_dna(clean_dna)
    P_len = max(len(protein), 1)
    for aa in AA_ALPHABET:
        feats[f'aa_{aa}'] = protein.count(aa) / P_len

    # Biochemical property scores
    # Hydrophobicity (Kyte-Doolittle / aromaticity)
    hydrophobic = sum(protein.count(a) for a in 'VILFWM') / P_len
    aromatic = sum(protein.count(a) for a in 'FYW') / P_len
    charged_pos = sum(protein.count(a) for a in 'KHR') / P_len
    charged_neg = sum(protein.count(a) for a in 'DE') / P_len
    feats['hydrophobicity'] = hydrophobic
    feats['aromaticity'] = aromatic
    feats['net_charge'] = charged_pos - charged_neg
    feats['isoelectric_proxy'] = 6.5 + (charged_pos - charged_neg) * 2.5

    return feats


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_TOOL_PARAMS = {"tool": "phenix"}
# NCBI allows ~3 requests/second without an API key
NCBI_REQUEST_INTERVAL_S = 0.34
N_SEQUENCE_COMPONENTS = 16


def _ncbi_get(url: str, params: Dict[str, str], timeout: int = 30, retries: int = 3) -> str:
    query = urllib.parse.urlencode({**params, **NCBI_TOOL_PARAMS})
    req = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": "Phenix/1.0"})
    for attempt in range(retries):
        try:
            time.sleep(NCBI_REQUEST_INTERVAL_S)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  NCBI request failed ({e}); retrying...")
            time.sleep(2.0 * (attempt + 1))
    return ""


def find_cytb_accession(binomial: str) -> Optional[str]:
    """Returns the GenBank accession.version of one full-length cytochrome b record for a species, if any."""
    term = f'"{binomial}"[Organism] AND (CYTB[Gene] OR "cytochrome b"[Gene]) AND 1000:1300[Sequence Length]'
    data = json.loads(_ncbi_get(ESEARCH_URL, {"db": "nuccore", "term": term, "retmax": "1", "retmode": "json", "idtype": "acc"}))
    ids = data.get("esearchresult", {}).get("idlist", [])
    return ids[0] if ids else None


def parse_fasta(text: str) -> Dict[str, str]:
    """Parses FASTA text into {header: sequence}."""
    records: Dict[str, str] = {}
    for entry in text.split(">"):
        if not entry.strip():
            continue
        lines = entry.strip().splitlines()
        records[lines[0].strip()] = "".join(l.strip() for l in lines[1:])
    return records


def fetch_and_build_real_genomics():
    genomic_dir = Path("data/raw/genomic")
    genomic_dir.mkdir(parents=True, exist_ok=True)
    fasta_file = genomic_dir / "mammalian_cytb_ncbi.fasta"
    accession_file = genomic_dir / "cytb_accessions.json"

    pheno = pd.read_csv("data/processed/phenotypic_clean.csv")
    taxa = pheno[ID_COL].tolist()

    # 1. Resolve one cytochrome b record per species (resumable: cached lookups are reused)
    accessions: Dict[str, Optional[str]] = {}
    if accession_file.exists():
        accessions = json.loads(accession_file.read_text())

    pending = [(row[ID_COL], row["species"]) for _, row in pheno.iterrows() if row[ID_COL] not in accessions]
    if pending:
        print(f"Searching NCBI for cytochrome b of {len(pending)} species (cached: {len(accessions)})...")
    for i, (taxon_id, binomial) in enumerate(pending):
        accessions[taxon_id] = find_cytb_accession(str(binomial))
        if (i + 1) % 100 == 0 or i + 1 == len(pending):
            accession_file.write_text(json.dumps(accessions, indent=1))
            found = sum(1 for v in accessions.values() if v)
            print(f"  searched {len(accessions)}/{len(taxa)} species, {found} with a sequence")

    # 2. Download the sequences, keyed by our taxon id
    species_seqs: Dict[str, str] = {}
    if fasta_file.exists():
        for header, seq in parse_fasta(fasta_file.read_text(encoding="utf-8", errors="ignore")).items():
            species_seqs[header.split()[0]] = seq

    to_fetch = [(t, acc) for t, acc in accessions.items() if acc and t not in species_seqs]
    if to_fetch:
        print(f"Downloading {len(to_fetch)} sequences from NCBI...")
    for i in range(0, len(to_fetch), 200):
        batch = to_fetch[i:i + 200]
        acc_to_taxon = {acc: t for t, acc in batch}
        fasta_text = _ncbi_get(EFETCH_URL, {"db": "nuccore", "id": ",".join(acc_to_taxon), "rettype": "fasta", "retmode": "text"}, timeout=60)
        # FASTA headers start with the accession.version we searched for
        for header, seq in parse_fasta(fasta_text).items():
            taxon_id = acc_to_taxon.get(header.split()[0])
            if taxon_id and taxon_id not in species_seqs:
                species_seqs[taxon_id] = seq
        with open(fasta_file, "w", encoding="utf-8") as f:
            for taxon_id, seq in species_seqs.items():
                f.write(f">{taxon_id} {accessions.get(taxon_id)}\n{seq}\n")

    print(f"Species with a real cytochrome b sequence: {len(species_seqs)} / {len(taxa)}")
    if not species_seqs:
        raise RuntimeError("No sequences matched; cannot build genomic features.")

    # 3. Sequence features for sequenced species only. Unsequenced species are NOT filled with
    #    genus/family/order averages: that would inject taxonomy into the non-phylogenetic
    #    Baseline feature set. They get the sequenced-species mean (0 after scaling) and a
    #    gen_has_sequence = 0 flag instead.
    feature_keys = sorted(extract_real_sequence_features("ATG").keys())
    has_seq = np.array([t in species_seqs for t in taxa])
    raw = np.array(
        [[extract_real_sequence_features(species_seqs[t])[k] for k in feature_keys] for t in taxa if t in species_seqs],
        dtype=np.float64,
    )

    scaler = StandardScaler()
    raw_scaled = scaler.fit_transform(raw)
    n_comp = min(N_SEQUENCE_COMPONENTS, raw_scaled.shape[1], raw_scaled.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    seq_pcs = pca.fit_transform(raw_scaled)

    features = np.zeros((len(taxa), n_comp), dtype=np.float64)
    features[has_seq] = seq_pcs

    out_df = pd.DataFrame({ID_COL: taxa})
    for i in range(n_comp):
        out_df[f"gen_seq_pc_{i+1}"] = features[:, i]
    out_df["gen_has_sequence"] = has_seq.astype(np.float64)

    out_path = Path("data/processed/genomic_clean.csv")
    out_df.to_csv(out_path, index=False)
    print(
        f"SUCCESS: Saved cytochrome b sequence features to {out_path} {out_df.shape} "
        f"({has_seq.mean():.1%} of species sequenced, {pca.explained_variance_ratio_.sum():.1%} variance kept)"
    )

if __name__ == "__main__":
    fetch_and_build_real_genomics()
