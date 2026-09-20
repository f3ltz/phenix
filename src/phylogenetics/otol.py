import json
import re
import urllib.request
from typing import List, Dict, Optional, Tuple, Union, Set
import pandas as pd
from src.data.schemas import ID_COL, normalize_taxon_id


class PhyloNode:
    """Represents a phylogenetic tree node with topology, branch length, and divergence age."""

    def __init__(
        self,
        name: Optional[str] = None,
        length: float = 1.0,
        age: Optional[float] = None,
    ):
        self.name = name
        self.length = float(length)
        self.age = float(age) if age is not None else None
        self.children: List['PhyloNode'] = []
        self.parent: Optional['PhyloNode'] = None

    def add_child(self, child: 'PhyloNode') -> None:
        child.parent = self
        self.children.append(child)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_leaves(self) -> List['PhyloNode']:
        if self.is_leaf():
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

    def get_leaf_names(self) -> List[str]:
        return [leaf.name for leaf in self.get_leaves() if leaf.name]

    def to_newick(self, include_branch_lengths: bool = True) -> str:
        """Serializes tree to standard Newick format string."""
        if self.is_leaf():
            base = self.name or ""
        else:
            child_strs = [child.to_newick(include_branch_lengths) for child in self.children]
            base = f"({','.join(child_strs)}){self.name or ''}"

        if include_branch_lengths and self.parent is not None:
            return f"{base}:{self.length:.4f}"
        return base

    @classmethod
    def from_newick(cls, newick_str: str) -> 'PhyloNode':
        """Parses Newick format string into PhyloNode hierarchy."""
        cleaned = newick_str.strip()
        if cleaned.endswith(";"):
            cleaned = cleaned[:-1]

        def parse_tokens(s: str) -> 'PhyloNode':
            s = s.strip()
            # If leaf
            if not s.startswith("("):
                parts = s.split(":")
                name = parts[0].strip() or None
                length = float(parts[1]) if len(parts) > 1 else 1.0
                return cls(name=name, length=length)

            # Find matching closing parenthesis
            depth = 0
            split_idx = -1
            for i, ch in enumerate(s):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        split_idx = i
                        break

            inside = s[1:split_idx]
            rest = s[split_idx + 1 :]

            name = None
            length = 1.0
            if rest:
                parts = rest.split(":")
                name = parts[0].strip() or None
                if len(parts) > 1 and parts[1].strip():
                    length = float(parts[1].strip())

            node = cls(name=name, length=length)

            # Split children on top-level commas
            child_tokens = []
            curr = []
            depth = 0
            for ch in inside:
                if ch == "(":
                    depth += 1
                    curr.append(ch)
                elif ch == ")":
                    depth -= 1
                    curr.append(ch)
                elif ch == "," and depth == 0:
                    child_tokens.append("".join(curr))
                    curr = []
                else:
                    curr.append(ch)
            if curr:
                child_tokens.append("".join(curr))

            for tok in child_tokens:
                node.add_child(parse_tokens(tok))

            return node

        return parse_tokens(cleaned)


# Major mammalian higher-level phylogenetic hierarchy (based on TimeTree / Meredith et al.)
MAMMAL_CLADE_HIERARCHY = {
    # Monotremata
    "Monotremata": ("Prototheria", "Mammalia"),
    # Metatheria (Marsupials)
    "Didelphimorphia": ("Ameridelphia", "Metatheria"),
    "Paucituberculata": ("Ameridelphia", "Metatheria"),
    "Dasyuromorphia": ("Australidelphia", "Metatheria"),
    "Peramelemorphia": ("Australidelphia", "Metatheria"),
    "Notoryctemorphia": ("Australidelphia", "Metatheria"),
    "Diprotodontia": ("Australidelphia", "Metatheria"),
    "Microbiotheria": ("Australidelphia", "Metatheria"),
    # Eutheria -> Xenarthra
    "Cingulata": ("Xenarthra", "Eutheria"),
    "Pilosa": ("Xenarthra", "Eutheria"),
    # Eutheria -> Afrotheria
    "Afrosoricida": ("Afrotheria", "Eutheria"),
    "Macroscelidea": ("Afrotheria", "Eutheria"),
    "Tubulidentata": ("Afrotheria", "Eutheria"),
    "Hyracoidea": ("Afrotheria", "Eutheria"),
    "Proboscidea": ("Afrotheria", "Eutheria"),
    "Sirenia": ("Afrotheria", "Eutheria"),
    # Eutheria -> Boreoeutheria -> Laurasiatheria
    "Erinaceomorpha": ("Eulipotyphla", "Laurasiatheria"),
    "Soricomorpha": ("Eulipotyphla", "Laurasiatheria"),
    "Chiroptera": ("Scrotifera", "Laurasiatheria"),
    "Carnivora": ("Ferae", "Laurasiatheria"),
    "Pholidota": ("Ferae", "Laurasiatheria"),
    "Perissodactyla": ("Euungulata", "Laurasiatheria"),
    "Artiodactyla": ("Euungulata", "Laurasiatheria"),
    "Cetartiodactyla": ("Euungulata", "Laurasiatheria"),
    # Eutheria -> Boreoeutheria -> Euarchontoglires
    "Rodentia": ("Glires", "Euarchontoglires"),
    "Lagomorpha": ("Glires", "Euarchontoglires"),
    "Primates": ("Euarchonta", "Euarchontoglires"),
    "Scandentia": ("Euarchonta", "Euarchontoglires"),
    "Dermoptera": ("Euarchonta", "Euarchontoglires"),
}


class OToLClient:
    """
    Open Tree of Life (OToL) interface for Taxonomic Name Resolution (TNRS)
    and induced subtree topology extraction, with fallback mammalian consensus backbone.
    """

    TNRS_URL = "https://api.opentreeoflife.org/v3/tnrs/match_names"
    SUBTREE_URL = "https://api.opentreeoflife.org/v3/tree_of_life/induced_subtree"

    def match_names_tnrs(self, names: List[str], timeout: int = 10) -> Dict[str, int]:
        """Queries OToL TNRS API to resolve scientific names to OTT IDs."""
        ott_map = {}
        try:
            payload = json.dumps({"names": names[:250]}).encode("utf-8")
            req = urllib.request.Request(
                self.TNRS_URL,
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "Phenix/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for result in data.get("results", []):
                    query_name = result.get("name")
                    matches = result.get("matches", [])
                    if matches:
                        ott_id = matches[0].get("taxon", {}).get("ott_id")
                        if ott_id:
                            ott_map[normalize_taxon_id(query_name)] = ott_id
        except Exception as e:
            print(f"[OToLClient] TNRS online query notice: {e}. Falling back to internal taxonomy.")
        return ott_map

    def build_mammal_backbone_topology(
        self,
        taxa_df: pd.DataFrame,
    ) -> PhyloNode:
        """
        Constructs an evolutionary consensus tree topology for target taxa
        based on mammalian phylogenetic hierarchy and classification.
        Guarantees zero missing tips and strictly matches canonical_taxon_id.
        """
        if ID_COL not in taxa_df.columns:
            raise ValueError(f"Taxa dataframe must contain '{ID_COL}'")

        # Root: Mammalia
        root = PhyloNode(name="Mammalia", length=0.0)

        # Build higher-level nodes
        nodes: Dict[str, PhyloNode] = {"Mammalia": root}

        def get_or_create_node(name: str, parent_name: Optional[str] = None) -> PhyloNode:
            if name not in nodes:
                node = PhyloNode(name=name, length=1.0)
                nodes[name] = node
                if parent_name:
                    p = get_or_create_node(parent_name)
                    p.add_child(node)
            return nodes[name]

        # Target taxa set
        target_set = set(taxa_df[ID_COL].tolist())

        # Group taxa by order and family
        for _, row in taxa_df.iterrows():
            taxon_id = row[ID_COL]
            order = str(row.get("order", "Unknown"))
            family = str(row.get("family", order))

            if order in MAMMAL_CLADE_HIERARCHY:
                higher_clade, root_clade = MAMMAL_CLADE_HIERARCHY[order]
                get_or_create_node(root_clade, "Mammalia")
                get_or_create_node(higher_clade, root_clade)
                order_node = get_or_create_node(order, higher_clade)
            else:
                order_node = get_or_create_node(order, "Mammalia")

            fam_node = get_or_create_node(f"fam_{family}", order)
            leaf_node = PhyloNode(name=taxon_id, length=1.0)
            fam_node.add_child(leaf_node)

        # Prune any internal nodes that have no target taxa leaves
        def prune_empty(node: PhyloNode) -> bool:
            if node.is_leaf():
                return bool(node.name and node.name in target_set)
            valid_children = []
            for child in node.children:
                if prune_empty(child):
                    valid_children.append(child)
            node.children = valid_children
            return len(node.children) > 0

        prune_empty(root)
        return root
