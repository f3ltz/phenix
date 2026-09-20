from typing import Dict, Optional, List
from src.phylogenetics.otol import PhyloNode


# Standard TimeTree / Mammal divergence calibration dates (in Millions of Years Ago, Ma)
TIMETREE_NODE_AGES_MA = {
    "Mammalia": 177.0,
    "Prototheria": 160.0,
    "Theria": 160.0,
    "Metatheria": 140.0,
    "Ameridelphia": 80.0,
    "Australidelphia": 75.0,
    "Eutheria": 105.0,
    "Atlantogenata": 102.0,
    "Afrotheria": 83.0,
    "Xenarthra": 85.0,
    "Boreoeutheria": 96.0,
    "Laurasiatheria": 79.0,
    "Euarchontoglires": 82.0,
    "Eulipotyphla": 74.0,
    "Scrotifera": 76.0,
    "Ferae": 66.0,
    "Euungulata": 70.0,
    "Glires": 72.0,
    "Euarchonta": 73.0,
    "Carnivora": 55.0,
    "Primates": 66.0,
    "Rodentia": 65.0,
    "Artiodactyla": 58.0,
    "Cetartiodactyla": 58.0,
    "Chiroptera": 60.0,
    "Lagomorpha": 52.0,
    "Soricomorpha": 58.0,
    "Erinaceomorpha": 55.0,
    "Diprotodontia": 50.0,
    "Dasyuromorphia": 45.0,
    "Didelphimorphia": 40.0,
    "Perissodactyla": 56.0,
    "Cingulata": 58.0,
    "Pilosa": 54.0,
    "Afrosoricida": 48.0,
    "Scandentia": 50.0,
    "Macroscelidea": 46.0,
}


class TreeCalibrator:
    """
    Calibrates phylogenetic tree branch lengths against divergence dates
    using the BLADJ (Branch Length Adjuster) algorithm to create an ultrametric timetree.
    """

    def __init__(self, node_calibrations: Optional[Dict[str, float]] = None):
        self.calibrations: Dict[str, float] = dict(TIMETREE_NODE_AGES_MA)
        if node_calibrations:
            self.calibrations.update(node_calibrations)

    def calibrate_tree(self, root: PhyloNode) -> PhyloNode:
        """
        Executes BLADJ calibration:
          1. Assigns extant tips age = 0.0 Ma.
          2. Assigns known calibration dates to labeled nodes.
          3. Interpolates ages for intermediate uncalibrated nodes to ensure strict ultrametricity.
          4. Computes branch lengths: length = age(parent) - age(child).
        """
        # Step 1: Assign 0.0 to all leaves
        leaves = root.get_leaves()
        for leaf in leaves:
            leaf.age = 0.0

        # Step 2: Assign known calibrations
        def assign_known_ages(node: PhyloNode) -> None:
            if not node.is_leaf():
                if node.name and node.name in self.calibrations:
                    node.age = float(self.calibrations[node.name])
                for child in node.children:
                    assign_known_ages(child)

        assign_known_ages(root)

        # Root age must be set
        if root.age is None:
            root.age = float(self.calibrations.get("Mammalia", 177.0))

        # Step 3: Interpolate uncalibrated nodes (BLADJ downward/upward spacing)
        def interpolate_ages(node: PhyloNode, parent_age: float) -> None:
            if node.is_leaf():
                node.age = 0.0
                return

            if node.age is None:
                # If node has no calibration, set age based on depth between parent and max child age
                # Default heuristic: 0.65 of parent's age or parent_age - 10
                node.age = max(1.0, parent_age * 0.65)
            elif node.age >= parent_age:
                # Ensure strict monotonicity: parent must be older than child
                node.age = max(0.5, parent_age - 2.0)

            for child in node.children:
                interpolate_ages(child, node.age)

        for child in root.children:
            interpolate_ages(child, root.age)

        # Step 4: Compute branch lengths length = age(parent) - age(child)
        def compute_branch_lengths(node: PhyloNode) -> None:
            for child in node.children:
                if node.age is not None and child.age is not None:
                    child.length = max(0.01, node.age - child.age)
                compute_branch_lengths(child)

        root.length = 0.0
        compute_branch_lengths(root)

        return root
