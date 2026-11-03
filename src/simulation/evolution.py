"""
Macroevolutionary Trait Simulator for PHENIX
Implements continuous-time trait evolution on calibrated phylogenetic trees:
  - Brownian Motion (BM): Neutral phenotypic drift (Felsenstein 1985).
  - Ornstein-Uhlenbeck (OU): Stabilizing selection toward environmental optima (Hansen 1997, Butler & King 2004).
"""

from typing import Dict, List, Optional, Union
import numpy as np
from src.phylogenetics.otol import PhyloNode


class MacroevolutionarySimulator:
    """
    Simulates macroevolutionary continuous trait evolution along phylogenetic timetrees.
    Supports both tree-wide evolution (from crown root to tips) and forward evolutionary
    projection (from extant tips forward into the future under changing selection regimes).
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def simulate_brownian_motion(
        self,
        root: PhyloNode,
        sigma2: float = 0.05,
        root_value: float = 2.5,
    ) -> Dict[str, float]:
        """
        Simulates neutral phenotypic drift along tree branches:
          X_child = X_parent + N(0, sigma2 * delta_t)
        Returns mapping of tip taxon name -> simulated trait value.
        """
        tip_values: Dict[str, float] = {}

        def traverse(node: PhyloNode, current_val: float) -> None:
            if node.is_leaf():
                if node.name:
                    tip_values[node.name] = float(current_val)
                return

            for child in node.children:
                dt = max(float(child.length), 0.001)
                variance = float(sigma2 * dt)
                noise = self.rng.normal(0.0, np.sqrt(variance))
                child_val = current_val + noise
                traverse(child, child_val)

        traverse(root, root_value)
        return tip_values

    def simulate_ornstein_uhlenbeck(
        self,
        root: PhyloNode,
        alpha: float = 0.05,
        sigma2: float = 0.05,
        root_value: float = 2.5,
        environmental_optima: Optional[Dict[str, float]] = None,
        default_optimum: float = 2.5,
    ) -> Dict[str, float]:
        """
        Simulates stabilizing selection toward an adaptive phenotypic optimum:
          dX(t) = alpha * (theta - X(t)) dt + sigma * dW(t)
        Exact discrete-time transition formula:
          E[X(t + dt) | X(t)] = theta + (X(t) - theta) * exp(-alpha * dt)
          Var[X(t + dt) | X(t)] = (sigma2 / (2 * alpha)) * (1 - exp(-2 * alpha * dt))
        """
        tip_values: Dict[str, float] = {}
        alpha = max(float(alpha), 1e-5)
        sigma2 = max(float(sigma2), 1e-5)

        def get_optimum(node: PhyloNode) -> float:
            if environmental_optima and node.name and node.name in environmental_optima:
                return float(environmental_optima[node.name])
            return float(default_optimum)

        def traverse(node: PhyloNode, current_val: float) -> None:
            if node.is_leaf():
                if node.name:
                    tip_values[node.name] = float(current_val)
                return

            for child in node.children:
                dt = max(float(child.length), 0.001)
                theta = get_optimum(child)

                exp_decay = np.exp(-alpha * dt)
                mean_val = theta + (current_val - theta) * exp_decay
                var_val = (sigma2 / (2.0 * alpha)) * (1.0 - np.exp(-2.0 * alpha * dt))
                var_val = max(var_val, 1e-8)

                child_val = float(self.rng.normal(mean_val, np.sqrt(var_val)))
                traverse(child, child_val)

        traverse(root, root_value)
        return tip_values

    def forward_simulate_tips(
        self,
        current_traits: Dict[str, float],
        future_optima: Dict[str, float],
        years_forward_ma: float = 1.0,
        alpha: float = 0.1,
        sigma2: float = 0.02,
    ) -> Dict[str, float]:
        """
        Forward-projects extant tip species forward in time under climate change:
          Extant tip value X(0) -> Future tip value X(delta_t)
        Models microevolutionary/macroevolutionary adaptive tracking vs. inertia:
          X(delta_t) = theta_future + (X(0) - theta_future) * exp(-alpha * delta_t) + noise
        """
        alpha = max(float(alpha), 1e-5)
        sigma2 = max(float(sigma2), 1e-5)
        dt = float(years_forward_ma)

        exp_decay = np.exp(-alpha * dt)
        var_val = (sigma2 / (2.0 * alpha)) * (1.0 - np.exp(-2.0 * alpha * dt))
        std_val = np.sqrt(max(var_val, 1e-8))

        future_traits: Dict[str, float] = {}
        for taxon, x0 in current_traits.items():
            theta = future_optima.get(taxon, x0)
            mean_val = theta + (x0 - theta) * exp_decay
            val = float(self.rng.normal(mean_val, std_val))
            future_traits[taxon] = val

        return future_traits

    def compute_bergmann_optima(
        self,
        bio1_temperatures_c: np.ndarray,
        base_body_mass_log10: np.ndarray,
        temperature_coefficient: float = -0.015,
        delta_temp_c: float = 3.0,
    ) -> np.ndarray:
        """
        Calculates shifting phenotypic optima based on Bergmann's rule:
          theta_future = theta_current + beta_temp * delta_temp
        In endotherms, higher ambient temperatures favor smaller body size for heat dissipation:
          beta_temp < 0 (e.g. -0.015 log10 mass per °C warming ~ 3.4% mass decrease per °C).
        """
        shifted_optima = base_body_mass_log10 + (temperature_coefficient * delta_temp_c)
        return shifted_optima
