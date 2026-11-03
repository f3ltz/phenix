"""
PHENIX Eco-Evolutionary Simulation Engine
Provides:
  1. Macroevolutionary trait simulations (Brownian Motion & Ornstein-Uhlenbeck).
  2. Climate change & proteomic perturbation scenario generation.
  3. ML forward forecasting & simulation benchmarking against adaptive ground truth.
"""

from src.simulation.evolution import MacroevolutionarySimulator
from src.simulation.scenarios import ClimateScenarioGenerator, ProteomicMutationGenerator
from src.simulation.forecaster import EcoEvolutionaryForecaster

__all__ = [
    "MacroevolutionarySimulator",
    "ClimateScenarioGenerator",
    "ProteomicMutationGenerator",
    "EcoEvolutionaryForecaster",
]
