#!/usr/bin/env python3
"""Run a Monte Carlo simulation and display results."""

import argparse
import sys
from pathlib import Path

# Add parent to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dronecomm.config import Config
from dronecomm.simulation import run_monte_carlo


def main():
    parser = argparse.ArgumentParser(description="Drone deployment simulation")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file. Uses defaults if not provided.",
    )
    parser.add_argument(
        "--trials", type=int, default=None,
        help="Override number of Monte Carlo trials.",
    )
    args = parser.parse_args()

    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    if args.trials is not None:
        # Create new config with overridden trial count
        from dataclasses import replace
        sim = replace(config.simulation, n_trials=args.trials)
        config = replace(config, simulation=sim)

    print(f"Running {config.simulation.n_trials} trials with "
          f"{config.network.n_drones} drones, "
          f"{config.simulation.placement_strategy} placement...")

    result = run_monte_carlo(config, verbose=True)

    print(f"\nDone. {result.n_trials} trials completed.")


if __name__ == "__main__":
    main()
