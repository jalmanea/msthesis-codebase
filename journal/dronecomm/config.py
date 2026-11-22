"""Dataclass-based configuration with YAML loading support."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class AntennaConfig:
    """Antenna parameters for downlink and backhaul."""

    dl_g_max_dbi: float = 18.0
    dl_beamwidth_deg: float = 60.0
    dl_sla_db: float = 20.0
    bh_g_max_dbi: float = 12.0
    bh_beamwidth_deg: float = 30.0
    bh_sla_db: float = 25.0


@dataclass(frozen=True)
class ChannelConfig:
    """Channel model parameters."""

    environment: str = "urban"
    carrier_freq_hz: float = 2.0e9


@dataclass(frozen=True)
class NetworkConfig:
    """Network and power parameters."""

    n_drones: int = 15
    altitude_m: float = 150.0
    p_tx_dl_dbm: float = 30.0  # 1 W
    p_tx_bh_dbm: float = 27.0  # 0.5 W
    bandwidth_hz: float = 10.0e6  # 10 MHz
    noise_psd_dbm_hz: float = -174.0
    noise_figure_db: float = 0.0  # receiver noise figure, added to the thermal PSD
    sinr_threshold_db: float = 3.0
    min_separation_m: float = 100.0
    dl_tilt_deg: float = 0.0
    bh_tilt_deg: float = 90.0
    z_min_m: float = 50.0  # altitude bounds of every altitude rule and search
    z_max_m: float = 300.0
    z_step_m: float = 10.0  # spacing of the sweep and greedy altitude grid

    @property
    def noise_dbm_hz(self) -> float:
        """Effective noise PSD at the receiver: thermal PSD plus noise figure."""
        return self.noise_psd_dbm_hz + self.noise_figure_db

    def altitude_grid(self) -> np.ndarray:
        """Candidate altitudes z_min, z_min + step, ..., z_max [m]."""
        n = int(round((self.z_max_m - self.z_min_m) / self.z_step_m)) + 1
        return self.z_min_m + self.z_step_m * np.arange(n)


@dataclass(frozen=True)
class ScenarioConfig:
    """Scenario generation parameters."""

    area_size_m: float = 2000.0
    n_clusters: int = 5
    users_per_cluster_mean: int = 40
    users_per_cluster_std: int = 10
    cluster_spread_m: float = 100.0


@dataclass(frozen=True)
class CapacityConfig:
    """Load-aware rate model (see dronecomm.capacity)."""

    se_alpha: float = 0.6  # TR 36.942 Annex A.1, DL
    se_sinr_min_db: float = -10.0
    se_max_bps_hz: float = 4.4
    max_users_per_cell: int | None = None  # None: no admission cap


@dataclass(frozen=True)
class BackhaulConfig:
    """Drone-to-gateway transport (see dronecomm.backhaul)."""

    link_model: str = "none"  # none: unconstrained | sinr: shared-channel links | fixed: fixed_link_bps
    bandwidth_hz: float = 20.0e6  # backhaul channel, separate from the access band
    noise_figure_db: float = 5.0  # drone (relay) receiver noise figure
    fixed_link_bps: float = 1.0e9  # per-link capacity when link_model is fixed
    gateway_bps: float | None = None  # gateway ground link; None: unconstrained
    link_range_m: float | None = None  # longest single hop; longer links get relay drones; None: any
    interfaces: int | None = None  # backhaul radios per drone; links beyond it time-share; None: any
    half_duplex: bool = False  # links send one direction at a time (802.11ad is TDD)
    two_way: bool = False  # the tier's traffic flows both ways (calls); set per cell by the runners
    topology: str = "mst_corner"  # mst_corner | mst_central | balanced (dronecomm.backhaul)


@dataclass(frozen=True)
class DemandConfig:
    """Who the simulated user points stand for (see dronecomm.capacity)."""

    population: float | None = None  # people represented by all points; None: one per point
    device_share: float = 1.0  # share of people carrying a phone
    activity_factor: float = 1.0  # share of devices active at once
    r_min_bps: float = 38.0e3  # service minimum per active user

    def active_per_point(self, n_points: int) -> float:
        """Active users each simulated point stands for."""
        people = 1.0 if self.population is None else self.population / n_points
        return people * self.device_share * self.activity_factor


@dataclass(frozen=True)
class SimulationConfig:
    """Monte Carlo simulation parameters."""

    n_trials: int = 100
    seed: int = 42
    placement_strategy: str = "kmeans"


@dataclass(frozen=True)
class Config:
    """Top-level configuration combining all sub-configs."""

    antenna: AntennaConfig = field(default_factory=AntennaConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    capacity: CapacityConfig = field(default_factory=CapacityConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    backhaul: BackhaulConfig = field(default_factory=BackhaulConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(
            antenna=AntennaConfig(**data.get("antenna", {})),
            channel=ChannelConfig(**data.get("channel", {})),
            network=NetworkConfig(**data.get("network", {})),
            scenario=ScenarioConfig(**data.get("scenario", {})),
            simulation=SimulationConfig(**data.get("simulation", {})),
            capacity=CapacityConfig(**data.get("capacity", {})),
            demand=DemandConfig(**data.get("demand", {})),
            backhaul=BackhaulConfig(**data.get("backhaul", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
