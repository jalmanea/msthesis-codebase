# dronecomm

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Geometry-aware drone deployment simulation for disaster emergency communications.

This package implements the simulation framework for the paper *"Geometry-Aware Optimal Drone Deployment for Disaster Emergency Communications"*. It models inter-drone interference accounting for directional antenna radiation patterns (main lobes, sidelobes, backlobes) and evaluates tiered connectivity for ground users in post-disaster scenarios where terrestrial infrastructure is unavailable.

> **Terminology note — paper vs. codebase:** The paper (`paper/main_icc_sweep.tex`) uses the term **"userlink"** for the ground-facing antenna link (the link from a drone to ground users). The codebase and all scripts/notebooks use the older term **"downlink"** (abbreviated `dl`, e.g. `dl_tilt_rad`, `DOWNLINK_ANTENNA`, `P_tx_dl`). These refer to the same concept. The paper term was updated to "userlink" to better reflect the drone-to-user direction; the codebase identifier names were left unchanged to avoid breaking existing scripts and experiments.

## Problem overview

Deploy up to $N$ drones as aerial base stations to serve $M$ ground users (survivors and first responders) while **minimizing inter-drone interference** caused by antenna geometry. Unlike existing approaches that treat interference as isotropic, this framework models interference as a function of:

- 3D drone positions and inter-drone distances
- Antenna orientations (tilt and azimuth)
- Directional radiation patterns (parameterized by beamwidth and sidelobe level)

Users are split into two tiers:
- **Critical users** (first responders, command/control) require hard SINR guarantees ($\geq 10$ dB)
- **Regular users** (general population) receive best-effort coverage ($\geq 3$ dB)

---

## Platform model

This framework models **generic aerial base stations** (UAV-mounted cell sites, also called Flying Base Stations or Cell-on-Wings) rather than a specific drone platform. The simulation abstracts away airframe type (multi-rotor vs fixed-wing), flight dynamics, battery constraints, and specific radio hardware.

### Dual-antenna architecture

Each drone carries two independent directional antennas operating on separate frequency bands:

1. **Downlink antenna** (18 dBi, 60° beamwidth) - Serves ground users with wide-area coverage, typically pointed nadir (straight down)
2. **Backhaul antenna** (12 dBi, 30° beamwidth) - Forms a drone-to-drone mesh network for data relay to command centers, typically pointed horizontally

This architecture is realistic and widely deployed:
- **AT&T Flying COW** (Cell on Wings) - disaster recovery
- **Nokia F-Cell** - drone-mounted LTE small cells
- **AERPAW testbed** - experimental aerial cellular platform (used for channel model validation in this work)
- **Qualcomm/EHang aerial base stations** - commercial deployments

The frequency isolation between downlink and backhaul bands ensures zero cross-interference, allowing simultaneous operation.

---

## System model

### Antenna radiation model

Each drone carries two independent antennas operating on separate frequency bands (zero cross-interference). The gain pattern follows the **3GPP TR 38.901** single-element model:

$$G(\theta) = G_{\max} - \min\!\left(12\left(\frac{\theta}{\theta_{3\text{dB}}}\right)^{2},\; \text{SLA}\right) \quad [\text{dBi}]$$

where:
- $\theta$ is the off-boresight angle (radians) between the antenna's pointing direction and the direction to the target
- $G_{\max}$ is the peak boresight gain
- $\theta_{3\text{dB}}$ is the full $-3$ dB beamwidth
- $\text{SLA}$ is the sidelobe level attenuation (floor on gain reduction)

The pattern is **azimuthally symmetric** (depends only on the off-boresight angle). The gain drops by exactly 3 dB at $\theta = \theta_{3\text{dB}}/2$ from boresight. The linear gain is $G_{\text{lin}}(\theta) = 10^{G(\theta)/10}$.

| Parameter | Downlink antenna | Backhaul antenna |
|-----------|:----------------:|:----------------:|
| $G_{\max}$ [dBi] | 18.0 | 12.0 |
| $\theta_{3\text{dB}}$ [deg] | 60 | 30 |
| SLA [dB] | 20 | 25 |
| Typical pointing | Nadir (tilt = 0°) | Horizontal (tilt = 90°) |

The **boresight vector** $\mathbf{b}$ of a drone antenna is parameterized by tilt $\phi$ and azimuth $\psi$:

$$\mathbf{b}(\phi, \psi) = \begin{pmatrix} \sin\phi \cos\psi \\ \sin\phi \sin\psi \\ -\cos\phi \end{pmatrix}$$

where tilt $\phi = 0$ points straight down (nadir) and $\phi = \pi/2$ points horizontally. The off-boresight angle to a target at direction $\hat{\mathbf{d}}$ is:

$$\theta = \arccos\!\left(\mathbf{b} \cdot \hat{\mathbf{d}}\right)$$

### Air-to-ground channel model

The propagation model uses the **Al-Hourani / ITU-R** probabilistic LoS/NLoS framework.

**LoS probability** as a function of elevation angle $\theta_{\text{elev}}$:

$$P_{\text{LoS}}(\theta_{\text{elev}}) = \frac{1}{1 + a \cdot \exp\!\left(-b\,(\theta_{\text{elev}}^{\circ} - a)\right)}$$

where $\theta_{\text{elev}}^{\circ}$ is the elevation angle in degrees (0° = horizontal, 90° = directly overhead), and $a$, $b$ are environment-dependent parameters.

**Free-space path loss:**

$$\text{FSPL}(d, f_c) = 20\log_{10}\!\left(\frac{4\pi d f_c}{c}\right) \quad [\text{dB}]$$

where $d$ is the 3D distance [m], $f_c$ is the carrier frequency [Hz], and $c = 3 \times 10^8$ m/s.

**Total air-to-ground path loss** (LoS/NLoS blend):

$$\text{PL}_{\text{A2G}}(d, \theta_{\text{elev}}) = \text{FSPL}(d, f_c) + P_{\text{LoS}} \cdot \eta_{\text{LoS}} + (1 - P_{\text{LoS}}) \cdot \eta_{\text{NLoS}} \quad [\text{dB}]$$

where $\eta_{\text{LoS}}$ and $\eta_{\text{NLoS}}$ are additional attenuation factors for LoS and NLoS links.

**Air-to-air path loss** (drone-to-drone): pure FSPL (no LoS model needed for unobstructed aerial links).

| Environment | $a$ | $b$ | $\eta_{\text{LoS}}$ [dB] | $\eta_{\text{NLoS}}$ [dB] |
|-------------|:---:|:---:|:---:|:---:|
| Suburban | 4.88 | 0.43 | 1.0 | 20.0 |
| Urban | 9.61 | 0.16 | 1.0 | 20.0 |
| Dense Urban | 12.08 | 0.11 | 1.6 | 23.0 |

### Received power and interference

**Downlink power matrix** $\mathbf{P}_{\text{DL}} \in \mathbb{R}^{N \times M}$, where entry $(j, u)$ is the received power at ground user $u$ from drone $j$:

$$P_{\text{DL}}[j, u] = P_{\text{tx}}^{\text{DL}} \cdot G_{\text{DL}}\!\left(\theta_{j \to u}^{\text{off-bore}}\right) \cdot L_{\text{A2G}}(d_{j,u},\, \theta_{j,u}^{\text{elev}})$$

where:
- $P_{\text{tx}}^{\text{DL}}$ is the downlink transmit power (default: 30 dBm = 1 W)
- $G_{\text{DL}}(\cdot)$ is the downlink antenna gain (linear)
- $L_{\text{A2G}}(\cdot) = 10^{-\text{PL}_{\text{A2G}}/10}$ is the path loss as a linear attenuation factor
- $\theta_{j \to u}^{\text{off-bore}}$ is the off-boresight angle from drone $j$'s antenna to user $u$

**Backhaul interference matrix** $\mathbf{I}_{\text{BH}} \in \mathbb{R}^{N \times N}$, where entry $(i, j)$ is the interference power leaked from drone $i$ to drone $j$:

$$I_{\text{BH}}[i, j] = P_{\text{tx}}^{\text{BH}} \cdot G_{\text{BH}}^{\text{Tx}}\!\left(\theta_{i \to j}\right) \cdot G_{\text{BH}}^{\text{Rx}}\!\left(\theta_{j \leftarrow i}\right) \cdot L_{\text{A2A}}(d_{i,j})$$

where $G_{\text{BH}}^{\text{Tx}}$ and $G_{\text{BH}}^{\text{Rx}}$ are the transmit and receive backhaul antenna gains, and $L_{\text{A2A}}$ is the free-space attenuation. The diagonal is zero (no self-interference).

### SINR and throughput

**User-drone association**: each user $u$ is served by the nearest drone (minimum 3D Euclidean distance):

$$\text{serving}(u) = \arg\min_{j \in \{1,\ldots,N\}} \|\mathbf{p}_j - \mathbf{q}_u\|_2$$

**Per-user SINR:**

$$\text{SINR}_u = \frac{P_{\text{DL}}[\text{serving}(u),\, u]}{\displaystyle N_0 + \sum_{j \neq \text{serving}(u)} P_{\text{DL}}[j,\, u]}$$

where the thermal noise power is:

$$N_0 = 10^{(\text{NoisePSD}_{\text{dBm/Hz}} - 30)/10} \times W$$

with $\text{NoisePSD} = -174$ dBm/Hz and bandwidth $W = 10$ MHz (default), giving $N_0 \approx 10^{-11.4}$ W.

**Shannon throughput per user:**

$$R_u = W \cdot \log_2(1 + \text{SINR}_u) \quad [\text{bps}]$$

### Performance metrics

| Metric | Definition |
|--------|------------|
| Coverage fraction | $\frac{1}{M}\sum_{u=1}^{M} \mathbb{1}[\text{SINR}_u \geq \gamma_{\text{reg}}]$ |
| Critical satisfaction | $\frac{1}{M_c}\sum_{u \in \mathcal{C}} \mathbb{1}[\text{SINR}_u \geq \gamma_{\text{crit}}]$ |
| Regular satisfaction | $\frac{1}{M_r}\sum_{u \in \mathcal{R}} \mathbb{1}[\text{SINR}_u \geq \gamma_{\text{reg}}]$ |
| Sum throughput | $\sum_{u=1}^{M} R_u$ |
| 5th percentile SINR | Edge-user performance (worst 5%) |

where $\gamma_{\text{crit}} = 10$ dB and $\gamma_{\text{reg}} = 3$ dB by default.

### Combined score

The heuristic methods are evaluated using a combined score that balances coverage and interference:

$$\text{score} = \zeta \cdot \text{coverage} + (1 - \zeta) \cdot \text{interference\_quality}$$

where $\zeta \in [0, 1]$ controls the tradeoff (default $\zeta = 0.7$) and interference quality is normalized to $[0, 1]$ from total backhaul interference in dBm.

---

## Heuristic deployment methods

Instead of black-box optimization, this framework exploits problem structure through **geometry-aware analytic heuristics** that derive drone altitude, position, and antenna orientations from closed-form expressions. The `dronecomm.heuristic` module implements seven methods:

### K-Means (baseline)

Places drones at K-means centroids of user positions. Altitude is fixed at 150 m, downlink antenna points nadir (straight down), and backhaul antennas point along MST edges. Simple and fast, but ignores inter-drone interference.

### Analytic

Improves on K-means with two physics-based enhancements:

1. **Altitude**: set so the $-3$ dB beam footprint matches the user cluster radius:

$$z_j = \frac{\sigma_j}{\tan(\theta_{3\text{dB}} / 2)}$$

where $\sigma_j$ is the spatial spread (standard deviation) of users assigned to drone $j$, clamped to $[50, 300]$ m.

2. **Downlink orientation**: the boresight vector is pointed from the drone toward its cluster centroid on the ground (closed-form tilt and azimuth).

3. **Backhaul orientation**: each drone points its backhaul antenna toward its parent in the minimum spanning tree (MST) of the drone graph.

This enables **zero-simulation optimal $N^*$ selection** for any tradeoff parameter $\zeta$.

### Analytic + PCA

Extends the analytic method with PCA-based downlink orientation. For each drone's user sub-cluster:

1. Compute the 2D covariance matrix of user positions relative to the drone.
2. Eigen-decompose to find the direction of maximum spread ($\sigma_{\text{major}} = \sqrt{\lambda_1}$).
3. Tilt the DL beam along this direction by $\theta_{\text{tilt}} = \arctan(\sigma_{\text{major}} / z)$, shifting the beam center by one standard deviation along the major axis.
4. Set the azimuth to the major eigenvector direction.

This better handles **elongated or anisotropic clusters** where a nadir beam wastes coverage on empty space while missing users at the cluster edges.

### Repulsive Lloyd's

Modifies Lloyd's (centroidal Voronoi) iteration with an inverse-square repulsive force between drones:

$$\Delta \mathbf{p}_j = (1 - \beta) \cdot (\text{centroid}_j - \mathbf{p}_j) + \beta \cdot \sum_{k \neq j} \frac{\mathbf{p}_j - \mathbf{p}_k}{\|\mathbf{p}_j - \mathbf{p}_k\|^2}$$

The repulsion strength $\beta = 0.15$ pushes drones apart to improve spatial frequency reuse, while the attraction term keeps drones near their user clusters. Converges in $\leq 20$ iterations. Orientations are set analytically after convergence.

### Altitude staggering

Uses graph colouring to assign drones to altitude tiers, reducing beam footprint overlap between neighbouring drones:

1. Build an overlap graph: edge $(j, k)$ exists if $\|\mathbf{p}_j - \mathbf{p}_k\| < r_j + r_k$ (beam footprint radii).
2. Greedy-colour the graph (largest-degree-first) into $n_{\text{tiers}}$ colours.
3. Each colour maps to an altitude band; the analytic altitude formula is applied within the band.

Adjacent drones end up at different altitudes, so their beams overlap minimally.

### Greedy grid (coverage-only)

Greedily places drones one-by-one from a $40 \times 40$ candidate grid ($1{,}600$ positions), choosing at each step the candidate that maximizes **coverage fraction** alone. Fast per step (closed-form SINR), but does not penalize interference.

### Interference-aware greedy

Same greedy framework but scores each candidate by the **combined objective** (coverage + interference quality). Explicitly trades coverage for lower interference at each placement step. Slower than greedy grid (full simulation per candidate), but produces better combined scores at higher drone counts.

### Method summary

| Method | Positions | Altitude | DL orientation | BH orientation |
|--------|-----------|----------|----------------|----------------|
| K-Means | K-means centroids | Fixed 150 m | Nadir (0°) | MST pointing |
| Analytic | K-means centroids | Cluster-spread formula | Centroid geometry | MST pointing |
| Analytic+PCA | K-means centroids | Cluster-spread formula | PCA major axis | MST pointing |
| Repulsive Lloyd's | Lloyd's + repulsion | Cluster-spread formula | Centroid geometry | MST pointing |
| Altitude stagger | K-means centroids | Graph-coloured tiers | Centroid geometry | MST pointing |
| Greedy grid | 40x40 grid search | Cluster-spread formula | Centroid geometry | MST pointing |
| IA Greedy | 40x40 grid search | Cluster-spread formula | Centroid geometry | MST pointing |

---

## Experiments

The experiments of the ICC paper (`paper/main_icc_sweep.tex`) compare K-means, the analytic beam-footprint heuristic and the K-means global altitude sweep. Every number and figure in the paper is regenerated by the pipeline in [Reproducing the paper](#reproducing-the-paper).

### Baseline experiments (`notebooks/experiment.ipynb`)

The original experiment notebook validates the simulation framework:

- **Antenna radiation patterns**: validates 3 dB drop at half-beamwidth and sidelobe floor enforcement.
- **Channel model**: LoS probability curves and path-loss-vs-distance for suburban/urban/dense-urban environments.
- **Single scenario analysis**: 15 K-means drones, 219 users, 2 km $\times$ 2 km urban area. Coverage: 72.6%, critical satisfaction: 25.0%.
- **Placement comparison**: grid placement (86.6% coverage) outperforms K-means (69.0%) and random (68.3%) across 50 Monte Carlo trials.
- **Tilt sensitivity**: coverage optimal at 0°--5° tilt; degrades steeply beyond 20°.
- **Altitude sensitivity**: coverage improves up to 100--150 m then plateaus. FAA Part 107 limit (120 m) is a favorable operating point.

---

## Scenario generation

### Synthetic distributions

**Clustered** (Gaussian mixture):
- $K$ spatial clusters with centers drawn uniformly in the area
- Users within each cluster: $\mathcal{N}(\mu_k, \sigma_{\text{spread}}^2 I)$
- Each user is independently labeled critical with probability $p_{\text{crit}}$

**Uniform**: users placed uniformly at random across the area.

**Hotspot**: fewer, denser clusters ($K = 3$, $\sigma_{\text{spread}} = 50$ m) simulating concentrated survivor groups.

### Real-world distributions

User positions extracted from the **Telecom Italia Milan** dataset (CDR activity on the 235 m Milan grid; the hottest 10 $\times$ 10 cell patch is mapped to a 2 km $\times$ 2 km area). 24 snapshots (4 weeks $\times$ weekday/weekend $\times$ 3 hours) capture varying traffic patterns. See `scripts/extract_telecom_snapshots.py`.

### Drone placement strategies

- **K-means**: run K-means on user positions; place drones above the $N$ cluster centroids
- **Grid**: place drones on a regular $\sqrt{N} \times \sqrt{N}$ grid covering the area
- **Random**: uniform random placement with iterative repulsion to enforce $d_{\min}$
- **Heuristic methods**: see [Heuristic deployment methods](#heuristic-deployment-methods) above

---

## Installation

Requires Python 3.11+.

```bash
cd dronecomm
pip install -e ".[dev]"
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `pyyaml`. Dev: `pytest`, `pytest-cov`.

## Quick start

### Run a Monte Carlo simulation

```bash
# With default parameters (15 drones, urban, K-means placement, 100 trials)
python scripts/run_simulation.py

# Custom config
python scripts/run_simulation.py --config configs/default.yaml --trials 50
```

### Run the heuristic comparison

```bash
# Quick test (3 drone counts, 2 user counts, 2 distributions)
python scripts/run_heuristic_experiment.py --quick --output results/heuristic_quick/

# Full sweep with SLURM array jobs
python scripts/run_heuristic_experiment.py --list-tasks  # see all task IDs
python scripts/run_heuristic_experiment.py --task-id $SLURM_ARRAY_TASK_ID --output results/heuristic_50/
```

### Reproducing the paper

`results/` is not tracked in git: it is large and every file in it is produced by the scripts below. Run them from the repository root, in this order. Each step writes the directory the later steps read.

| Step | Command | Writes |
|---|---|---|
| 1. Synthetic K-means / analytic (HPC) | `sbatch scripts/submit_heuristic_50.sh` | `results/heuristic_50/` |
| 2. Synthetic altitude sweep | `python scripts/run_altitude_sweep_addition.py` | `results/sweep_addition/synthetic_sweep.csv` |
| 3. Per-drone orientation sweep (HPC) | `sbatch scripts/submit_per_drone_angle_sweep.sh` | `results/angle_sweep/run_<jobid>/` |
| 4. Milan snapshots | `python scripts/extract_telecom_snapshots.py` | `results/telecom_milan/snapshots/` |
| 5. Milan telecom experiment (HPC) | `sbatch scripts/submit_telecom_statistical.sh --methods kmeans analytic kmeans_altitude_sweep` | `results/telecom_milan/main/` |
| 6. Figures | `python scripts/plot_sweep_addition_figures.py` | `paper/figures/fig_sweep_*` |
| 7. Check the numbers quoted in the paper | `python scripts/paper_numbers.py` and `python scripts/telecom_numbers.py` | stdout |

The journal extension adds a per-drone greedy altitude heuristic. After steps 1–5:

| Step | Command | Writes |
|---|---|---|
| 8. Synthetic greedy altitude | `python scripts/run_greedy_addition.py` | `results/greedy_addition/synthetic_greedy.csv` |
| 9. Milan greedy altitude | `python scripts/run_telecom_greedy_addition.py` | `results/telecom_milan_greedy/main/` |
| 10. Figures with the greedy series | `python scripts/plot_sweep_addition_figures.py --greedy` | `paper/figures/fig_greedy_*` |

Step 4 needs the raw Telecom Italia Milan files (Harvard Dataverse, [doi:10.7910/DVN/EGZHFV](https://doi.org/10.7910/DVN/EGZHFV)) in `dataverse_files_milan/`, which is also not tracked. Use `--list-tasks` on `run_heuristic_experiment.py`, `run_per_drone_angle_sweep.py` and `run_telecom_statistical.py` to see the SLURM array sizes. `paper_numbers.py` reads the orientation runs from `results/angle_sweep/run_46540284` (K-means, analytic) and `results/angle_sweep/run_local_altsweep` (altitude sweep).

### Use the library directly

```python
from dronecomm.config import Config
from dronecomm.simulation import run_monte_carlo

config = Config()  # or Config.from_yaml("configs/default.yaml")
result = run_monte_carlo(config, verbose=True)

# Access results
for key, stats in result.summary().items():
    print(f"{key}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

# Access individual trial data
trial = result.trial_results[0]
print(trial.sinr_result.sinr_db)       # Per-user SINR [dB]
print(trial.metrics.coverage_fraction)  # Fraction of users above threshold
```

### Deploy with a heuristic

```python
from dronecomm.config import Config
from dronecomm.heuristic import deploy_analytic_heuristic, deploy_repulsive_lloyd_heuristic
from dronecomm.scenario import generate_users

config = Config()
rng = __import__("numpy").random.default_rng(42)
users = generate_users(config.scenario, rng)

# Analytic heuristic
scenario = deploy_analytic_heuristic(config, users, seed=42)

# Repulsive Lloyd's
scenario = deploy_repulsive_lloyd_heuristic(config, users, seed=42)
```

### Visualize a scenario

```python
from dronecomm.antenna import DOWNLINK_ANTENNA
from dronecomm.channel import ChannelModel, URBAN
from dronecomm.scenario import create_scenario
from dronecomm.sinr import nearest_drone_association, dbm_to_watts
from dronecomm.visualization import (
    plot_scenario_3d, plot_sinr_heatmap,
    plot_sinr_cdf, plot_antenna_pattern_polar,
)
import matplotlib.pyplot as plt

scenario = create_scenario(n_drones=15, placement="kmeans", seed=42)

# 3D scatter plot
assoc = nearest_drone_association(scenario.drone_positions, scenario.user_positions)
plot_scenario_3d(scenario.drone_positions, scenario.user_positions,
                 scenario.user_tiers, assoc)
plt.show()

# SINR heat map
plot_sinr_heatmap(
    drone_positions=scenario.drone_positions,
    dl_tilt_rad=scenario.dl_tilt_rad,
    dl_azimuth_rad=scenario.dl_azimuth_rad,
    dl_antenna=DOWNLINK_ANTENNA,
    channel=ChannelModel(env=URBAN),
    p_tx_dl_w=dbm_to_watts(30.0),
    area_size_m=2000.0,
)
plt.show()
```

---

## Configuration

All parameters are configured via YAML or Python dataclasses. See `configs/default.yaml` for the full parameter set.

| Group | Parameter | Default | Description |
|-------|-----------|:-------:|-------------|
| Antenna | $G_{\max}^{\text{DL}}$ | 18.0 dBi | Downlink boresight gain |
| | $\theta_{3\text{dB}}^{\text{DL}}$ | 60° | Downlink beamwidth |
| | $G_{\max}^{\text{BH}}$ | 12.0 dBi | Backhaul boresight gain |
| | $\theta_{3\text{dB}}^{\text{BH}}$ | 30° | Backhaul beamwidth |
| Channel | Environment | Urban | Suburban / Urban / Dense Urban |
| | $f_c$ | 2.0 GHz | Carrier frequency |
| Network | $N$ | 15 | Number of drones |
| | Altitude | 150 m | Drone flight altitude |
| | $P_{\text{tx}}^{\text{DL}}$ | 30 dBm (1 W) | Downlink transmit power |
| | $P_{\text{tx}}^{\text{BH}}$ | 27 dBm (0.5 W) | Backhaul transmit power |
| | $W$ | 10 MHz | Bandwidth per user |
| | NoisePSD | $-174$ dBm/Hz | Thermal noise (~290 K) |
| | $\gamma_{\text{crit}}$ | 10 dB | Critical user SINR threshold |
| | $\gamma_{\text{reg}}$ | 3 dB | Regular user SINR threshold |
| | $d_{\min}$ | 100 m | Minimum inter-drone separation |
| Scenario | Area | 2000 m $\times$ 2000 m | Disaster area size |
| | Clusters | 5 | Number of user hotspots |
| | Users/cluster | $40 \pm 10$ | Gaussian cluster size |
| | $\sigma_{\text{spread}}$ | 100 m | Spatial spread within cluster |
| | $p_{\text{crit}}$ | 0.15 | Fraction of critical users |
| Simulation | Trials | 100 | Monte Carlo repetitions |

---

## Project structure

```
dronecomm/
├── pyproject.toml
├── configs/
│   └── default.yaml               # Default scenario parameters
├── scripts/
│   ├── run_simulation.py           # CLI: Monte Carlo simulation
│   ├── run_heuristic_experiment.py # CLI: synthetic heuristic comparison
│   ├── run_altitude_sweep_addition.py  # CLI: altitude sweep on the synthetic grid
│   ├── run_per_drone_angle_sweep.py    # CLI: per-drone DL orientation sweep
│   ├── extract_telecom_users.py    # Telecom Italia grid regions and user sampling
│   ├── extract_telecom_snapshots.py    # Extract the 24 Milan snapshots
│   ├── run_telecom_statistical.py  # CLI: telecom experiment on the snapshots
│   ├── plot_sweep_addition_figures.py  # Paper figures
│   ├── paper_numbers.py            # Check synthetic/orientation numbers in the paper
│   ├── telecom_numbers.py          # Check telecom numbers in the paper
│   └── submit_*.sh                 # SLURM submission scripts
├── notebooks/
│   ├── analyze_per_drone_angle_sweep.ipynb  # Orientation sweep analysis
│   ├── experiment.ipynb            # Baseline experiments (antenna, channel, placement)
│   ├── optimization.ipynb          # Gradient-free joint optimizers
│   └── validation_aerpaw.ipynb     # AERPAW channel model validation
├── paper/
│   ├── main_icc_sweep.tex          # ICC paper
│   ├── figures/                    # Figures used by the paper
│   └── archive/                    # Earlier revisions of the ICC paper
├── data/
│   └── aerpaw/                     # AERPAW channel measurement references
├── dronecomm/                      # Main package
│   ├── geometry.py                 # 3D vector math, angle computations
│   ├── antenna.py                  # Parametric antenna radiation model
│   ├── channel.py                  # Air-to-ground path loss (LoS/NLoS)
│   ├── interference.py             # Power matrix and backhaul interference
│   ├── sinr.py                     # SINR, throughput, user-drone association
│   ├── scenario.py                 # User generation, drone placement
│   ├── heuristic.py                # Geometry-aware analytic heuristics (7 methods)
│   ├── metrics.py                  # Coverage, tier satisfaction, statistics
│   ├── config.py                   # Dataclass config with YAML support
│   ├── simulation.py               # Monte Carlo runner
│   └── visualization.py            # Plots: 3D, heat maps, CDFs, PCA overlays
└── tests/
    ├── test_geometry.py
    ├── test_antenna.py
    ├── test_channel.py
    ├── test_interference.py
    └── test_sinr.py
```

## Data flow

```
Config
  │
  ▼
Scenario generation ──► user positions (M, 3) + tiers (M,)
  │                      drone positions (N, 3) + orientations
  │
  ├─► Baseline placement (K-means / grid / random)
  │
  └─► Heuristic deployment ──► analytic altitude + orientation
      │                         PCA tilt, repulsive Lloyd's,
      │                         altitude staggering, greedy grid
      ▼
Geometry ──────────────► off-boresight angles (N, M), distances, elevations
  │
  ▼
Antenna gain ──────────► gain matrix (N, M) [linear]
  │
  ▼
Channel path loss ─────► attenuation matrix (N, M) [linear]
  │
  ▼
Interference ──────────► power matrix (N, M) = P_tx × gain × attenuation
  │
  ▼
SINR ──────────────────► per-user SINR (M,), throughput (M,)
  │
  ▼
Metrics ───────────────► coverage, tier satisfaction, combined score
  │
  ▼
Visualization ─────────► 3D plots, SINR heat maps, CDFs, PCA overlays
```

## Running tests

```bash
cd dronecomm
python -m pytest tests/ -v
```

38 tests covering geometry, antenna physics, channel model, interference, and SINR computation. Key physics validations include:

- Antenna gain drops exactly 3 dB at the half-beamwidth point
- Tilting an interfering drone toward a victim user increases interference
- Symmetric drone placements produce symmetric SINR
- LoS probability increases with elevation angle
- Single-drone SINR is noise-limited (~60+ dB with no interferers)

## Extending the framework

### Swap the antenna model

Implement the `AntennaModel` protocol from `antenna.py`:

```python
class MyAntenna:
    def gain_dbi(self, theta: np.ndarray) -> np.ndarray: ...
    def gain_linear(self, theta: np.ndarray) -> np.ndarray: ...
    @property
    def g_max_dbi(self) -> float: ...
    @property
    def beamwidth_rad(self) -> float: ...
```

### Add a new heuristic

Add a `deploy_<name>_heuristic()` function in `heuristic.py` following the existing pattern:

```python
def deploy_my_heuristic(config: Config, user_positions: np.ndarray, seed: int = 42) -> Scenario:
    """My custom deployment heuristic."""
    # 1. Compute drone positions from user_positions
    # 2. Set altitudes (e.g., altitude_from_cluster_spread)
    # 3. Set DL orientations (e.g., dl_orientation_from_centroid or pca_dl_orientations)
    # 4. Set BH orientations (e.g., mst_backhaul_orientations)
    # 5. Return Scenario
    ...
```

Then register it in `run_heuristic_experiment.py` to include it in the comparison sweep.

### Add a new placement strategy

Add a `deploy_drones_<name>()` function in `scenario.py` following the existing signature, and register it in `create_scenario()`.

---

## Real-world validation

### AERPAW channel measurements

The framework includes validation against real air-to-ground channel measurements from the **AERPAW** testbed (NSF PAWR platform). See `notebooks/validation_aerpaw.ipynb` for:

- Loading SigMF-format channel sounding data
- Fitting environment parameters to measured LoS/NLoS statistics
- Comparing the Al-Hourani model predictions against real measurements
- Frequency adjustment between measurement band (3.5 GHz CBRS) and simulation band (2.0 GHz)

Data references and download instructions are in `data/aerpaw/README.md`.

### Telecom Italia Milan dataset

Real-world user distributions derived from CDR activity data in Milan (Harvard Dataverse, doi:10.7910/DVN/EGZHFV). 24 snapshots are used to evaluate heuristic robustness across traffic conditions. See `scripts/extract_telecom_snapshots.py` for the extraction pipeline and `scripts/telecom_numbers.py` for the analysis.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{dronecomm2025,
  title={Geometry-Aware Optimal Drone Deployment for Disaster Emergency Communications},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2025},
  note={Code available at https://github.com/[repository]}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

