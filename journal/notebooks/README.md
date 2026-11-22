# Notebook Results

This document explains the results produced by the two Jupyter notebooks in this directory. Both notebooks use the `dronecomm` simulation framework with 15 UAV base stations deployed over a 2 km x 2 km urban disaster area serving ~220 ground users (15% critical, 85% regular).

---

## Experiment Notebook (`experiment.ipynb`)

This notebook validates the physics models and explores the parameter space to motivate the need for geometry-aware optimization. It runs seven experiments.

### 1. Antenna Radiation Patterns

The 3GPP TR 38.901 parametric antenna model is visualized for both antenna types:

| Antenna | Peak Gain | Beamwidth | Sidelobe Floor |
|---------|:---------:|:---------:|:--------------:|
| Downlink | 18 dBi | 60° | -2 dBi (SLA = 20 dB) |
| Backhaul | 12 dBi | 30° | -13 dBi (SLA = 25 dB) |

The downlink antenna has a wide beam to cover ground users beneath each drone, while the backhaul antenna has a narrow beam for point-to-point drone-to-drone links. The polar and Cartesian plots confirm that the -3 dB point occurs at exactly half the stated beamwidth, and gain floors at `G_max - SLA` for all off-axis angles beyond the main lobe.

### 2. Channel Model Validation

Two sets of curves validate the Al-Hourani air-to-ground channel:

- **LoS probability vs elevation angle**: Suburban environments reach ~95% LoS at 30° elevation, while dense urban requires nearly 70° for the same probability. This means low-altitude drones in cities suffer severe NLoS penalties.
- **Path loss vs horizontal distance** (at 150 m altitude): Urban path loss is 5-10 dB worse than free-space at short distances (where NLoS dominates), but all environments converge toward FSPL at large distances where the elevation angle is small and LoS probability approaches its asymptote.

### 3. Single Scenario Analysis

A single K-means deployment with nadir-pointing antennas at 150 m altitude produces:

| Metric | Value |
|--------|------:|
| Mean SINR | 6.1 dB |
| 5th percentile SINR | -0.1 dB |
| Overall coverage (>= 3 dB) | 72.6% |
| Critical users satisfied (>= 10 dB) | **25.0%** |
| Sum throughput | 5,393 Mbps |
| Backhaul interference (total) | -49.5 dBm |

**Key observation**: Only 25% of critical users meet the 10 dB SINR threshold. The SINR heat map shows tight high-SINR spots directly beneath each drone with rapid rolloff --- users between drones receive poor service. The backhaul interference is low (-49.5 dBm total) because nadir-pointing downlink antennas direct minimal energy toward other drones horizontally.

### 4. Placement Strategy Comparison

Three baseline strategies are compared over 50 Monte Carlo trials:

| Strategy | Coverage | Critical Satisfied | Mean SINR | Throughput |
|----------|:--------:|:------------------:|:---------:|:----------:|
| K-means | 69.0% +/- 7.7% | 26.5% | 7.5 dB | 5,859 Mbps |
| **Grid** | **86.6% +/- 6.2%** | **69.3%** | **16.4 dB** | **11,239 Mbps** |
| Random | 68.3% +/- 13.2% | 41.4% | 9.6 dB | 7,346 Mbps |

**Why grid wins**: K-means clusters drones near user hotspots, which seems intuitive but causes two problems: (1) drones are packed close together, increasing mutual interference, and (2) large areas between clusters are left uncovered. Grid placement spaces drones uniformly, avoiding interference concentration and providing more even coverage. Random placement has high variance (+/- 13.2%) since some realizations happen to produce good spacing while others don't.

**Why K-means still matters**: Despite grid's superior average performance, K-means is the appropriate initialization for optimization because it starts drones near users --- the optimizer then adjusts positions and orientations to reduce interference while maintaining proximity to demand.

### 5. Tilt Sensitivity Analysis

Downlink antenna tilt is swept from 0° (nadir) to 45° with K-means positions fixed:

- **0-5° tilt**: Optimal zone. Coverage peaks at 5° with a slight improvement over pure nadir because a small tilt redistributes sidelobe energy to users at moderate horizontal distances.
- **10-20° tilt**: Rapid degradation. The main beam starts missing directly-below users while not yet reaching far-away users effectively.
- **>25° tilt**: Severe performance collapse. The main beam points nearly horizontal, and most users are in sidelobes or backlobes.

**This is the core thesis insight**: Antenna orientation has a dramatic, nonlinear impact on network performance. A 15° tilt difference can halve coverage. This motivates jointly optimizing orientation alongside position rather than using a fixed nadir-pointing assumption.

### 6. Altitude Sensitivity Analysis

Altitude is swept from 50 m to 320 m:

- **50 m**: Highest raw SINR for users directly below (short path), but poor LoS probability and tiny coverage footprint per drone. Coverage is ~65%.
- **100-150 m**: Sweet spot. Good balance of LoS probability, path loss, and coverage radius. Coverage plateaus at ~75-80%.
- **200+ m**: Diminishing returns. LoS improves but the extra path loss and broader interference footprints offset the gains. Critical satisfaction reaches ~50% at 200 m.

The FAA Part 107 limit (120 m) sits at a favorable operating point. The optimization notebook uses this altitude as the default.

### 7. Summary

| Finding | Best Value |
|---------|-----------|
| Best placement strategy | Grid (86.6% coverage) |
| Best altitude for coverage | 50 m (dominates at short range) |
| Best tilt for coverage | 5° (slightly off-nadir) |

All three parameters --- placement, altitude, and tilt --- significantly affect performance, confirming that they should be optimized jointly rather than set independently.

---

## Optimization Notebook (`optimization.ipynb`)

This notebook runs three gradient-free optimizers to jointly optimize drone positions (x, y) and antenna orientations (downlink and backhaul tilt and azimuth) --- a 90-dimensional problem for 15 drones.

### Setup

- **Baseline**: K-means placement at 120 m altitude with nadir-pointing antennas
- **Objective**: Maximize coverage and critical-user satisfaction with penalties for critical SINR violations (lambda = 50) and minimum separation violations (lambda = 5)
- **Decision variables**: 6 per drone (x, y, DL tilt, DL azimuth, BH tilt, BH azimuth) = 90 total
- **Evaluation**: Each candidate is evaluated using the full simulation pipeline (power matrix, SINR, metrics). Final results are averaged over 10 Monte Carlo user realizations.

### Optimizer Results

| Method | Coverage | Critical Satisfied | Throughput | Mean SINR | 5th %ile | Evaluations | Time |
|--------|:--------:|:------------------:|:----------:|:---------:|:--------:|:-----------:|:----:|
| **Baseline** | 33.6% | 17.2% | 3,289 Mbps | 1.0 dB | -6.5 dB | 1 | 0 s |
| DE | 61.3% | 42.9% | 6,832 Mbps | 7.0 dB | -9.9 dB | 271,350 | 207 s |
| Alternating | **68.1%** | 46.1% | **7,825 Mbps** | **9.3 dB** | -8.8 dB | 221,102 | 165 s |
| PSO | 65.5% | **46.3%** | 7,576 Mbps | 8.5 dB | **-7.6 dB** | 6,030 | 5 s |


### Analysis of Each Method

**Baseline (K-means, nadir-pointing)**

The poor baseline (33.6% coverage, 17.2% critical satisfaction) at 120 m altitude with nadir-pointing antennas is expected. At this altitude with a 60° beamwidth, each drone's -3 dB footprint has a radius of only ~69 m. Fifteen such circles cover roughly 225,000 m^2 out of a 4,000,000 m^2 area (5.6%). Coverage above 33% comes entirely from sidelobe energy and noise-limited regions where even weak signal exceeds the 3 dB SINR threshold.

**Differential Evolution**

DE achieves 61.3% coverage but is the slowest optimizer (271K evaluations, 207 s). It searches the full 90D space simultaneously, which is thorough but computationally expensive. The 5th-percentile SINR of -9.9 dB is the worst among optimizers, suggesting DE gets trapped in local minima that help average users at the expense of edge users.

**Alternating Optimization (Best Coverage)**

Alternating achieves the best coverage (68.1%) and throughput (7,825 Mbps) by decomposing the problem: first optimize 60 orientation variables with positions fixed, then optimize 30 position variables with orientations fixed, and repeat. This is effective because:
1. The orientation subproblem (60D) finds good antenna tilts/azimuths quickly since each drone's orientation primarily affects its own users.
2. The position subproblem (30D) is much smaller than the joint problem.
3. The alternating structure naturally handles the coupling: positions determine which users each drone serves, orientations determine how well they're served.

**PSO (Best Efficiency)**

PSO is dramatically faster (6,030 evaluations, 4.5 s vs 165-207 s) while achieving competitive results (65.5% coverage, 46.3% critical satisfaction). Its 5th-percentile SINR (-7.6 dB) is the best among optimizers, meaning it finds solutions that are more balanced across users rather than optimizing for the average. However, with the original 30 particles for a 90D problem, it was under-exploring the search space.

### Convergence Behavior

The convergence plot shows:
- **DE**: Steady improvement over ~270K evaluations, still improving at termination (not converged).
- **Alternating**: Step-wise improvement with clear plateaus between outer iterations. Converges faster than DE in wall-clock time.
- **PSO**: Rapid initial improvement (first ~1000 evaluations), then diminishing returns. Most gains come early due to the swarm's parallel exploration.

### What the Optimizer Changes

The drone position shift plot and orientation bar charts reveal:
- **Positions**: Drones shift 50-300 m from K-means centroids, generally moving apart to reduce mutual interference and toward uncovered areas.
- **DL tilt**: The optimizer tilts antennas 5-25° from nadir for most drones, consistent with the tilt sensitivity experiment's finding that small tilts improve coverage.
- **DL azimuth**: Nonzero azimuths are assigned, pointing antennas toward user clusters that are laterally offset from the drone position.
- **BH orientation**: Backhaul antennas are reoriented from the default horizontal to point toward neighboring drones in the mesh.

### Remaining Gaps

Even after optimization, critical satisfaction reaches only ~46% (vs the 100% target). This indicates a fundamental capacity limitation: 15 drones at 120 m with these antenna parameters and 10 dB SINR threshold cannot serve all critical users. Possible remedies:
- Increase the number of drones
- Allow altitude optimization (the code now supports this)
- Use higher transmit power
- Relax the critical SINR threshold
- Use NOMA or other multiple access schemes (outside current model scope)

---

## Code Improvements Applied

After reviewing the optimization results, several improvements were made to `dronecomm/optimize.py`:

### PSO Fixes
1. **Velocity clamping** (bug fix): Velocities are now clamped to `+/- 0.2 * range` per dimension. Without this, particles overshoot bounds, get clipped, and lose their velocity information --- effectively getting stuck at boundaries.
2. **Linear inertia decay**: Inertia weight decays from 0.9 (exploration) to 0.4 (exploitation) over the iteration budget. A fixed inertia of 0.7 was neither exploring well initially nor exploiting well finally.
3. **Default particles increased**: From 30 to 50 (notebook uses 90 to match the problem dimension).

### Objective Function
4. **Critical weight 2 -> 5**: The objective function now weights critical satisfaction 5x more than coverage, strongly steering the optimizer toward satisfying the SINR >= 10 dB constraint.
5. **Critical penalty lambda 10 -> 50**: The penalty for violating critical-user SINR is 5x larger. With lambda = 10, the optimizer could achieve a lower overall cost by accepting critical violations; with lambda = 50, this is no longer cost-effective.

### Alternating Optimization
6. **Full-objective convergence check**: Previously compared costs from orientation-only and position-only subproblems. Now evaluates the full combined objective at each outer iteration for an apples-to-apples comparison.
7. **Altitude optimization support**: New `optimize_altitude=True` flag includes altitude (z) in the position phase, expanding the search from 2N to 3N position variables.

### `run_comparison` API
8. **Passthrough parameters**: `weights`, `penalty_lambdas`, `pso_n_particles`, and `optimize_altitude` can now be passed directly to `run_comparison` without editing source code, enabling rapid hyperparameter experimentation in the notebook.
