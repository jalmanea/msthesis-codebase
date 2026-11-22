# msthesis-codebase

Simulation code for the M.S. thesis *Deployment and Self-Organization of UAV Base
Station Fleets for Post-Disaster Services* (KAUST, 2026).

| Folder | Thesis chapters | Contents |
| --- | --- | --- |
| `journal/` | 3, 4, 5 and 7 | Grounded model, fleet sizing, joint search and backhaul delivery |
| `swarm/` | 6 | Decentralized play, radius sweeps, self-sizing and the language-model runs |
| `thesis/` | 1 and 4 | The no-interference counterfactual, which is in the thesis only |

`journal/` and `swarm/` are two versions of the same Python package, `dronecomm`. Each
folder is self-contained, with its own configurations, scripts and tests, and its own
`README.md` describing the package.

## Setup

Python 3.11 or later. Install one folder at a time, each in its own environment:

```bash
cd journal            # or: cd swarm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -q
```

`thesis/noint_counterfactual.py` imports the code in `journal/` and runs in that
environment.

## Data

The data are not redistributed here. The Milan demand comes from the Telecom Italia
dataset on the Harvard Dataverse (doi:10.7910/DVN/EGZHFV), unpacked into a
`dataverse_files*/` folder inside `journal/` or `swarm/`. The residents of each tile
come from the ISTAT 2021 census sections, which the scripts download into `external/`.

## License

MIT, see `LICENSE`.
