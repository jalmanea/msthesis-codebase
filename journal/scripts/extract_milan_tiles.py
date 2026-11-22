"""Milan demand across the city: activity-stratified tiles with census populations.

The earlier snapshots crop the hottest 2.35 km patch at each (date, hour). This script
instead splits the 100 x 100 Telecom Italia grid into 100 non-overlapping 10 x 10
tiles (2.35 km x 2.35 km), ranks them by mean activity over the paper's 24
(date, hour) snapshots, and keeps the tiles at the activity percentiles in
PERCENTILES (100 = hottest). For each kept tile it

  * sums the 2021 census residents of the ISTAT census sections it overlaps,
    weighting each section by its overlapping area share (docs/parameters.md D10);
  * samples 800 user points per snapshot, weighted by that snapshot's activity
    and rescaled to the 2 km square, as extract_telecom_snapshots.py does.

Writes results/telecom_milan/tiles/tile<P>_snap<SS>.npy and tiles_metadata.json.
Needs the raw files in dataverse_files_milan/ (Telecom Italia, doi:10.7910/DVN/EGZHFV)
and external/istat2021/sections/R03_21/SHP/R03_21_WGS84.* (ISTAT basi territoriali
2021, Lombardy, CC BY 4.0: https://www.istat.it/storage/cartografia/basi_territoriali/2021/R03_21.zip).

python scripts/extract_milan_tiles.py --data-dir ../dronecomm/dataverse_files_milan
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import shapefile
import shapely
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from extract_telecom_snapshots import SNAPSHOT_DESIGN  # noqa: E402
from extract_telecom_users import (  # noqa: E402
    COL_CALL_IN, COL_CALL_OUT, COL_INTERNET, COL_SMS_IN, COL_SMS_OUT, COL_SQUARE, COL_TIME_MS,
    sample_users, sq_id_to_xy,
)

GRID = 100
TILE = 10
CELL_M = 235.0
N_USERS = 800
AREA_M = 2000.0
PERCENTILES = [100, 90, 75, 50, 25]
SECTIONS = ROOT / "external" / "istat2021" / "sections" / "R03_21" / "SHP" / "R03_21_WGS84"
OUT = ROOT / "results" / "telecom_milan" / "tiles"


def load_day(path: Path, hours: list[int]) -> dict[int, np.ndarray]:
    """{hour: GRID x GRID activity} for the given UTC hours, reading the file once."""
    grids = {h: np.zeros((GRID, GRID)) for h in hours}
    cols = (COL_SMS_IN, COL_SMS_OUT, COL_CALL_IN, COL_CALL_OUT, COL_INTERNET)
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            h = (int(parts[COL_TIME_MS]) % 86_400_000) // 3_600_000
            if h not in grids:
                continue
            r, c = sq_id_to_xy(int(parts[COL_SQUARE]), GRID)
            grids[h][r, c] += sum(float(parts[k]) for k in cols if k < len(parts) and parts[k].strip())
    return grids


def tile_polygons(grid_geojson: Path) -> dict[tuple[int, int], object]:
    """{(r0, c0): tile polygon in WGS84} from the official grid squares."""
    feats = json.loads(grid_geojson.read_text())["features"]
    cells: dict[tuple[int, int], list] = {}
    for f in feats:
        r, c = sq_id_to_xy(int(f["properties"]["cellId"]), GRID)
        cells.setdefault((r - r % TILE, c - c % TILE), []).append(shape(f["geometry"]))
    return {k: unary_union(v) for k, v in cells.items()}


def census_population(tiles: dict, keys: list) -> dict:
    """Area-weighted 2021 residents of each tile, plus its area [km2], in UTM 32N."""
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    utm = {k: shapely.transform(tiles[k], lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
           for k in keys}
    bbox = unary_union(list(utm.values())).bounds
    reader = shapefile.Reader(str(SECTIONS), encoding="utf-8")
    names = [f[0] for f in reader.fields[1:]]
    i_pop = names.index("POP21")
    geoms, pops = [], []
    for sr in reader.iterShapeRecords():
        x0, y0, x1, y1 = sr.shape.bbox
        if x1 < bbox[0] or x0 > bbox[2] or y1 < bbox[1] or y0 > bbox[3]:
            continue
        g = shape(sr.shape.__geo_interface__)
        if not g.is_valid:
            g = g.buffer(0)
        geoms.append(g)
        pops.append(float(sr.record[i_pop] or 0))
    tree = STRtree(geoms)
    out = {}
    for k, poly in utm.items():
        pop = 0.0
        for i in tree.query(poly):
            g = geoms[i]
            if g.area > 0:
                pop += pops[i] * g.intersection(poly).area / g.area
        out[k] = {"census_pop": pop, "area_km2": poly.area / 1e6}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "dataverse_files_milan")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    by_date: dict[str, list[int]] = {}
    for date, hour, _, _ in SNAPSHOT_DESIGN:
        by_date.setdefault(date, []).append(hour)
    act = {}
    for date, hours in by_date.items():
        print(f"  reading {date} (hours {hours})", flush=True)
        for h, g in load_day(args.data_dir / f"sms-call-internet-mi-{date}.txt", hours).items():
            act[(date, h)] = g
    snaps = [(i, d, h, w, t) for i, (d, h, w, t) in enumerate(SNAPSHOT_DESIGN)]
    stack = np.stack([act[(d, h)] for _, d, h, _, _ in snaps])                  # (24, 100, 100)
    tile_act = stack.reshape(len(snaps), GRID // TILE, TILE, GRID // TILE, TILE).sum(axis=(2, 4))
    mean_act = tile_act.mean(axis=0)                                            # (10, 10)
    order = np.argsort(mean_act.ravel(), kind="stable")                         # ascending
    chosen = []
    for p in PERCENTILES:
        idx = order[int(round(p / 100 * (order.size - 1)))]
        chosen.append((p, int(idx // (GRID // TILE)) * TILE, int(idx % (GRID // TILE)) * TILE))

    tiles = tile_polygons(args.data_dir / "milano-grid.geojson")
    census = census_population(tiles, [(r0, c0) for _, r0, c0 in chosen])

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"tile_cells": TILE, "cell_size_m": CELL_M, "area_size_m": AREA_M, "n_users": N_USERS,
            "seed": args.seed, "percentiles": PERCENTILES,
            "census_source": "ISTAT basi territoriali 2021, Lombardy (R03_21), POP21, area-weighted",
            "tiles": [], "snapshots": []}
    rng = np.random.default_rng(args.seed)
    for p, r0, c0 in chosen:
        c = census[(r0, c0)]
        meta["tiles"].append({"percentile": p, "row0": r0, "col0": c0,
                              "mean_activity": float(mean_act[r0 // TILE, c0 // TILE]),
                              "census_pop": round(c["census_pop"], 1), "area_km2": round(c["area_km2"], 3),
                              "density_per_km2": round(c["census_pop"] / c["area_km2"], 1)})
        print(f"  tile p{p:3d} at ({r0},{c0}): {c['census_pop']:9.0f} residents on {c['area_km2']:.2f} km2 "
              f"({c['census_pop'] / c['area_km2']:7.0f}/km2)", flush=True)
        for i, d, h, w, t in snaps:
            a = {sq: v for sq, v in ((rr * GRID + cc + 1, act[(d, h)][rr, cc]) for rr in range(r0, r0 + TILE)
                                     for cc in range(c0, c0 + TILE)) if v > 0}
            users = sample_users(a, N_USERS, r0, c0, TILE, AREA_M, rng, GRID, CELL_M)
            np.save(OUT / f"tile{p:03d}_snap{i:02d}.npy", users)
    meta["snapshots"] = [{"idx": i, "date": d, "hour": h, "week": w, "day_type": t} for i, d, h, w, t in snaps]
    (OUT / "tiles_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"-> {OUT} ({len(chosen)} tiles x {len(snaps)} snapshots)")


if __name__ == "__main__":
    main()
