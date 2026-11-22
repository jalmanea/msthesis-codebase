"""One LLM per drone (dronecomm.llmagent): small open-weight Chinese models (at most ~10B parameters, so that a
4-bit copy fits the 8-16 GB of a drone's companion computer), served by vLLM.

A unit is (model, layout, framing, repeat). The fleet starts at the paper's deployment of the layout
(results/swarm/central/*.npz, from run_swarm.py --stage central) and plays ROUNDS simultaneous rounds
with inertia 0.5; every agent sees INFO_M around itself. Stage l1 has a fixed fleet (no land/call);
l2 adds land/call with a reserve of 30 % of N. Stage l1base is the matching algorithmic baseline
(dronecomm.swarm.run, simultaneous, inertia 0.5, info_m = INFO_M, selfish and global utilities), run
on CPU. Stage l1pol runs the scripted policies of dronecomm.llmagent.Policy (stay, goto, rand6, groups6)
through the same tools, budget and dynamics as l1, on CPU: the baselines that separate what the model
contributes from what the what_if tool does. Stage smoke is one layout, one framing, two rounds.

Stages l1log and l1gate repeat l1 for the area framing only (the l1 code logs every tool result, the
model's text and each drone's realised change; l1 runs before 2026-10-31 did not). l1log is the plain
area framing, so its moves can be checked against the what_if answers the model was given; l1gate adds
llmagent.GATE_TEXT (move only on a predicted gain), a diagnostic of the decision rule.

The job serves the model itself (`--serve`: vllm serve on localhost with the model card's tool-call and
reasoning parsers; inside the Docker image $VLLM_IMAGE when that is set, e.g. on an NVIDIA Brev instance),
waits for it, runs the model's units of the stage and stops it. A key ending in -think runs the same weights
with thinking on (sampling as the card recommends for thinking mode); the others run with thinking off.

python scripts/run_llm.py --list
VLLM_IMAGE=vllm/vllm-openai:v0.31.0 python scripts/run_llm.py --model qwen3.5-4b --stage smoke --serve
VLLM_IMAGE=vllm/vllm-openai:v0.31.0 python scripts/run_llm.py --model qwen3.5-4b --stage l1 --serve
python scripts/run_llm.py --stage l1base --procs 16                          # CPU
python scripts/run_llm.py --stage l1pol --procs 16                           # CPU
python scripts/run_llm.py --model qwen3.5-4b --stage l1 --check
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dronecomm import joint, llmagent, swarm  # noqa: E402
from run_joint import _write  # noqa: E402
from run_swarm import BASE_SEED, DEPOT_XY, _problem, central_path  # noqa: E402

OUT = ROOT / "results" / "llm"


@dataclass(frozen=True)
class Model:
    hf: str                        # Hugging Face id
    tool_parser: str               # vLLM --tool-call-parser
    reasoning_parser: str | None   # vLLM --reasoning-parser
    thinking: bool
    sampling: dict                 # the model card's recommended sampling for this mode
    serve_args: tuple = ()
    served: str = ""               # served model name (several keys can share one server); default the key

    def endpoint(self, base_url: str, key: str) -> llmagent.Endpoint:
        smp = dict(self.sampling)
        return llmagent.Endpoint(base_url, self.served or key, temperature=smp.pop("temperature"),
                                 extra={**smp, "chat_template_kwargs": {"enable_thinking": self.thinking}})


# Qwen3.5 card (Qwen/Qwen3.5-4B, "vllm serve ... --tool-call-parser qwen3_coder"; recommended sampling for
# instruct (non-thinking) and thinking mode, general tasks); --language-model-only skips the vision encoder.
QWEN_OFF = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}
QWEN_ON = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}
QWEN_ARGS = ("--language-model-only",)
MODELS = {
    **{f"qwen3.5-{s}": Model(f"Qwen/Qwen3.5-{s.upper()}", "qwen3_coder", "qwen3", False, QWEN_OFF, QWEN_ARGS)
       for s in ("0.8b", "2b", "4b", "9b")},
    **{f"qwen3.5-{s}-think": Model(f"Qwen/Qwen3.5-{s.upper()}", "qwen3_coder", "qwen3", True, QWEN_ON, QWEN_ARGS,
                                   served=f"qwen3.5-{s}")
       for s in ("4b", "9b")},
}
LAYOUTS = [("urban_broad", "voice", None, BASE_SEED, 11), ("urban_broad", "voice", None, BASE_SEED + 1, 11),
           ("milan_p100", "voice", 0, BASE_SEED, 14)]
FRAMINGS = list(llmagent.FRAMINGS)
REPEATS = 2
ROUNDS = 8
INFO_M = 600.0
INERTIA = 0.5
RESERVE_SHARE = 0.3
PORT = 8011
STAGES = ["smoke", "l1", "l2", "l1log", "l1gate", "l1base", "l1pol"]
STAGE_FRAMINGS = {"smoke": ["network"], "l1log": ["area"], "l1gate": ["area" + llmagent.GATED]}
CPU_STAGES = ("l1base", "l1pol")
COLUMNS = ["stage", "model", "hf_id", "server", "thinking", "scenario", "tier", "snapshot", "seed", "n_drones", "framing", "repeat", "info_m",
           "rounds", "served_start_pct", "served_pct", "n_active_end", "best_round_pct", "turns", "tool_calls",
           "what_ifs", "prompt_tokens", "completion_tokens", "model_seconds", "errors", "moves", "stays", "lands",
           "calls", "applied", "time_s", "history"]


def _lay(u: dict) -> dict:
    return {"scenario": u["scenario"], "tier": u["tier"], "snapshot": u["snapshot"], "seed": u["seed"], "n": u["n"]}


def units(stage: str, model: str | None) -> list[dict]:
    if stage == "l1base":
        return [{"stage": stage, "model": f"br-{k}", "scenario": sc, "tier": t, "snapshot": snap, "seed": seed,
                 "n": n, "framing": k, "repeat": r}
                for sc, t, snap, seed, n in LAYOUTS for k in ("selfish", "global") for r in range(REPEATS)]
    if stage == "l1pol":       # framing matters only to the policies that rank what_if results
        return [{"stage": stage, "model": f"pol-{k}", "scenario": sc, "tier": t, "snapshot": snap, "seed": seed,
                 "n": n, "framing": f, "repeat": r}
                for sc, t, snap, seed, n in LAYOUTS for k in llmagent.POLICIES
                for f in (("selfish", "network") if k in ("rand6", "groups6") else ("network",))
                for r in range(REPEATS)]
    lays = LAYOUTS[:1] if stage == "smoke" else LAYOUTS
    frs = STAGE_FRAMINGS.get(stage, FRAMINGS)
    reps = 1 if stage == "smoke" else REPEATS
    return [{"stage": stage, "model": model, "scenario": sc, "tier": t, "snapshot": snap, "seed": seed, "n": n,
             "framing": f, "repeat": r} for sc, t, snap, seed, n in lays for f in frs for r in range(reps)]


def unit_path(u: dict) -> Path:
    s = f"seed_{u['seed']}" if u["snapshot"] is None else f"snap{u['snapshot']:02d}_seed_{u['seed']}"
    return OUT / u["stage"] / u["model"] / f"{u['scenario']}_{u['tier']}_{s}_n{u['n']:03d}_{u['framing']}_r{u['repeat']}.csv"


def start_world(u: dict, fleet: bool) -> tuple[llmagent.World, float]:
    cfg, users, phys = _problem(_lay(u))
    z = np.load(central_path(_lay(u)).with_suffix(".npz"))
    pos, t = z["paper_pos"], z["paper_t"]
    n_res = int(np.ceil(RESERVE_SHARE * u["n"])) if fleet else 0
    if n_res:
        pos = np.vstack([pos, np.column_stack([np.tile(DEPOT_XY, (n_res, 1)), np.full(n_res, phys.z_min)])])
        t = np.vstack([t, np.zeros((n_res, 2))])
    active = np.arange(pos.shape[0]) < u["n"]
    w = llmagent.World(phys, users, joint.Deployment(pos.copy(), t.copy()), active, INFO_M,
                       np.array(DEPOT_XY), fleet_actions=fleet)
    return w, float(np.mean(swarm.evaluate(phys, w.power())[1]))


def server_tag() -> str:
    return os.environ.get("VLLM_IMAGE") or "vllm " + subprocess.run(
        ["vllm", "--version"], capture_output=True, text=True).stdout.strip()


def run_llm_unit(u: dict, ep: llmagent.Endpoint, meta: dict | None = None) -> None:
    path = unit_path(u)
    world, s0 = start_world(u, fleet=u["stage"] == "l2")
    trace = path.with_suffix(".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(trace, "w") as tf:
        def log(r, hist, row):
            tf.write(json.dumps({"round": r, "served_pct": round(100 * hist[0], 3), "n_active": hist[1],
                                 "turns": row}) + "\n")
            tf.flush()
            print(f"    round {r}: served {100 * hist[0]:.1f} % with {hist[1]} drones", flush=True)
        r = llmagent.run(ep, world, u["framing"], rounds=2 if u["stage"] == "smoke" else ROUNDS,
                         inertia=INERTIA, seed=u["seed"] * 10 + u["repeat"], log=log)
    turns = [t for row in r.turns for t in row]
    acts = Counter(t["action"] for t in turns)
    _write(path, COLUMNS, [{
        "stage": u["stage"], "model": u["model"], **(meta or {}), "scenario": u["scenario"], "tier": u["tier"],
        "snapshot": u["snapshot"], "seed": u["seed"], "n_drones": u["n"], "framing": u["framing"],
        "repeat": u["repeat"], "info_m": INFO_M, "rounds": r.rounds, "served_start_pct": 100 * s0,
        "served_pct": 100 * r.served, "n_active_end": r.n_active,
        "best_round_pct": 100 * max(h[0] for h in r.history), "turns": len(turns),
        "tool_calls": sum(t["calls"] for t in turns), "what_ifs": sum(t["what_ifs"] for t in turns),
        "prompt_tokens": sum(t["prompt_tokens"] for t in turns),
        "completion_tokens": sum(t["completion_tokens"] for t in turns),
        "model_seconds": round(sum(t["seconds"] for t in turns), 1), "errors": sum(bool(t["error"]) for t in turns),
        "moves": acts["move_to"], "stays": acts["stay"], "lands": acts["land"], "calls": acts["call_reserve"],
        "applied": sum(t["applied"] is not None for t in turns), "time_s": round(r.time_s, 1),
        "history": json.dumps([[round(100 * s, 3), n] for s, n in r.history])}])


def run_base_unit(u: dict) -> int:
    path = unit_path(u)
    if path.exists():
        return 0
    world, s0 = start_world(u, fleet=False)
    util = swarm.Utility(u["framing"])
    r = swarm.run(world.phys, world.users, world.dep, util, "simultaneous", inertia=INERTIA, max_rounds=ROUNDS,
                  seed=u["seed"] * 10 + u["repeat"], info_m=INFO_M)
    _write(path, COLUMNS, [{
        "stage": u["stage"], "model": u["model"], "scenario": u["scenario"], "tier": u["tier"],
        "snapshot": u["snapshot"], "seed": u["seed"], "n_drones": u["n"], "framing": u["framing"],
        "repeat": u["repeat"], "info_m": INFO_M, "rounds": r.rounds, "served_start_pct": 100 * s0,
        "served_pct": 100 * r.served, "n_active_end": u["n"], "best_round_pct": 100 * max(r.history),
        "turns": "", "tool_calls": "", "what_ifs": "", "prompt_tokens": "", "completion_tokens": "",
        "model_seconds": "", "errors": "", "moves": sum(len(m) for m in r.moves), "stays": "", "lands": 0,
        "calls": 0, "applied": sum(len(m) for m in r.moves), "time_s": round(r.time_s, 1),
        "history": json.dumps([[round(100 * s, 3), u["n"]] for s in r.history])}])
    return 1


def run_pol_unit(u: dict) -> int:
    if unit_path(u).exists():
        return 0
    _, _, phys = _problem(_lay(u))
    ep = llmagent.Policy("local", u["model"], kind=u["model"].removeprefix("pol-"), framing=u["framing"],
                         z_range=(phys.z_min, phys.z_max), seed=u["seed"] * 10 + u["repeat"])
    run_llm_unit(u, ep)
    return 1


CONTAINER = "dc-vllm"


def serve(key: str, log_path: Path) -> subprocess.Popen:
    m = MODELS[key]
    args = ["serve", m.hf, "--served-model-name", m.served or key, "--port", str(PORT),
            "--enable-auto-tool-choice", "--tool-call-parser", m.tool_parser,
            "--max-model-len", "32768", "--gpu-memory-utilization", "0.90", "--trust-remote-code", *m.serve_args]
    if m.reasoning_parser:
        args += ["--reasoning-parser", m.reasoning_parser]
    image = os.environ.get("VLLM_IMAGE")
    if image:      # the pinned image; weights from the host's Hugging Face cache, published on localhost only
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
        cmd = ["docker", "run", "--rm", "--name", CONTAINER, "--gpus", "all", "--ipc=host",
               "-p", f"127.0.0.1:{PORT}:{PORT}", "-v", f"{hf_home}:/root/.cache/huggingface",
               "-e", "HF_HUB_OFFLINE=1", "--entrypoint", "vllm", image, *args, "--host", "0.0.0.0"]
    else:
        cmd = ["vllm", *args, "--host", "127.0.0.1"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("serving:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 3600:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited with {proc.returncode}; see {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    print(f"server ready after {time.time() - t0:.0f}s", flush=True)
                    return proc
        except OSError:
            pass
        time.sleep(10)
    proc.terminate()
    raise RuntimeError("vllm did not become healthy within an hour")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--stage", choices=STAGES, default="l1")
    ap.add_argument("--serve", action="store_true", help="start vllm serve for --model on this node")
    ap.add_argument("--base-url", default=f"http://127.0.0.1:{PORT}/v1")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.list and not args.model and args.stage not in CPU_STAGES:
        for k, m in MODELS.items():
            print(f"{k:20s} {m.hf:22s} tool={m.tool_parser} reasoning={m.reasoning_parser} thinking={m.thinking}")
        return
    if args.stage not in CPU_STAGES and not args.model:
        ap.error("--model is required for this stage")
    mine = units(args.stage, args.model)
    done = [u for u in mine if unit_path(u).exists()]
    if args.list or args.check:
        print(f"llm {args.stage} {args.model or ''}: {len(mine)} units, {len(done)} on disk")
        if args.check:
            ok = len(done) == len(mine)
            print(f"RESULT: {'PASS' if ok else 'FAIL'} llm {args.stage} {args.model or ''}")
            sys.exit(0 if ok else 1)
        return
    todo = [u for u in mine if not unit_path(u).exists()]
    print(f"llm {args.stage} {args.model or ''}: {len(todo)} of {len(mine)} units to run", flush=True)
    if args.stage in CPU_STAGES:
        with Pool(args.procs) as pool:
            fn = run_base_unit if args.stage == "l1base" else run_pol_unit
            for c, _ in enumerate(pool.imap_unordered(fn, todo), start=1):
                print(f"  {c}/{len(todo)}", flush=True)
        return
    proc = serve(args.model, OUT / "serve_logs" / f"{args.model}_{os.environ.get('SLURM_JOB_ID', 'local')}.log") \
        if args.serve else None
    try:
        m = MODELS[args.model]
        ep = m.endpoint(args.base_url, args.model)
        meta = {"hf_id": m.hf, "server": server_tag(), "thinking": int(m.thinking)}
        for c, u in enumerate(todo, start=1):
            t0 = time.time()
            print(f"  unit {c}/{len(todo)}: {unit_path(u).name}", flush=True)
            run_llm_unit(u, ep, meta)
            print(f"  unit {c}/{len(todo)} done in {time.time() - t0:.0f}s", flush=True)
    finally:
        if proc:
            if os.environ.get("VLLM_IMAGE"):
                subprocess.run(["docker", "stop", CONTAINER], capture_output=True)
            proc.terminate()
            proc.wait(timeout=120)


if __name__ == "__main__":
    main()
