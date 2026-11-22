"""One LLM per drone: local tools over dronecomm.swarm, served by any OpenAI-compatible endpoint (vLLM on Ibex).

Every model and every objective framing gets the same tools; only the objective sentence of the
system prompt changes (FRAMINGS). A framing ending in GATED (e.g. "area+gate") adds GATE_TEXT, an
instruction to move only on a gain its own what_if predicted: a diagnostic of the decision rule. What an agent can see is what a `swarm` agent with information
radius `info_m` sees: its own cell, the drones within comm range (`comm_m`) and the users those
drones serve. `what_if` predicts a candidate pose with exactly that restricted model
(swarm.AgentMove with info_m), reporting both its own cell and its observed area, so the
best-response baselines of stage p3 (selfish / global with the same info_m) are the matching
algorithmic agents.

One round: every drone in the air takes a turn on the same snapshot (in parallel, so the server
batches), using at most MAX_CALLS tool calls of which at most WHAT_IF_BUDGET are `what_if`, and
ending with exactly one action (move_to, stay, land, call_reserve). Each action is applied with
probability `inertia` (simultaneous play with inertia; dronecomm.swarm.run). A turn that ends
without an action counts as `stay`. Agents keep no conversation across rounds; each turn opens
with a one-line memory of the agent's previous action and what changed since.

Units shown to agents are people (active users), metres and degrees.

`Policy` is a scripted agent behind the same interface, as a baseline with the LLM's information and
budget: `stay`; `goto` (fly to the largest unserved group observe reports, no what_if); `rand6` (what_if
on WHAT_IF_BUDGET random poses within RAND_RADIUS_M, move to the best if it gains); `groups6` (the
same, trying the unserved groups first, nadir at the current altitude). The rand6 and groups6 policies
rank candidates on the framing's number (cell for selfish, area otherwise).
"""

from __future__ import annotations

import json
import math
import time
import zlib
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import swarm
from .joint import Deployment, Physics, boresight, rx_power, tilt_deg

FRAMINGS = {
    "selfish": "Your goal: maximise the number of people YOUR drone serves. Other drones' users do not count for you.",
    "area": "Your goal: maximise the number of people served in YOUR AREA (the users of your cell and of your "
            "neighbours' cells), whichever drone serves them.",
    "network": "Your goal: maximise the number of people served in the WHOLE NETWORK. You only see your area, "
               "so judge the network by what you can see.",
}
GATED = "+gate"
GATE_TEXT = ("Move only to a pose for which a what_if predicted that the people counted by your goal would "
             "rise; if no what_if did, stay.")
WHAT_IF_BUDGET = 6
MAX_CALLS = 12
MAX_TOKENS = 8192                 # per model call; thinking models reason before the tool call
REQUEST_TIMEOUT_S = 600.0
POLICIES = ("stay", "goto", "rand6", "groups6")
RAND_RADIUS_M = 300.0             # rand6 / groups6: random candidates within this of the drone
RAND_TILT_DEG = 60.0

SYSTEM = """You control one drone, {name}, in a fleet of aerial base stations restoring mobile service after a disaster.
Users attach to the drone they hear strongest. A user is served only if their drone's radio band has room for their
guaranteed rate; drones share one frequency band, so a drone's beam also interferes with other drones' users.
Your drone has a directional antenna you can tilt (0 deg = straight down, up to 80 deg) and turn (azimuth, deg,
counter-clockwise from east). The area is {area:.0f} m x {area:.0f} m; altitude must be {zmin:.0f}-{zmax:.0f} m.
You see only your own cell and the drones within {comm:.0f} m of you, through the inter-drone link.

{objective}

Each turn: look (observe), optionally test candidate poses with what_if (at most {budget} per turn; predictions
assume the other drones stay where they are), then end the turn with exactly ONE action: {actions}.
Keep reasoning short."""
ACTION_TEXT = {True: "move_to, stay, land (go home if you add little), or call_reserve (ask for one more drone to be "
                     "sent to your unserved users)", False: "move_to or stay"}

TOOLS = [
    {"type": "function", "function": {
        "name": "observe", "description": "Your drone, your cell, the drones within comm range, and where nearby "
        "users are going unserved.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "what_if", "description": "Predict the outcome if YOUR drone took this pose now (others fixed): "
        "people served in your cell and in your area, and the change from now.",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"},
            "tilt_deg": {"type": "number"}, "azimuth_deg": {"type": "number"}},
            "required": ["x", "y", "z", "tilt_deg", "azimuth_deg"]}}},
    {"type": "function", "function": {
        "name": "move_to", "description": "ACTION: fly to this pose (ends your turn).",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"},
            "tilt_deg": {"type": "number"}, "azimuth_deg": {"type": "number"}},
            "required": ["x", "y", "z", "tilt_deg", "azimuth_deg"]}}},
    {"type": "function", "function": {
        "name": "stay", "description": "ACTION: keep your pose (ends your turn).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "land", "description": "ACTION: return to base; your users will attach to other drones (ends your turn).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "call_reserve", "description": "ACTION: request one reserve drone, sent to your largest group of "
        "unserved users (ends your turn). Fails if no reserve drone is left.",
        "parameters": {"type": "object", "properties": {}}}},
]
ACTIONS = {"move_to", "stay", "land", "call_reserve"}


def objective(framing: str) -> str:
    """The objective sentence(s) of a framing: FRAMINGS[base], plus GATE_TEXT for a gated framing."""
    if framing.endswith(GATED):
        return FRAMINGS[framing.removesuffix(GATED)] + " " + GATE_TEXT
    return FRAMINGS[framing]


# ── Model endpoint ───────────────────────────────────────────────────


class RequestRefused(Exception):
    """The server rejected the request (HTTP 4xx), e.g. a conversation past the context length: the model's
    failure, not a transient one, so the turn ends (run_turn) instead of the run."""


@dataclass
class Endpoint:
    """An OpenAI-compatible /v1/chat/completions endpoint (vLLM serve --enable-auto-tool-choice)."""

    base_url: str
    model: str
    temperature: float = 0.7
    extra: dict = field(default_factory=dict)      # e.g. chat_template_kwargs

    def chat(self, messages: list[dict], tools: list[dict] = TOOLS) -> tuple[dict, dict, float]:
        """(assistant message, usage, seconds); retries transient failures; RequestRefused on an HTTP 4xx."""
        body = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto",
                "max_tokens": MAX_TOKENS, "temperature": self.temperature, **self.extra}
        data = json.dumps(body).encode()
        for attempt in range(4):
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(f"{self.base_url}/chat/completions", data=data,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as r:
                    out = json.loads(r.read())
                return out["choices"][0]["message"], out.get("usage", {}), time.perf_counter() - t0
            except urllib.error.HTTPError as e:
                if 400 <= e.code < 500:
                    raise RequestRefused(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from e
                if attempt == 3:
                    raise RuntimeError(f"endpoint failed: {e}") from e
                time.sleep(5.0 * (attempt + 1))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == 3:
                    raise RuntimeError(f"endpoint failed: {e}") from e
                time.sleep(5.0 * (attempt + 1))
        raise AssertionError


@dataclass
class Policy(Endpoint):
    """A scripted agent answering through the tool interface (module docstring); reads its own transcript."""

    kind: str = "rand6"
    framing: str = "network"
    z_range: tuple = (50.0, 120.0)
    seed: int = 0

    def chat(self, messages: list[dict], tools: list[dict] = TOOLS) -> tuple[dict, dict, float]:
        results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        if self.kind == "stay":
            return self._call("stay", {}, messages)
        if not results:
            return self._call("observe", {}, messages)
        obs = results[0]
        me = obs["me"]
        if self.kind == "goto":
            if not obs["unserved_groups"]:
                return self._call("stay", {}, messages)
            g = max(obs["unserved_groups"], key=lambda q: q["people"])
            return self._call("move_to", {"x": g["x"], "y": g["y"], "z": me["z"], "tilt_deg": 0.0, "azimuth_deg": 0.0},
                              messages)
        cands = self.candidates(obs)
        tried = results[1:]
        if len(tried) < len(cands):
            return self._call("what_if", cands[len(tried)], messages)
        key = "change_in_your_cell" if self.framing == "selfish" else "change_in_your_area"
        best = max(tried, key=lambda r: r.get(key, -math.inf))
        if best.get(key, 0) > 0:
            return self._call("move_to", best["pose"], messages)
        return self._call("stay", {}, messages)

    def candidates(self, obs: dict) -> list[dict]:
        me = obs["me"]
        rng = np.random.default_rng([self.seed, zlib.crc32(json.dumps(me, sort_keys=True).encode())])
        out = []
        if self.kind == "groups6":
            out = [{"x": g["x"], "y": g["y"], "z": me["z"], "tilt_deg": 0.0, "azimuth_deg": 0.0}
                   for g in obs["unserved_groups"][:WHAT_IF_BUDGET]]
        while len(out) < WHAT_IF_BUDGET:
            r, a = RAND_RADIUS_M * math.sqrt(rng.random()), 2.0 * math.pi * rng.random()
            out.append({"x": me["x"] + r * math.cos(a), "y": me["y"] + r * math.sin(a),
                        "z": float(rng.uniform(*self.z_range)), "tilt_deg": float(rng.uniform(0.0, RAND_TILT_DEG)),
                        "azimuth_deg": float(rng.uniform(0.0, 360.0))})
        return out

    @staticmethod
    def _call(name: str, args: dict, messages: list[dict]) -> tuple[dict, dict, float]:
        call = {"id": f"p{len(messages)}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        return {"content": "", "tool_calls": [call]}, {"prompt_tokens": 0, "completion_tokens": 0}, 0.0


# ── World: the shared state and what one agent may see ───────────────


@dataclass
class World:
    phys: Physics
    users: np.ndarray
    dep: Deployment                 # every drone owned
    active: np.ndarray
    info_m: float
    depot_xy: np.ndarray
    fleet_actions: bool = True      # offer land and call_reserve (False: a fixed fleet)

    def tools(self) -> list[dict]:
        return [t for t in TOOLS if self.fleet_actions or t["function"]["name"] not in ("land", "call_reserve")]

    def power(self) -> np.ndarray:
        p = rx_power(self.phys, self.dep.pos, self.dep.bore, self.users)
        p[~self.active] = 0.0
        return p

    def people(self, points: float) -> int:
        return int(round(points * self.phys.lam))


def _pose(v: np.ndarray) -> dict:
    t = v[3:5]
    return {"x": round(float(v[0]), 1), "y": round(float(v[1]), 1), "z": round(float(v[2]), 1),
            "tilt_deg": round(float(tilt_deg(t)), 1),
            "azimuth_deg": round(float(np.degrees(math.atan2(t[1], t[0])) % 360.0), 1) if np.hypot(*t) > 1e-9 else 0.0}


def _vec(phys: Physics, a: dict) -> np.ndarray:
    tilt = math.radians(min(max(float(a["tilt_deg"]), 0.0), 80.0))
    az = math.radians(float(a["azimuth_deg"]))
    return np.array([min(max(float(a["x"]), 0.0), phys.area_m), min(max(float(a["y"]), 0.0), phys.area_m),
                     min(max(float(a["z"]), phys.z_min), phys.z_max),
                     math.tan(tilt) * math.cos(az), math.tan(tilt) * math.sin(az)])


class AgentTools:
    """Agent j's tools on a frozen snapshot of the world (its turn sees the round's start)."""

    def __init__(self, world: World, power: np.ndarray, j: int):
        self.w, self.j = world, j
        self.v0 = np.concatenate([world.dep.pos[j], world.dep.t[j]])
        self.cell = swarm.AgentMove(world.phys, world.users, power, j, swarm.Utility("selfish"),
                                    world.dep.pos[j], world.info_m, world.dep.pos)
        self.area = swarm.AgentMove(world.phys, world.users, power, j, swarm.Utility("global"),
                                    world.dep.pos[j], world.info_m, world.dep.pos)
        self.now = (self.cell(self.v0)[0], self.area(self.v0)[0])
        self.assoc, self.adm, self.band = swarm.evaluate(world.phys, power)
        self.what_ifs = 0

    def observe(self) -> dict:
        w, j = self.w, self.j
        m = w.users.shape[0]
        mine = self.assoc == j
        near = (np.hypot(*(w.dep.pos[:, :2] - w.dep.pos[j, :2]).T) <= w.info_m) & w.active
        near[j] = False
        bc, mass = swarm.beacons(w.users, self.assoc, self.adm, near | (np.arange(len(near)) == j))
        return {
            "me": {**_pose(self.v0), "people_in_cell": w.people(np.sum(mine)),
                   "people_served": w.people(np.sum(self.adm[mine])),
                   "band_used_pct": round(100.0 * float(np.sum(self.band[mine])), 1),
                   "people_unserved": w.people(np.sum(1.0 - self.adm[mine]))},
            "area": {"people_served": w.people(self.now[1] * m), "people_seen": w.people(self.area.cols.size)},
            "neighbours": [{"id": int(k), **_pose(np.concatenate([w.dep.pos[k], w.dep.t[k]])),
                            "people_served": w.people(np.sum(self.adm[self.assoc == k])),
                            "band_used_pct": round(100.0 * float(np.sum(self.band[self.assoc == k])), 1),
                            "people_unserved": w.people(np.sum(1.0 - self.adm[self.assoc == k]))}
                           for k in np.nonzero(near)[0]],
            "unserved_groups": [{"x": round(float(c[0]), 1), "y": round(float(c[1]), 1), "people": w.people(s)}
                                for c, s in zip(bc[:6], mass[:6])],
            "reserve_drones": int(np.sum(~w.active)),
        }

    def what_if(self, a: dict) -> dict:
        if self.what_ifs >= WHAT_IF_BUDGET:
            return {"error": f"what_if budget ({WHAT_IF_BUDGET}) used up this turn; act now"}
        self.what_ifs += 1
        v = _vec(self.w.phys, a)
        m = self.w.users.shape[0]
        cell, area = self.cell(v)[0], self.area(v)[0]
        return {"pose": _pose(v), "people_served_in_your_cell": self.w.people(cell * m),
                "change_in_your_cell": self.w.people((cell - self.now[0]) * m),
                "people_served_in_your_area": self.w.people(area * m),
                "change_in_your_area": self.w.people((area - self.now[1]) * m)}


def run_turn(ep: Endpoint, world: World, power: np.ndarray, j: int, framing: str, memory: str) -> dict:
    """One agent's turn: returns {'action', 'args', 'calls', 'what_ifs', 'usage', 'seconds', 'error', 'trace', 'text'}.

    trace: every tool call with its arguments and the result the agent was given; text: what the model
    wrote on each call (content, and reasoning when the server returns it).
    """
    tools = AgentTools(world, power, j)
    phys = world.phys
    system = SYSTEM.format(name=f"D{j}", area=phys.area_m, zmin=phys.z_min, zmax=phys.z_max, comm=world.info_m,
                           objective=objective(framing), budget=WHAT_IF_BUDGET,
                           actions=ACTION_TEXT[world.fleet_actions])
    offered = world.tools()
    allowed = {t["function"]["name"] for t in offered}
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": f"New round. {memory} Take your turn."}]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    out = {"action": "stay", "args": {}, "calls": 0, "what_ifs": 0, "seconds": 0.0, "error": None, "trace": [],
           "text": []}
    for _ in range(MAX_CALLS):
        try:
            msg, u, sec = ep.chat(msgs, offered)
        except RequestRefused as e:          # e.g. the model's own output pushed the turn past the context
            out.update(what_ifs=tools.what_ifs, usage=usage, error=f"refused: {e}")
            return out
        out["seconds"] += sec
        for k in usage:
            usage[k] += int(u.get(k, 0) or 0)
        calls = msg.get("tool_calls") or []
        said = {"content": msg.get("content") or "", "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or ""}
        if said["content"] or said["reasoning"]:
            out["text"].append(said)
        msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls} if calls
                    else {"role": "assistant", "content": msg.get("content") or ""})
        if not calls:
            out["error"] = "no_tool_call"
            msgs.append({"role": "user", "content": f"End your turn with one action tool: {ACTION_TEXT[world.fleet_actions]}."})
            continue
        for c in calls:
            out["calls"] += 1
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = None
            entry = {"tool": name, "args": args}
            out["trace"].append(entry)
            if args is None:
                res = {"error": "arguments are not valid JSON"}
            elif name not in allowed:
                res = {"error": f"unknown tool {name}"}
            elif name in ACTIONS:
                if name == "move_to":
                    try:
                        _vec(phys, args)
                    except (KeyError, TypeError, ValueError):
                        res = {"error": "move_to needs x, y, z, tilt_deg, azimuth_deg"}
                        entry["result"] = res
                        msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(res)})
                        continue
                out.update(action=name, args=args, what_ifs=tools.what_ifs, usage=usage, error=None)
                return out
            elif name == "observe":
                res = tools.observe()
            elif name == "what_if":
                try:
                    res = tools.what_if(args)
                except (KeyError, TypeError, ValueError):
                    res = {"error": "what_if needs x, y, z, tilt_deg, azimuth_deg"}
            else:
                res = {"error": f"unknown tool {name}"}
            entry["result"] = res
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(res)})
    out.update(what_ifs=tools.what_ifs, usage=usage, error=out["error"] or "max_calls")
    return out


def apply(world: World, j: int, turn: dict) -> str | None:
    """Apply agent j's action to the world; returns an event ('move', 'land', 'call') or None."""
    w = world
    if turn["action"] == "move_to":
        v = _vec(w.phys, turn["args"])
        w.dep.pos[j], w.dep.t[j] = v[:3], v[3:5]
        return "move"
    if turn["action"] == "land" and w.active.sum() > 1:
        w.active[j] = False
        w.dep.pos[j, :2] = w.depot_xy
        return "land"
    if turn["action"] == "call_reserve" and not w.active.all():
        p = w.power()
        assoc, adm, _ = swarm.evaluate(w.phys, p)
        c, _ = swarm.beacons(w.users, assoc, adm, np.arange(len(w.active)) == j, per_drone=1)
        if len(c):
            r = int(np.nonzero(~w.active)[0][0])
            w.dep.pos[r] = [*c[0], w.dep.pos[j, 2]]
            w.dep.t[r] = 0.0
            w.active[r] = True
            return "call"
    return None


@dataclass
class LLMRunResult:
    served: float
    n_active: int
    rounds: int
    history: list                      # (served share, drones in the air) after each round
    turns: list                        # per round, per agent: action, calls, what_ifs, tokens, seconds, error
    time_s: float


def run(ep: Endpoint, world: World, framing: str, rounds: int = 10, inertia: float = 0.5, seed: int = 0,
        workers: int = 32, log=None) -> LLMRunResult:
    """Simultaneous rounds with inertia; every active agent takes a turn on the round's snapshot."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    p = world.power()
    history = [(float(np.mean(swarm.evaluate(world.phys, p)[1])), int(world.active.sum()))]
    memory = {j: "This is your first turn." for j in range(len(world.active))}
    turns_all = []
    for r in range(1, rounds + 1):
        p = world.power()
        ids = [int(j) for j in np.nonzero(world.active)[0]]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            turns = list(pool.map(lambda j: run_turn(ep, world, p, j, framing, memory[j]), ids))
        before = {j: AgentTools(world, p, j).now for j in ids}
        events = {}
        for j, t in zip(ids, turns):
            if t["action"] != "stay" and rng.random() < inertia:
                events[j] = apply(world, j, t)
        p = world.power()
        m = world.users.shape[0]
        after_now = {}
        for j, t in zip(ids, turns):
            ev = events.get(j)
            if not world.active[j]:
                continue
            after = after_now[j] = AgentTools(world, p, j).now
            act = t["action"] if ev else ("stay" if t["action"] == "stay" else f"{t['action']} (not applied this round)")
            memory[j] = (f"Last round you chose {act}. Since then the people served in your cell went from "
                         f"{world.people(before[j][0] * m)} to {world.people(after[0] * m)}, in your area from "
                         f"{world.people(before[j][1] * m)} to {world.people(after[1] * m)}.")
        history.append((float(np.mean(swarm.evaluate(world.phys, p)[1])), int(world.active.sum())))
        row = [{"agent": j, "action": t["action"], "applied": events.get(j), "calls": t["calls"],
                "what_ifs": t["what_ifs"], "prompt_tokens": t["usage"].get("prompt_tokens", 0),
                "completion_tokens": t["usage"].get("completion_tokens", 0), "seconds": round(t["seconds"], 2),
                "error": t["error"], "trace": t["trace"], "text": t["text"],
                "people_before": [world.people(x * m) for x in before[j]],
                "people_after": [world.people(x * m) for x in after_now[j]] if j in after_now else None}
               for j, t in zip(ids, turns)]
        turns_all.append(row)
        if log:
            log(r, history[-1], row)
    return LLMRunResult(history[-1][0], history[-1][1], rounds, history, turns_all, time.perf_counter() - t0)
