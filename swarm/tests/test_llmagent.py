"""Tests for dronecomm.llmagent: the tool loop with a scripted model in place of an endpoint."""

import json
from dataclasses import replace

import numpy as np
import pytest

from dronecomm import joint, llmagent, swarm
from dronecomm.config import Config


class Scripted(llmagent.Endpoint):
    """Replies with the given tool calls, one assistant message per call, then stays."""

    def __init__(self, script):
        super().__init__("http://unused", "scripted")
        self.script = list(script)

    def chat(self, messages, tools=llmagent.TOOLS):
        if not self.script:
            name, args = "stay", {}
        else:
            name, args = self.script.pop(0)
        if name is None:                                   # a reply with no tool call
            return {"content": "thinking"}, {"prompt_tokens": 10, "completion_tokens": 2}, 0.01
        call = {"id": f"c{len(messages)}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}
        return {"content": "", "tool_calls": [call]}, {"prompt_tokens": 10, "completion_tokens": 5}, 0.01


@pytest.fixture
def world():
    cfg = Config()
    rng = np.random.default_rng(3)
    centres = np.array([[300.0, 300.0], [700.0, 650.0], [400.0, 800.0]])
    xy = np.concatenate([c + rng.normal(0.0, 60.0, (40, 2)) for c in centres])
    users = np.column_stack([np.clip(xy, 0.0, cfg.scenario.area_size_m), np.zeros(len(xy))])
    phys = joint.Physics.from_config(cfg, users.shape[0])
    phys = replace(phys, lam=phys.lam * swarm.kappa(phys, users.shape[0]) / 0.4)
    pos = np.vstack([np.column_stack([centres + 80.0, np.full(3, phys.z_min)]), [[50.0, 50.0, phys.z_min]]])
    dep = joint.Deployment(pos, np.zeros((4, 2)))
    return llmagent.World(phys, users, dep, np.array([True, True, True, False]), 600.0, np.array([50.0, 50.0]))


def test_turn_observes_tests_and_moves(world):
    ep = Scripted([("observe", {}), ("what_if", {"x": 320, "y": 310, "z": 60, "tilt_deg": 10, "azimuth_deg": 45}),
                   ("move_to", {"x": 320, "y": 310, "z": 60, "tilt_deg": 10, "azimuth_deg": 45})])
    t = llmagent.run_turn(ep, world, world.power(), 0, "network", "First turn.")
    assert t["action"] == "move_to" and t["calls"] == 3 and t["what_ifs"] == 1 and t["error"] is None
    assert llmagent.apply(world, 0, t) == "move"
    assert world.dep.pos[0] == pytest.approx([320.0, 310.0, 60.0])
    assert joint.tilt_deg(world.dep.t[0]) == pytest.approx(10.0)


def test_what_if_matches_the_restricted_model(world):
    tools = llmagent.AgentTools(world, world.power(), 1)
    a = {"x": 650, "y": 600, "z": 80, "tilt_deg": 20, "azimuth_deg": 200}
    res = tools.what_if(a)
    mv = swarm.AgentMove(world.phys, world.users, world.power(), 1, swarm.Utility("global"),
                         world.dep.pos[1], world.info_m, world.dep.pos)
    m = world.users.shape[0]
    assert res["people_served_in_your_area"] == world.people(mv(llmagent._vec(world.phys, a))[0] * m)


def test_what_if_budget_and_missing_action(world):
    script = [("what_if", {"x": 300, "y": 300, "z": 50, "tilt_deg": 0, "azimuth_deg": 0})] * (llmagent.WHAT_IF_BUDGET + 1)
    t = llmagent.run_turn(Scripted(script + [(None, {})]), world, world.power(), 0, "selfish", "")
    assert t["what_ifs"] == llmagent.WHAT_IF_BUDGET and t["action"] == "stay"


def test_land_and_call(world):
    assert llmagent.apply(world, 2, {"action": "call_reserve", "args": {}}) == "call"
    assert world.active.all()
    assert llmagent.apply(world, 2, {"action": "call_reserve", "args": {}}) is None      # no reserve left
    assert llmagent.apply(world, 1, {"action": "land", "args": {}}) == "land"
    assert not world.active[1]


def test_run_rounds(world):
    class Stay(Scripted):
        def __init__(self):
            super().__init__([])
    r = llmagent.run(Stay(), world, "area", rounds=2, workers=2)
    assert len(r.history) == 3 and r.history[0] == r.history[-1]
    assert all(t["action"] == "stay" for row in r.turns for t in row)


def test_trace_keeps_tool_results_and_text(world):
    a = {"x": 320, "y": 310, "z": 60, "tilt_deg": 10, "azimuth_deg": 45}
    ep = Scripted([(None, {}), ("observe", {}), ("what_if", a), ("move_to", a)])
    t = llmagent.run_turn(ep, world, world.power(), 0, "area", "")
    assert [e["tool"] for e in t["trace"]] == ["observe", "what_if", "move_to"]
    assert "me" in t["trace"][0]["result"] and "change_in_your_area" in t["trace"][1]["result"]
    assert "result" not in t["trace"][2]                   # the action ends the turn
    assert t["text"] == [{"content": "thinking", "reasoning": ""}]


def test_run_logs_realised_change(world):
    a = {"x": 320, "y": 310, "z": 60, "tilt_deg": 10, "azimuth_deg": 45}
    r = llmagent.run(Scripted([("move_to", a)]), world, "area", rounds=1, inertia=1.0, workers=1)
    row = r.turns[0]
    assert all(len(t["people_before"]) == 2 and len(t["people_after"]) == 2 for t in row)
    assert sum(t["applied"] == "move" for t in row) == 1


def test_gated_framing_adds_the_rule(world):
    class Capture(Scripted):
        def chat(self, messages, tools=llmagent.TOOLS):
            self.system = messages[0]["content"]
            return super().chat(messages, tools)
    for framing, gated in (("area", False), ("area" + llmagent.GATED, True)):
        ep = Capture([])
        llmagent.run_turn(ep, world, world.power(), 0, framing, "")
        assert llmagent.FRAMINGS["area"] in ep.system and (llmagent.GATE_TEXT in ep.system) == gated


def test_fixed_fleet_hides_land_and_call(world):
    world.fleet_actions = False
    assert {t["function"]["name"] for t in world.tools()} == {"observe", "what_if", "move_to", "stay"}
    t = llmagent.run_turn(Scripted([("land", {})]), world, world.power(), 0, "area", "")
    assert t["action"] == "stay" and world.active.sum() == 3


def test_policies(world):
    p = world.power()
    stay = llmagent.run_turn(llmagent.Policy("local", "pol-stay", kind="stay"), world, p, 0, "network", "")
    assert stay["action"] == "stay" and stay["calls"] == 1
    goto = llmagent.run_turn(llmagent.Policy("local", "pol-goto", kind="goto"), world, p, 0, "network", "")
    groups = llmagent.AgentTools(world, p, 0).observe()["unserved_groups"]
    if groups:
        top = max(groups, key=lambda g: g["people"])
        assert goto["action"] == "move_to" and (goto["args"]["x"], goto["args"]["y"]) == (top["x"], top["y"])
    pol = llmagent.Policy("local", "pol-rand6", kind="rand6", framing="selfish", seed=1)
    t = llmagent.run_turn(pol, world, p, 0, "selfish", "")
    assert t["what_ifs"] == llmagent.WHAT_IF_BUDGET and t["action"] in ("move_to", "stay")
    if t["action"] == "move_to":
        assert llmagent.AgentTools(world, p, 0).what_if(t["args"])["change_in_your_cell"] > 0


def test_refused_request_ends_the_turn_not_the_run(world):
    """A request the server rejects (HTTP 4xx, e.g. past the context length) ends that agent's turn as stay."""
    class Refusing(Scripted):
        def chat(self, messages, tools=llmagent.TOOLS):
            if len(messages) > 3:
                raise llmagent.RequestRefused("HTTP 400: maximum context length")
            return super().chat(messages, tools)

    ep = Refusing([("observe", {}), ("what_if", {"x": 300, "y": 300, "z": 50, "tilt_deg": 0, "azimuth_deg": 0})])
    t = llmagent.run_turn(ep, world, world.power(), 0, "network", "")
    assert t["action"] == "stay" and t["error"].startswith("refused: HTTP 400") and t["calls"] == 1
