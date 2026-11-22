"""Tests for antenna module."""

import numpy as np
import pytest

from dronecomm.antenna import ParametricAntenna, DOWNLINK_ANTENNA, BACKHAUL_ANTENNA


def test_boresight_gain():
    """Gain at theta=0 should be G_max."""
    ant = ParametricAntenna(g_max_dbi=18.0, beamwidth_deg=60.0, sla_db=20.0)
    assert ant.gain_dbi(np.array(0.0)) == pytest.approx(18.0)


def test_3db_point():
    """Gain at theta_3dB/2 should be G_max - 3 dB."""
    ant = ParametricAntenna(g_max_dbi=18.0, beamwidth_deg=60.0, sla_db=20.0)
    theta_half = np.deg2rad(60.0) / 2  # Half of full beamwidth
    gain = ant.gain_dbi(np.array(theta_half))
    assert gain == pytest.approx(18.0 - 3.0, abs=1e-10)


def test_sidelobe_floor():
    """Gain far off-boresight should be G_max - SLA."""
    ant = ParametricAntenna(g_max_dbi=18.0, beamwidth_deg=60.0, sla_db=20.0)
    theta_far = np.deg2rad(180.0)
    gain = ant.gain_dbi(np.array(theta_far))
    assert gain == pytest.approx(18.0 - 20.0, abs=1e-10)


def test_gain_monotonically_decreasing():
    """Gain should decrease (or stay flat) as theta increases."""
    ant = DOWNLINK_ANTENNA
    thetas = np.linspace(0, np.pi, 200)
    gains = ant.gain_dbi(thetas)
    diffs = np.diff(gains)
    assert np.all(diffs <= 1e-10)


def test_gain_linear_positive():
    """Linear gain should always be positive."""
    ant = BACKHAUL_ANTENNA
    thetas = np.linspace(0, np.pi, 200)
    gains = ant.gain_linear(thetas)
    assert np.all(gains > 0)


def test_gain_linear_at_boresight():
    """Linear gain at boresight = 10^(G_max/10)."""
    ant = ParametricAntenna(g_max_dbi=18.0)
    g_lin = ant.gain_linear(np.array(0.0))
    assert g_lin == pytest.approx(10.0 ** (18.0 / 10.0), rel=1e-6)


def test_beamwidth_property():
    """beamwidth_rad should match deg2rad of beamwidth_deg."""
    ant = ParametricAntenna(beamwidth_deg=45.0)
    assert ant.beamwidth_rad == pytest.approx(np.deg2rad(45.0))


def test_vectorized():
    """Gain computation should work on arrays."""
    ant = DOWNLINK_ANTENNA
    thetas = np.array([0, 0.1, 0.5, 1.0, np.pi])
    gains = ant.gain_dbi(thetas)
    assert gains.shape == (5,)


def _radiated_over_isotropic(antenna):
    th = np.linspace(0.0, np.pi, 20001)
    return np.trapz(antenna.gain_linear(th) * np.sin(th), th) / 2.0


def test_grounded_patterns_conserve_power():
    """A passive antenna cannot radiate more than an isotropic one: mean gain over the sphere <= 1."""
    from dronecomm.config import Config
    from dronecomm.optimize import _build_models
    dl, bh, _ = _build_models(Config.from_yaml("configs/grounded.yaml"))
    assert _radiated_over_isotropic(dl) <= 1.0
    assert _radiated_over_isotropic(bh) <= 1.0


def test_icc_userlink_pattern_violates_power_conservation():
    """The idealised userlink antenna (18 dBi, 60 deg, -2 dBi floor) radiates about 6x isotropic: kept only for reproduction."""
    from dronecomm.antenna import ParametricAntenna
    assert _radiated_over_isotropic(ParametricAntenna(18.0, 60.0, 20.0)) > 6.0
