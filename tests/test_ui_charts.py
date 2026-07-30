"""Smoke tests for the Plotly chart builders (skipped if plotly not installed)."""
import os, sys
import numpy as np
import pytest
pytest.importorskip("plotly")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import ui_charts as ui  # noqa: E402


def test_chart_builders_return_figures():
    import plotly.graph_objects as go
    assert isinstance(ui.cost_waterfall(0.4, 0.07, 0.5, 0.0), go.Figure)
    assert isinstance(ui.mac_distribution(np.random.default_rng(0).normal(1, .3, 500),
                                          0.4, 0.6, 1.0, 1.6), go.Figure)
    X, Y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    assert isinstance(ui.envelope_heatmap(X, Y, X + Y, (X + Y) < 1.0, "x", "y"), go.Figure)
    assert isinstance(ui.sobol_tornado(["a", "b"], [0.2, 0.1], [0.5, 0.3]), go.Figure)
    assert isinstance(ui.reliability_diagram([0.5, 0.9], [0.4, 0.8], [0.5, 0.9]), go.Figure)
