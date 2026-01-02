"""
Pytest fixtures for Streamlit app testing.

Provides mock Streamlit functions and sample result dictionaries.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys


# ------------------------------------------------------------------
# Mock Streamlit module
# ------------------------------------------------------------------

class MockStreamlitColumn:
    """Mock for st.columns() context manager."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class MockStreamlit:
    """Complete mock of streamlit module."""

    def __init__(self):
        self._metrics = []
        self._errors = []
        self._warnings = []
        self._infos = []
        self._markdowns = []
        self._expanders = []
        self._session_state = {}  # Mock session state storage

    def metric(self, label, value, help=None, delta=None):
        self._metrics.append({"label": label, "value": value, "help": help})

    def error(self, msg):
        self._errors.append(msg)

    def warning(self, msg):
        self._warnings.append(msg)

    def info(self, msg):
        self._infos.append(msg)

    def markdown(self, text, **kwargs):
        self._markdowns.append(text)

    def latex(self, text):
        self._markdowns.append(f"LATEX:{text}")

    def columns(self, *args, **kwargs):
        # Return a list of mock columns
        n = args[0] if args else 4
        if isinstance(n, list):
            n = len(n)
        return [MockStreamlitColumn() for _ in range(n)]

    def expander(self, label, expanded=False):
        self._expanders.append(label)
        return MockStreamlitColumn()

    def plotly_chart(self, fig, use_container_width=False):
        pass

    def dataframe(self, df, **kwargs):
        pass

    def write(self, *args, **kwargs):
        pass

    def code(self, text, language=None):
        pass

    def progress(self, value):
        mock = MagicMock()
        mock.progress = MagicMock()
        mock.empty = MagicMock()
        return mock

    def empty(self):
        mock = MagicMock()
        mock.text = MagicMock()
        mock.empty = MagicMock()
        return mock

    def cache_data(self, ttl=None, **kwargs):
        def decorator(func):
            return func
        return decorator

    def session_state(self):
        return {}


@pytest.fixture(autouse=True)
def mock_streamlit(monkeypatch):
    """
    Auto-used fixture that mocks streamlit for all tests.

    Returns the mock for assertions.
    """
    import importlib

    mock_st = MockStreamlit()

    # Create a mock module
    mock_module = MagicMock()
    mock_module.metric = mock_st.metric
    mock_module.error = mock_st.error
    mock_module.warning = mock_st.warning
    mock_module.info = mock_st.info
    mock_module.markdown = mock_st.markdown
    mock_module.latex = mock_st.latex
    mock_module.columns = mock_st.columns
    mock_module.expander = mock_st.expander
    mock_module.plotly_chart = mock_st.plotly_chart
    mock_module.dataframe = mock_st.dataframe
    mock_module.write = mock_st.write
    mock_module.code = mock_st.code
    mock_module.progress = mock_st.progress
    mock_module.empty = mock_st.empty
    mock_module.cache_data = mock_st.cache_data
    mock_module.session_state = mock_st._session_state

    # Patch streamlit in sys.modules BEFORE any imports
    monkeypatch.setitem(sys.modules, 'streamlit', mock_module)

    # Remove any cached streamlit_app modules to force reload with mock
    modules_to_remove = [
        key for key in sys.modules.keys()
        if key.startswith('streamlit_app')
    ]
    for key in modules_to_remove:
        monkeypatch.delitem(sys.modules, key, raising=False)

    return mock_st


# ------------------------------------------------------------------
# Result fixtures
# ------------------------------------------------------------------

@pytest.fixture
def quick_result():
    """
    Quick result dict with 4 keys.

    Returned by cached_quick_kappa().
    """
    return {
        "kappa": 0.417293962,
        "c": 2.137454406,
        "valid": True,
        "message": "Computation successful",
    }


@pytest.fixture
def full_result():
    """
    Full result dict with 22 keys.

    Returned by cached_full_kappa().
    """
    return {
        # Core values
        "kappa": 0.417293962,
        "c": 2.137454406,
        "R": 1.3036,
        "theta": 4/7,
        "K": 3,

        # Decomposition components
        "S12_plus": 1.2345,
        "S12_minus": 0.1234,
        "S34": 0.4567,
        "m": 8.681,  # exp(R) + 5 for K=3

        # Integral components
        "I1_plus": 0.5123,
        "I1_minus": 0.0512,
        "I2_plus": 0.4123,
        "I2_minus": 0.0412,
        "I3_plus": 0.2123,
        "I4_plus": 0.1123,

        # Correction factors
        "g_I1": 1.0,
        "g_I2": 1.0,
        "g_total": 1.0,
        "base": 2.137,

        # Error bounds
        "error_bounds": {
            "quadrature": 1e-10,
            "numerical": 1e-8,
            "total": 1e-7,
        },

        # Rigorous kappa
        "kappa_rigorous": 0.417,

        # Per-pair breakdown
        "per_pair": {
            (1, 1): {"I1": 0.1, "I2": 0.2, "I3": 0.05, "I4": 0.03, "S12": 0.3, "S34": 0.08},
            (1, 2): {"I1": 0.15, "I2": 0.25, "I3": 0.06, "I4": 0.04, "S12": 0.4, "S34": 0.1},
            (1, 3): {"I1": 0.12, "I2": 0.22, "I3": 0.05, "I4": 0.03, "S12": 0.34, "S34": 0.08},
            (2, 2): {"I1": 0.18, "I2": 0.28, "I3": 0.07, "I4": 0.05, "S12": 0.46, "S34": 0.12},
            (2, 3): {"I1": 0.14, "I2": 0.24, "I3": 0.06, "I4": 0.04, "S12": 0.38, "S34": 0.1},
            (3, 3): {"I1": 0.16, "I2": 0.26, "I3": 0.065, "I4": 0.045, "S12": 0.42, "S34": 0.11},
        },
    }


@pytest.fixture
def full_result_with_none_values():
    """
    Full result dict with some None values (edge case).
    """
    return {
        "kappa": 0.417293962,
        "c": 2.137454406,
        "R": 1.3036,
        "theta": 4/7,
        "K": 3,
        "S12_plus": 1.2345,
        "S12_minus": 0.1234,
        "S34": 0.4567,
        "m": 8.681,
        "I1_plus": 0.5123,
        "I1_minus": 0.0512,
        "I2_plus": 0.4123,
        "I2_minus": 0.0412,
        "I3_plus": 0.2123,
        "I4_plus": 0.1123,
        "g_I1": 1.0,
        "g_I2": 1.0,
        "g_total": 1.0,
        "base": 2.137,
        "error_bounds": None,  # Could happen if compute_errors=False
        "kappa_rigorous": None,
        "per_pair": None,  # Could happen if compute_per_pair=False
    }


@pytest.fixture
def error_result():
    """
    Result dict with error sub-dicts.
    """
    return {
        "kappa": 0.417293962,
        "c": 2.137454406,
        "R": 1.3036,
        "theta": 4/7,
        "K": 3,
        "S12_plus": 1.2345,
        "S12_minus": 0.1234,
        "S34": 0.4567,
        "m": 8.681,
        "I1_plus": 0.5123,
        "I1_minus": 0.0512,
        "I2_plus": 0.4123,
        "I2_minus": 0.0412,
        "I3_plus": 0.2123,
        "I4_plus": 0.1123,
        "g_I1": 1.0,
        "g_I2": 1.0,
        "g_total": 1.0,
        "base": 2.137,
        "error_bounds": {"error": "Failed to compute error bounds"},
        "kappa_rigorous": None,
        "per_pair": {"error": "Failed to compute per-pair breakdown"},
    }


@pytest.fixture
def none_result():
    """
    None result (pre-computation state).
    """
    return None


# ------------------------------------------------------------------
# Polynomial coefficient fixtures
# ------------------------------------------------------------------

@pytest.fixture
def przz_coefficients():
    """
    PRZZ baseline coefficients.

    From przz_parameters.json.
    """
    return {
        "P1_coeffs": [1.0, -2.0, 1.0],  # Example - not real PRZZ values
        "P2_coeffs": [1.0, -3.0, 3.0, -1.0],
        "P3_coeffs": [1.0, -4.0, 6.0, -4.0, 1.0],
        "Q_coeffs": {0: 0.5, 1: 0.3, 2: 0.2},
        "R": 1.3036,
        "theta": 4/7,
        "K": 3,
    }


# ------------------------------------------------------------------
# Session state fixtures
# ------------------------------------------------------------------

@pytest.fixture
def initialized_session_state():
    """
    Pre-initialized session state with required keys.
    """
    return {
        "mode": "kappa",
        "P1_coeffs": [1.0, -2.0, 1.0],
        "P2_coeffs": [1.0, -3.0, 3.0, -1.0],
        "P3_coeffs": [1.0, -4.0, 6.0, -4.0, 1.0],
        "Q_coeffs": {0: 0.5, 1: 0.3, 2: 0.2},
        "R": 1.3036,
        "theta": 4/7,
        "K": 3,
        "n_quad": 60,
        "quick_result": None,
        "full_result": None,
        "computation_mode": "quick",
        "last_computed_hash": None,
    }


# ------------------------------------------------------------------
# Full result key set (for contract validation)
# ------------------------------------------------------------------

QUICK_RESULT_KEYS = {"kappa", "c", "valid", "message"}

FULL_RESULT_KEYS = {
    "kappa", "c", "R", "theta", "K",
    "S12_plus", "S12_minus", "S34", "m",
    "I1_plus", "I1_minus", "I2_plus", "I2_minus",
    "I3_plus", "I4_plus",
    "g_I1", "g_I2", "g_total", "base",
    "error_bounds", "kappa_rigorous", "per_pair",
}

PER_PAIR_KEYS = {"I1", "I2", "I3", "I4", "S12", "S34"}


@pytest.fixture
def quick_result_keys():
    """Expected keys in quick result."""
    return QUICK_RESULT_KEYS


@pytest.fixture
def full_result_keys():
    """Expected keys in full result."""
    return FULL_RESULT_KEYS


@pytest.fixture
def per_pair_keys():
    """Expected keys in per_pair sub-dicts."""
    return PER_PAIR_KEYS
