"""Regression tests for the fixes applied to waveletec.

These cover the pure/cheap logic that previously crashed or silently
misbehaved: formula parsing, CLI kwarg parsing, combination filtering, reader
de-duplication, conditional sampling, the DWT forward/inverse round-trip, and
the conditional-sampling partition wiring (distinct output suffixes, list/None
inputs). Continuous-wavelet paths are skipped when ``pycwt`` is unavailable.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from waveletec.core import wavelet_functions as wf
from waveletec.core import commons
from waveletec.core import handlers
from waveletec.extra.partitioning import coimbra_et_al_2025 as co
from waveletec.io import readers
from waveletec import main as wmain


# --------------------------------------------------------------------------- #
# formula_to_vars
# --------------------------------------------------------------------------- #
def test_formula_to_vars_parses_components():
    v = wf.formula_to_vars("w*co2|w*h2o")
    assert v.xy == ["w", "co2"]
    assert v.condsamp_pair == [["w", "h2o"]]
    assert v.condsamp_flat == ["w", "h2o"]
    assert v.uniquevars == ["w", "co2", "h2o"]
    assert v.combinations == ["w*co2", "w*h2o"]


def test_formula_to_vars_is_deterministic():
    # set()-based dedup used to make ordering non-reproducible across runs
    f = "w*co2|w*h2o|w*ch4|w*co2"
    first = wf.formula_to_vars(f)
    for _ in range(5):
        again = wf.formula_to_vars(f)
        assert again.uniquevars == first.uniquevars
        assert again.combinations == first.combinations
    # duplicate "w*co2" collapsed, order preserved
    assert first.combinations == ["w*co2", "w*h2o", "w*ch4"]
    assert first.uniquevars == ["w", "co2", "h2o", "ch4"]


# --------------------------------------------------------------------------- #
# __custom_params__
# --------------------------------------------------------------------------- #
def test_custom_params_scalar_flag_and_list():
    assert wmain.__custom_params__(["--alpha", "5"]) == {"alpha": 5}
    assert wmain.__custom_params__(["--beta", "1.5"]) == {"beta": 1.5}
    assert wmain.__custom_params__(["--verbose"]) == {"verbose": True}
    assert wmain.__custom_params__(["--xs", "1", "2", "3"]) == {"xs": [1, 2, 3]}
    assert wmain.__custom_params__(["--name", "abc"]) == {"name": "abc"}


def test_custom_params_value_without_key_raises():
    with pytest.raises(ValueError):
        wmain.__custom_params__(["7"])


# --------------------------------------------------------------------------- #
# available_combinations
# --------------------------------------------------------------------------- #
def test_available_combinations_filters_by_present_variables():
    out = commons.available_combinations(
        ["w*co2", "w*ch4"], variables_available=["u", "v", "w", "ts", "co2", "h2o"])
    assert out == ["w*co2"]


# --------------------------------------------------------------------------- #
# readers de-duplication
# --------------------------------------------------------------------------- #
def test_read_raw_and_read_fluxnet_are_equivalent(tmp_path):
    csv = tmp_path / "tiny.csv"
    csv.write_text(
        "TIMESTAMP_START,TIMESTAMP_END,co2\n"
        "202205130000,202205130030,1.0\n"
        "202205130030,202205130100,2.0\n")
    a = readers.read_raw(str(csv))
    b = readers.read_fluxnet(str(csv))
    xr.testing.assert_identical(a, b)
    assert "TIMESTAMP" in a.coords or "TIMESTAMP" in a.dims


# --------------------------------------------------------------------------- #
# _as_series
# --------------------------------------------------------------------------- #
def test_as_series_resolves_none_str_and_list():
    d = xr.Dataset({"a": ("t", [1.0, 2.0]), "b": ("t", [3.0, 4.0])})
    assert co._as_series(d, None) == 0
    np.testing.assert_array_equal(co._as_series(d, "a").values, [1.0, 2.0])
    np.testing.assert_array_equal(co._as_series(d, ["a", "b"]).values, [4.0, 6.0])


# --------------------------------------------------------------------------- #
# conditional_sampling
# --------------------------------------------------------------------------- #
def test_conditional_sampling_keys_and_masking():
    w = xr.DataArray(np.array([1.0, -1.0, 1.0, -1.0]), dims=["t"])
    c = xr.DataArray(np.array([1.0, 1.0, -1.0, -1.0]), dims=["t"])
    Y = w * c
    ds = co.conditional_sampling(Y, c, names=["wc"], label={1: "+", -1: "-"})
    assert set(ds.data_vars) == {"wc+", "wc-"}
    # the two sign-masked components partition Y exactly
    recombined = ds["wc+"] + ds["wc-"]
    np.testing.assert_allclose(recombined.values, Y.values)


# --------------------------------------------------------------------------- #
# universal_wt
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore:Level value of")
def test_universal_wt_dwt_roundtrip_reconstructs_signal():
    rng = np.random.default_rng(42)
    sig = xr.DataArray(rng.standard_normal(256), dims=["ns"])
    res = wf.universal_wt(sig, method="dwt", wt=True, iwt=True, fs=10, fn=8)
    assert {"signal", "wave", "approximation"}.issubset(res.data_vars)
    recon = res["wave"].sum("natural_frequency") + res["approximation"]
    np.testing.assert_allclose(recon.values, sig.values, atol=1e-9)


def test_universal_wt_rejects_unsupported_flag_combinations():
    sig = xr.DataArray(np.zeros(64), dims=["ns"])
    with pytest.raises(NotImplementedError):
        wf.universal_wt(sig, method="dwt", wt=True, iwt=False)
    with pytest.raises(NotImplementedError):
        wf.universal_wt(sig, method="dwt", wt=False, iwt=True)


@pytest.mark.skipif(wf.pycwt is None, reason="pycwt not installed")
def test_universal_wt_cwt_roundtrip_runs():
    rng = np.random.default_rng(1)
    sig = xr.DataArray(rng.standard_normal(128), dims=["ns"])
    res = wf.universal_wt(sig, method="cwt", wt=True, iwt=True, fs=10, fn=8)
    assert "approximation" in res.data_vars


# --------------------------------------------------------------------------- #
# data_partition: distinct suffixes, list/None inputs don't crash
# --------------------------------------------------------------------------- #
def _covariance_dataset():
    t = pd.date_range("2022-05-13", periods=4, freq="30min")
    rng = np.random.default_rng(7)
    cols = [
        "wco2",
        "wco2-wh2o+", "wco2-wh2o-",
        "wco2+wco+", "wco2+wco-", "wco2-wco+", "wco2-wco-",
        "wco2+wch4+", "wco2+wch4-", "wco2-wch4+", "wco2-wch4-",
    ]
    data = {c: ("TIMESTAMP", rng.standard_normal(4)) for c in cols}
    return xr.Dataset(data, coords={"TIMESTAMP": t})


def test_data_partition_produces_distinct_non_colliding_suffixes():
    ds = _covariance_dataset()
    out = handlers.data_partition(
        ds, variables_available=["w", "co2", "h2o", "co", "ch4"])
    # primary H2O channel stays unsuffixed
    for base in ("NEE", "GPP", "Reco"):
        assert base in out.data_vars
    # other channels are suffixed and therefore do not clobber each other
    for name in ("NEE_pH2O_CO", "NEE_pCO", "NEE_pCH4"):
        assert name in out.data_vars
    # the CO/CH4 branches use a list CO2neg_H2Opos and CO2neg_H2Oneg=None;
    # their presence proves _as_series handled both without raising.
    assert "ffCO2_pCO" in out.data_vars
    assert "ffCO2_pCH4" in out.data_vars


def test_data_partition_skips_channels_with_missing_variables():
    ds = _covariance_dataset()
    out = handlers.data_partition(ds, variables_available=["w", "co2", "h2o"])
    # only the H2O channel should run
    assert "NEE" in out.data_vars
    assert "NEE_pCO" not in out.data_vars
    assert "NEE_pCH4" not in out.data_vars


# --------------------------------------------------------------------------- #
# partition_DWCS_CO night masking / None handling
# --------------------------------------------------------------------------- #
def test_partition_dwcs_co_handles_list_and_none_inputs():
    ds = _covariance_dataset()
    out = co.partition_DWCS_CO(
        ds, CO2="wco2",
        CO2neg_H2Opos=["wco2-wco+", "wco2-wco-"], CO2neg_H2Oneg=None,
        CO2pos_COpos="wco2+wco+", CO2pos_COneg="wco2+wco-", NIGHT=None)
    for base in ("NEE", "GPP", "Reco", "ffCO2"):
        assert base in out.data_vars
    # with NIGHT=None, islight is 1 everywhere -> GPP is not forced to zero
    assert np.isfinite(out["GPP"].values).all()
