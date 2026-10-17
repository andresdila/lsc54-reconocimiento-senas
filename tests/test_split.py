"""Controles anti-fuga sobre la partición versionada (se ejecutan en cada cambio)."""
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lsc.data.splits import check_split, make_signer_split  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs" / "split.yaml").read_text(encoding="utf-8"))
COVERAGE = pd.read_csv(ROOT / "data" / "inventory" / "s2_coverage_zip_index.csv")
SPLIT_FILE = ROOT / "splits" / f"split_{CFG['version']}.csv"


def _split():
    return make_signer_split(COVERAGE, CFG["seed"], CFG["test_signers_per_group"], CFG["n_folds"])


def test_split_is_deterministic():
    assert _split().equals(_split())


def test_committed_split_matches_regeneration():
    committed = pd.read_csv(SPLIT_FILE)
    assert committed.equals(_split())


def test_no_signer_leakage_and_full_class_coverage():
    report = check_split(_split(), COVERAGE)
    assert len(report["signers_test"]) == 2 * CFG["test_signers_per_group"]


def test_check_split_catches_a_planted_leak():
    leaky = _split()
    leaky.loc[0, "signer_key"] = leaky.loc[len(leaky) - 1, "signer_key"]  # mismo señante dos veces
    with pytest.raises(AssertionError):
        check_split(leaky, COVERAGE)
