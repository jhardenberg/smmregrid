"""Tests for CDO executable resolution and version checking"""

import os
import stat
import pytest
from smmregrid.cdobinary import resolve_cdo, parse_version, cdo_version, check_cdo_version, CDO_MIN_VERSION


@pytest.fixture(autouse=True)
def clear_cache():
    """Version is cached per executable: clear it between tests"""
    cdo_version.cache_clear()
    yield
    cdo_version.cache_clear()


def fake_cdo(tmp_path, version):
    """Create a fake CDO executable printing the given version"""
    path = tmp_path / "cdo"
    path.write_text(f"#!/bin/sh\necho 'Climate Data Operators version {version} (https://mpimet.mpg.de/cdo)'\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_resolve_cdo(monkeypatch):
    """Explicit argument wins over CDO environment variable, which wins over PATH"""
    monkeypatch.setenv("CDO", "/opt/homebrew/bin/cdo")
    assert resolve_cdo("/my/cdo") == "/my/cdo"
    assert resolve_cdo() == "/opt/homebrew/bin/cdo"
    monkeypatch.delenv("CDO")
    assert resolve_cdo() == "cdo"


def test_parse_version():
    assert parse_version("2.6.10") > parse_version("2.6.5")
    assert parse_version("2.5.0") < parse_version(CDO_MIN_VERSION)


def test_installed_cdo_is_supported():
    """The CDO used for testing must satisfy the minimum version"""
    assert check_cdo_version() == cdo_version()


def test_fake_cdo_version(tmp_path):
    cdo = fake_cdo(tmp_path, "9.9.9")
    assert check_cdo_version(cdo) == "9.9.9"


def test_old_cdo_raises(tmp_path):
    cdo = fake_cdo(tmp_path, "2.5.0")
    with pytest.raises(RuntimeError, match=f"requires CDO >= {CDO_MIN_VERSION}"):
        check_cdo_version(cdo)


def test_old_cdo_from_env(tmp_path, monkeypatch):
    """CdoGenerate picks up the CDO environment variable and fails on old CDO"""
    from smmregrid import CdoGenerate
    monkeypatch.setenv("CDO", fake_cdo(tmp_path, "2.5.0"))
    with pytest.raises(RuntimeError):
        CdoGenerate("r10x10", "r20x20")


def test_missing_cdo():
    with pytest.raises(FileNotFoundError):
        check_cdo_version(os.path.join("nonexistent", "cdo"))
