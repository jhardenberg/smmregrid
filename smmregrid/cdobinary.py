"""Location and version checking of the CDO executable"""

import os
import re
import sys
import shutil
import subprocess
from functools import lru_cache

# Minimum CDO version supported by smmregrid
CDO_MIN_VERSION = "2.6.1"

MACOS_HINT = ("On macOS, recent CDO versions are not available from conda-forge: "
              "please install CDO with Homebrew ('brew install cdo') and remove "
              "conda's cdo from your environment, or point the CDO environment "
              "variable to the Homebrew binary ('export CDO=$(brew --prefix)/bin/cdo').")


def resolve_cdo(cdo=None):
    """
    Return the CDO executable to be used.

    Priority is: explicit argument, then the CDO environment variable
    (the same used by python-cdo), then 'cdo' from the PATH.
    """
    if cdo:
        return cdo
    return os.environ.get("CDO") or "cdo"


def parse_version(version):
    """Convert a version string as '2.6.4' into a comparable tuple (2, 6, 4)"""
    return tuple(int(x) for x in version.split("."))


@lru_cache(maxsize=None)
def cdo_version(cdo=None):
    """
    Return the version string of the CDO executable (e.g. '2.6.4').
    The result is cached for each executable.
    """
    cdo = resolve_cdo(cdo)
    if shutil.which(cdo) is None:
        message = f"CDO executable '{cdo}' not found."
        if sys.platform == "darwin":
            message += " " + MACOS_HINT
        raise FileNotFoundError(message)

    output = subprocess.run([cdo, "--version"], capture_output=True, text=True, check=False)
    match = re.search(r"version (\d+\.\d+\.\d+)", output.stdout + output.stderr)
    if not match:
        raise RuntimeError(f"Cannot determine the version of CDO executable '{cdo}'")
    return match.group(1)


def check_cdo_version(cdo=None, min_version=CDO_MIN_VERSION):
    """
    Raise an error if the CDO executable is older than the minimum supported version.

    Returns:
        str: The CDO version found.
    """
    cdo = resolve_cdo(cdo)
    version = cdo_version(cdo)
    if parse_version(version) < parse_version(min_version):
        message = (f"CDO {version} found at '{shutil.which(cdo)}', "
                   f"but smmregrid requires CDO >= {min_version}.")
        if sys.platform == "darwin":
            message += " " + MACOS_HINT
        raise RuntimeError(message)
    return version
