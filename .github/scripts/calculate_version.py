#!/usr/bin/env python3
"""
Derived version calculator for ferromode — trunk-based development.

Mirrors the pirn scheme. The version is NOT stored in the manifests as the
source of truth; it is computed at build time from the anchor registry's
published history (PyPI `ferromode`) and then stamped across all packages by
scripts/set-version.sh so every language stays in lockstep.

Schema: MAJOR.MINOR.PATCH[.devRUN]

  MAJOR  — hardcoded in CI env (bump for breaking changes)
  MINOR  — latest MINOR on PyPI for that MAJOR + 1  (so each merge = a new minor)
  PATCH  — PR number (0 for direct main pushes)
  .devRUN — GitHub run number, appended only for TestPyPI preview builds

Outputs (to $GITHUB_OUTPUT): version, should_publish_test, should_publish_prod.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request

PACKAGE_NAME = os.environ.get("PACKAGE_NAME", "ferromode").strip()
MAJOR_VERSION = int(os.environ.get("MAJOR_VERSION", "0").strip() or "0")
GITHUB_REF = os.environ.get("GITHUB_REF", "").strip()
GITHUB_RUN_NUMBER = os.environ.get("GITHUB_RUN_NUMBER", "").strip()
GITHUB_EVENT_NAME = os.environ.get("GITHUB_EVENT_NAME", "push").strip()
GITHUB_BASE_REF = os.environ.get("GITHUB_BASE_REF", "").strip()
GITHUB_EVENT_ACTION = os.environ.get("GITHUB_EVENT_ACTION", "").strip()


def get_pypi_minor(package: str) -> tuple[int, int] | None:
    """Return (major, minor) of the latest published release, or None."""
    url = f"https://pypi.org/pypi/{package}/json"
    print(f"Querying PyPI: {url}")
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        version = data["info"]["version"]
        m = re.match(r"^(\d+)\.(\d+)", version)
        if not m:
            print(f"Could not parse PyPI version {version!r}", file=sys.stderr)
            return None
        return int(m.group(1)), int(m.group(2))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print("Package not yet on PyPI — first release")
            return None
        print(f"PyPI query failed ({exc.code}): {exc}", file=sys.stderr)
        return None
    except Exception as exc:  # noqa: BLE001 — network best-effort
        print(f"PyPI query failed: {exc}", file=sys.stderr)
        return None


def extract_pr_number(ref: str) -> str | None:
    m = re.search(r"refs/pull/(\d+)/", ref)
    return m.group(1) if m else None


def calculate() -> tuple[str, bool, bool]:
    is_pr = GITHUB_EVENT_NAME == "pull_request"
    is_pr_to_main = GITHUB_BASE_REF in ("main", "refs/heads/main")
    is_main_push = GITHUB_EVENT_NAME == "push" and GITHUB_REF == "refs/heads/main"
    pr_number = extract_pr_number(GITHUB_REF) if is_pr else None

    should_publish_test = is_pr and is_pr_to_main and GITHUB_EVENT_ACTION != "closed"
    should_publish_prod = is_main_push

    pypi = get_pypi_minor(PACKAGE_NAME)
    if pypi is None:
        new_minor = 0
    else:
        pypi_major, pypi_minor = pypi
        new_minor = (pypi_minor + 1) if pypi_major == MAJOR_VERSION else 1

    if is_main_push:
        version = f"{MAJOR_VERSION}.{new_minor}.0"
    elif is_pr and pr_number:
        version = f"{MAJOR_VERSION}.{new_minor}.{pr_number}"
        if should_publish_test and GITHUB_RUN_NUMBER:
            # Semver pre-release (valid for Cargo/npm); maturin normalizes
            # `-dev.N` to the PEP 440 dev release `.devN` for the wheel.
            version = f"{version}-dev.{GITHUB_RUN_NUMBER}"
    else:
        # Feature-branch push — dev build keyed on the run number.
        version = f"{MAJOR_VERSION}.{new_minor}.0-dev.{GITHUB_RUN_NUMBER or '0'}"

    print(f"EVENT={GITHUB_EVENT_NAME!r} ACTION={GITHUB_EVENT_ACTION!r} BASE={GITHUB_BASE_REF!r}")
    print(f"is_pr={is_pr} pr_to_main={is_pr_to_main} pr={pr_number} main_push={is_main_push}")
    print(f"version={version} test={should_publish_test} prod={should_publish_prod}")
    return version, should_publish_test, should_publish_prod


def main() -> None:
    version, test, prod = calculate()
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"version={version}\n")
            f.write(f"should_publish_test={str(test).lower()}\n")
            f.write(f"should_publish_prod={str(prod).lower()}\n")


if __name__ == "__main__":
    main()
