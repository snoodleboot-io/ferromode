#!/usr/bin/env bash
# Inject a release version into every package manifest.
#
# The Rust workspace version (root Cargo.toml [workspace.package]) propagates to
# every crate via `version.workspace = true`, and both maturin (PyPI) and
# wasm-pack (npm) derive their package version from it. The manifests that do not
# read Cargo.toml — the R DESCRIPTION and the Julia Project.toml — are set
# explicitly.
#
# Usage: scripts/set-version.sh <X.Y.Z | vX.Y.Z>
set -euo pipefail

VERSION="${1:?usage: set-version.sh X.Y.Z}"
VERSION="${VERSION#v}"  # tolerate a leading 'v' from a git tag

if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z.-]+)?$ ]]; then
  echo "error: not a semver version: $VERSION" >&2
  exit 1
fi

root="$(cd "$(dirname "$0")/.." && pwd)"

# Rust workspace — the leading-anchored `version = "..."` is unique to
# [workspace.package] (dependency versions are `name = { version = ... }`).
sed -i -E 's/^version = "[^"]*"/version = "'"$VERSION"'"/' "$root/Cargo.toml"

# R package.
sed -i -E 's/^Version: .*/Version: '"$VERSION"'/' "$root/crates/ferromode-r/DESCRIPTION"

# Julia package.
sed -i -E 's/^version = .*/version = "'"$VERSION"'"/' \
  "$root/crates/ferromode-julia/julia/Ferromode.jl/Project.toml"

echo "Set release version to $VERSION:"
grep -m1 '^version = ' "$root/Cargo.toml"
grep -m1 '^Version: ' "$root/crates/ferromode-r/DESCRIPTION"
grep -m1 '^version = ' "$root/crates/ferromode-julia/julia/Ferromode.jl/Project.toml"
