# Releasing Ferromode

Ferromode ships from one source tree to every language registry. Releases are
**tag-driven and gated**: pushing a `vX.Y.Z` tag creates a GitHub Release and
then fans out to per-registry publish workflows, each of which waits for manual
approval via a GitHub **Environment** before it publishes.

## What ships where

| Channel | Registry / location | Workflow | Auth |
|---|---|---|---|
| Rust core | crates.io | `publish-crates.yml` | `CARGO_TOKEN` secret |
| Python | PyPI | `publish-pypi.yml` | `PYPI_TOKEN` secret |
| WASM/JS | npm | `publish-npm.yml` | `NPM_TOKEN` secret |
| R | CRAN (manual submission) | `publish-cran.yml` | — (prepares tarball) |
| Julia | General registry | Registrator app + `TagBot.yml` | — (GitHub App) |
| C++ | GitHub Release artifacts | `publish-native.yml` | `GITHUB_TOKEN` |
| MEX | GitHub Release artifacts (+ File Exchange, Octave `pkg install`) | `publish-native.yml` | `GITHUB_TOKEN` |

## One-time setup

1. **Secrets** (Settings → Secrets and variables → Actions): add `CARGO_TOKEN`,
   `PYPI_TOKEN`, `NPM_TOKEN`. C++/MEX use the automatic `GITHUB_TOKEN`; Julia and
   CRAN need no token.
2. **Environments** (Settings → Environments): `publish-crates`, `publish-pypi`,
   `publish-npm`, `publish-cran`, `publish-native`. Add a required reviewer to
   each so publishing waits for approval.
3. **Julia**: install the [JuliaRegistrator](https://github.com/JuliaRegistries/Registrator)
   GitHub App on this repo (one time).

## Cutting a release

1. Push a semver tag:
   ```bash
   git tag v0.2.0 && git push origin v0.2.0
   ```
   `release.yml` parses the version from the tag, creates the GitHub Release, and
   dispatches the publish workflows. Each build job runs `scripts/set-version.sh`
   to stamp the version into every manifest (Rust workspace → also drives maturin
   and wasm-pack; plus the R `DESCRIPTION` and Julia `Project.toml`), so the tag
   is the single source of truth — no manual version bump needed.
2. **Approve** each gated publish workflow in the Actions tab.
3. **Julia**: comment on the release commit:
   ```
   @JuliaRegistrator register subdir=crates/ferromode-julia/julia/Ferromode.jl
   ```
   Once the General-registry PR merges, `TagBot.yml` creates the Julia tag/release.
4. **CRAN**: `publish-cran.yml` produces a checked tarball; submit it via
   <https://cran.r-project.org/submit.html> (CRAN requires manual submission).
5. **MEX on File Exchange** (optional): link the GitHub release from a
   [File Exchange](https://www.mathworks.com/matlabcentral/fileexchange) entry,
   which can auto-package GitHub releases.

## Notes

- `scripts/set-version.sh` is idempotent, so it is safe whether you bump-commit
  before tagging or tag a bare commit.
- C++/MEX have no central registry; GitHub Release archives are the channel
  (`ferromode-cxx-*` = header + per-platform lib; `ferromode-mex-*` = per-platform
  MEX + `.m` wrappers, plus a Linux Octave build).
- A `.mltbx` MATLAB toolbox is intentionally **not** built yet: it requires a
  per-OS MATLAB MEX build matrix (free in CI for public repos via
  `matlab-actions/setup-matlab`) and is deferred until there's demand.
