# Releasing Ferromode

Releases are **trunk-based and automatic** — there are no release tags and no
manual version bumps. The version is **derived from PyPI's published history**
(`.github/scripts/calculate_version.py`) and stamped across every language by
`scripts/set-version.sh`, so all packages stay in lockstep.

## Version scheme — `MAJOR.MINOR.PATCH[.devN]`

| Part | Source |
|---|---|
| **MAJOR** | the `MAJOR_VERSION` env in `release.yml` (bump by hand for breaking changes) |
| **MINOR** | latest MINOR on PyPI for that MAJOR **+ 1** → every merge to `main` is a new minor |
| **PATCH** | the PR number (preview builds only); merges are always `.0` |
| **`.devN`** | GitHub run number, on TestPyPI previews only |

## Flow

| Event | What happens |
|---|---|
| **PR → main** | derive `MAJOR.MINOR.<PR>.devN` → publish **TestPyPI**, `cargo publish --dry-run`, `npm pack --dry-run`. No approval needed. |
| **Merge → main** | derive `MAJOR.MINOR.0` → publish **PyPI**, **npm**, **crates.io**, and cut a **GitHub Release** with the C++/MEX archives. Each prod publish waits on its environment's required reviewer. |

To cut a release you just **merge a PR to `main`** and approve the gates. That's it.

CRAN and Julia are **not** in the automatic flow (both require manual steps):
- **Julia** → comment on the merge commit: `@JuliaRegistrator register subdir=crates/ferromode-julia/julia/Ferromode.jl`; `TagBot.yml` then tags it.
- **CRAN** → run `publish-cran.yml` (manual dispatch) to build a checked tarball, then submit it at <https://cran.r-project.org/submit.html>.

## Channels & auth

| Channel | Registry | Auth |
|---|---|---|
| Python | TestPyPI (PR) / PyPI (merge) | **OIDC trusted publishing** (no token) |
| crates.io | crates.io (merge) | `CARGO_TOKEN` secret |
| npm | `@snoodleboot-io/ferromode` (merge) | `NPM_TOKEN` secret (bootstrap; → OIDC later) |
| C++ / MEX | GitHub Release archives (merge) | `GITHUB_TOKEN` |

## One-time setup

1. **Secrets** (Settings → Secrets and variables → Actions): `CARGO_TOKEN`, `NPM_TOKEN`. *(PyPI uses OIDC — no token.)*
2. **Environments** (already created): `pypi`, `npm`, `crates`, `native` with a required reviewer; `testpypi` ungated.
3. **PyPI + TestPyPI Trusted Publishers** (pending publishers — configure before the first publish, no token needed):
   - On **pypi.org** and **test.pypi.org** → *Publishing* → add a GitHub Actions trusted publisher:
     - Owner `snoodleboot-io`, repo `ferromode`, workflow `release.yml`
     - Environment: `pypi` (on pypi.org) / `testpypi` (on test.pypi.org)
4. **npm**: bootstrap creates `@snoodleboot-io/ferromode` on the first merge (via `NPM_TOKEN`); then add a Trusted Publisher on the package and drop `NPM_TOKEN`.
5. **Julia**: install the [JuliaRegistrator](https://github.com/apps/juliateam-registrator) GitHub App.

## Notes

- `MAJOR` is the only number a human touches — bump `MAJOR_VERSION` in `release.yml` for a breaking release.
- crates.io / npm versions are immutable; the derived-minor scheme guarantees a fresh version every merge, so re-publishes never collide.
- C++/MEX have no registry — the GitHub Release archives are the channel (`ferromode-cxx-*` = header + per-platform lib; `ferromode-mex-*` = per-platform MEX + `.m` wrappers, plus a Linux Octave build). Optionally link the release from MATLAB File Exchange.
