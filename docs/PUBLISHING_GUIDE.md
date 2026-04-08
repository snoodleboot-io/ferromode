# Ferromode V1.0 Publishing Guide

Complete guide for setting up and executing the automated V1.0 release across all 4 package registries.

---

## 📋 Quick Reference

| Registry | Type | Command | Time |
|----------|------|---------|------|
| **crates.io** | Rust | Tag → Auto-published | 1-5 min |
| **PyPI** | Python | Tag → Auto-published | 1-5 min |
| **npm** | JavaScript/WASM | Tag → Auto-published | 1-5 min |
| **CRAN** | R | Manual submission | 3-7 days |

---

## 🔐 Step 1: Configure Publishing Tokens (One-time Setup)

### 1.1 Create Tokens for Each Registry

#### PyPI Token
1. Visit: https://pypi.org/account/
2. Login to your PyPI account
3. Go to Account Settings → API tokens
4. Create new token with scope "Entire account"
5. Copy the token (starts with `pypi-`)

**Token format:** `pypi-AgEIc...` (long string)

#### crates.io Token
1. Visit: https://crates.io/me
2. Login with GitHub (or create account)
3. Go to Account Settings → API Tokens
4. Click "New Token"
5. Copy the token (save securely)

**Token format:** Long alphanumeric string

#### npm Token
1. Visit: https://www.npmjs.com/settings/~/tokens
2. Login to npm account
3. Click "Generate new token"
4. Select "Automation" type
5. Copy the token

**Token format:** `npm_...` (long string)

#### CRAN (No token needed)
CRAN requires manual submission — instructions provided in workflow.

### 1.2 Store Tokens in GitHub Secrets

1. Go to your GitHub repository
2. Navigate to: **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**

Add these secrets:

| Secret Name | Value | Source |
|-------------|-------|--------|
| `PYPI_TOKEN` | `pypi-AgE...` | PyPI Account |
| `CARGO_TOKEN` | (token) | crates.io Account |
| `NPM_TOKEN` | `npm_...` | npm Account |

**Screenshot locations:**
```
GitHub Repo
  └─ Settings (tab)
      └─ Secrets and variables (left sidebar)
          └─ Actions (submenu)
              └─ New repository secret (green button)
```

**Verification:**
```bash
# After adding secrets, they appear as "●●●●●●" in the UI
# They're only readable in workflows, not in browser
```

---

## 🚀 Step 2: Prepare Release

### 2.1 Verify All Tests Pass

```bash
# Run full test suite
cargo test --all

# Build documentation
cargo doc --no-deps

# Check formatting
cargo fmt -- --check

# Run clippy lints
cargo clippy --all-targets -- -D warnings

# Test Python bindings (if you have training model)
pytest crates/ferromode-py/tests/

# Build WASM
cd crates/ferromode-wasm
npm install
wasm-pack build --release
cd ../..

# Expected: All tests pass, no errors
```

### 2.2 Update Version Numbers

All version numbers must match:

```bash
# Check current versions
grep "^version" crates/ferromode/Cargo.toml
grep "^version" crates/ferromode-py/Cargo.toml
grep "^version" crates/ferromode-wasm/Cargo.toml
# All should show: version = "0.1.0"

# Update to 1.0.0
# 1. Edit Cargo.toml workspace version
sed -i 's/version = "0.1.0"/version = "1.0.0"/' Cargo.toml
sed -i 's/version = "0.1.0"/version = "1.0.0"/' crates/ferromode/Cargo.toml
sed -i 's/version = "0.1.0"/version = "1.0.0"/' crates/ferromode-py/Cargo.toml
sed -i 's/version = "0.1.0"/version = "1.0.0"/' crates/ferromode-wasm/Cargo.toml
sed -i 's/version = "0.1.0"/version = "1.0.0"/' crates/ferromode-r/DESCRIPTION

# Verify
grep "^version" crates/ferromode/Cargo.toml
# Expected: version = "1.0.0"

# Python version
grep "version" crates/ferromode-py/setup.py 2>/dev/null || echo "No setup.py"

# npm version
grep "version" crates/ferromode-wasm/package.json

# R version
grep "Version:" crates/ferromode-r/DESCRIPTION
```

### 2.3 Update Changelog

Create or update `CHANGELOG.md`:

```markdown
# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-04-08

### Added
- Initial public release
- LSTM neural boundary prediction models (V2.2)
- Multi-platform support: Python, JavaScript/WASM, R, Rust
- GPU acceleration support (NVIDIA CUDA, AMD ROCm, Apple Metal)
- Comprehensive documentation and examples

### Changed
- API stabilized for 1.0 release
- Performance optimizations across all platforms

### Fixed
- Various stability improvements

---

## [0.1.0] - Initial Development
(Previous pre-release versions)
```

### 2.4 Create Release Commit

```bash
# Stage changes
git add Cargo.toml crates/*/Cargo.toml crates/*/DESCRIPTION crates/*/package.json CHANGELOG.md

# Commit
git commit -m "chore: Release v1.0.0

- Update all version numbers to 1.0.0
- Update CHANGELOG.md
- Prepare for multi-registry publishing"

# Verify
git log -1 --oneline
```

---

## 🏷️ Step 3: Create Git Tag and Trigger Release

### 3.1 Create Release Tag

```bash
# Make sure you're on main or release branch
git branch --show-current

# Create annotated tag (required for release workflows)
git tag -a v1.0.0 -m "Release v1.0.0 - Ferromode GA

- Multi-platform support (Rust, Python, JavaScript, R)
- LSTM neural boundary prediction
- GPU acceleration (CUDA, ROCm, Metal)
- Production-ready deployment"

# Verify tag
git tag -l v1.0.0 -n5
```

### 3.2 Push Tag to Trigger Workflows

```bash
# Push tag (this triggers the release.yml workflow)
git push origin v1.0.0

# Check GitHub Actions tab for workflow status
# Expected: 4 workflows trigger automatically
# - publish-crates.yml (Rust)
# - publish-pypi.yml (Python)
# - publish-npm.yml (JavaScript)
# - publish-cran.yml (R)

echo "✅ Tag pushed. Check https://github.com/ferromode/ferromode/actions"
```

---

## 📊 Step 4: Monitor Publishing Progress

### 4.1 Watch Workflows

1. Go to GitHub repository
2. Click **Actions** tab
3. You'll see 4-5 workflows running:
   - ✅ "Release - Publish to All Registries" (master coordinator)
   - ✅ "Publish to crates.io"
   - ✅ "Publish to PyPI"
   - ✅ "Publish to npm"
   - ✅ "Publish to CRAN (R)" (manual submission)

### 4.2 Expected Timeline

```
T+0:00   Tag pushed
         ↓
T+0:05   GitHub Release created
         Release workflows triggered
         ↓
T+0:10   Rust, Python, JavaScript publishing complete
         ↓
T+0:15   All 3 auto-registries show package
         (check crates.io, PyPI, npm)
         ↓
T+0:30   R package ready (download from GitHub releases)
         Manual CRAN submission instructions provided
         ↓
T+3-7d   CRAN reviews and publishes R package
         (if no issues found)
```

### 4.3 Verify Publications

#### Check crates.io
```bash
# Within 1-5 minutes
cargo search ferromode
# Expected: ferromode = "1.0.0"

# Or visit:
# https://crates.io/crates/ferromode/1.0.0
```

#### Check PyPI
```bash
# Within 1-5 minutes
pip index versions ferromode
# Expected: ferromode 1.0.0

# Or visit:
# https://pypi.org/project/ferromode/1.0.0/
```

#### Check npm
```bash
# Within 1-5 minutes
npm view @ferromode/ferromode version
# Expected: 1.0.0

# Or visit:
# https://www.npmjs.com/package/@ferromode/ferromode
```

#### Check CRAN (R) - Manual Step
R package requires manual CRAN submission (see Step 5)

---

## 📦 Step 5: CRAN Submission (R Package)

### 5.1 Download R Package

1. Go to GitHub releases
2. Find v1.0.0 release
3. Download `ferromode_*.tar.gz`

### 5.2 Prepare CRAN Submission

Before submitting, verify locally (if you have R installed):

```bash
# Download tarball from release
cd /tmp
wget https://github.com/ferromode/ferromode/releases/download/v1.0.0/ferromode_1.0.0.tar.gz

# Check package (requires R)
R CMD check ferromode_1.0.0.tar.gz

# Expected: 0 errors, 0 warnings
```

### 5.3 Submit to CRAN

1. Visit: https://cran.r-project.org/submit.html
2. Select "Upload package"
3. Upload the `ferromode_*.tar.gz` file
4. Fill submission form:
   ```
   Title: Ferromode - Advanced Signal Processing for Empirical Mode Decomposition
   
   Comments: R bindings for the Ferromode library via Rust/extendr.
   This package enables high-performance signal decomposition 
   with neural network-based boundary prediction.
   ```
5. Click "Submit"

### 5.4 Monitor CRAN Review

- Timeline: Typically 3-7 days
- Check email for CRAN feedback
- If issues found, address and resubmit
- Once accepted, automatically published to CRAN

---

## ✅ Step 6: Post-Release Verification

### 6.1 Install and Test from Each Registry

```bash
# Rust
cargo new --bin test_ferromode
cd test_ferromode
cargo add ferromode@1.0.0
cargo build

# Python
python -m venv test_venv
source test_venv/bin/activate
pip install ferromode==1.0.0
python -c "import ferromode; print(ferromode.__version__)"

# JavaScript
mkdir test_js && cd test_js
npm init -y
npm install @ferromode/ferromode@1.0.0
node -e "const fm = require('@ferromode/ferromode'); console.log('OK')"

# R (after CRAN acceptance)
R -e "install.packages('ferromode')"
```

### 6.2 Announce Release

1. Create release announcement on GitHub (template below)
2. Update project README with version
3. Announce on social media/forums
4. Update documentation links

**Release Announcement Template:**

```markdown
# Ferromode v1.0.0 Released! 🎉

We're excited to announce **Ferromode v1.0.0** — the first stable release!

## What's New

- **Multi-platform support**: Rust, Python, JavaScript, R
- **Neural boundary prediction**: LSTM models for improved end-effect handling
- **GPU acceleration**: NVIDIA CUDA, AMD ROCm, Apple Metal support
- **Production-ready**: Extensive testing, documentation, examples

## Installation

### Rust
\`\`\`bash
cargo add ferromode@1.0.0
\`\`\`

### Python
\`\`\`bash
pip install ferromode==1.0.0
\`\`\`

### JavaScript
\`\`\`bash
npm install @ferromode/ferromode@1.0.0
\`\`\`

### R (after CRAN acceptance)
\`\`\`r
install.packages('ferromode')
\`\`\`

## Documentation

- [User Guide](docs/)
- [API Reference](https://docs.rs/ferromode)
- [Examples](examples/)
- [Training Guide](TRAINING_SETUP.md)

## Thank You

Thanks to all contributors who made this release possible!
```

---

## 🔧 Troubleshooting Publishing Issues

### ❌ Workflow Failed: Token Invalid

**Problem:** Publishing workflow fails with authentication error

**Solution:**
1. Check secret name: `PYPI_TOKEN`, `CARGO_TOKEN`, `NPM_TOKEN`
2. Verify token is correct (copy from registry account page)
3. Ensure no extra spaces in token value
4. Regenerate token if unsure

### ❌ Workflow Failed: Version Mismatch

**Problem:** Workflow fails because version in Cargo.toml doesn't match tag

**Solution:**
```bash
# Verify all versions match tag
TAG="v1.0.0"
VERSION="${TAG#v}"  # 1.0.0

grep "^version" crates/ferromode/Cargo.toml | grep "$VERSION"
grep "^version" crates/ferromode-py/Cargo.toml | grep "$VERSION"
```

### ❌ PyPI Upload Fails: File Already Exists

**Problem:** PyPI rejects wheel because it already exists

**Solution:**
```bash
# Don't re-upload same version twice
# If needed, create new patch version and re-tag
git tag -d v1.0.0
git tag -a v1.0.1 -m "Patch release"
git push origin v1.0.1 --force-with-lease
```

### ❌ CRAN Takes Longer Than Expected

**Problem:** CRAN hasn't published after 1 week

**Solution:**
1. Check email for CRAN feedback
2. If rejected, fix issues and resubmit
3. CRAN volunteers review — timeline varies
4. Follow up if needed: cran-submissions@r-project.org

### ❌ npm Publishing Fails: Registry Auth

**Problem:** npm workflow fails with authentication error

**Solution:**
1. Check `NPM_TOKEN` secret is set
2. Verify token is for correct npm account
3. Regenerate token if unsure

---

## 📋 Release Checklist

Before publishing, verify:

```markdown
## Pre-Release Checklist

- [ ] All tests passing locally
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] All versions synchronized (1.0.0)
- [ ] No uncommitted changes
- [ ] Publishing tokens configured in GitHub Secrets
- [ ] Release branch is up-to-date with main

## Publishing Checklist

- [ ] Git tag created (v1.0.0)
- [ ] Tag pushed to GitHub
- [ ] All workflows triggered and monitoring
- [ ] crates.io shows package within 5 minutes
- [ ] PyPI shows package within 5 minutes
- [ ] npm shows package within 5 minutes
- [ ] GitHub Release created automatically
- [ ] R package workflow completed

## Post-Release Checklist

- [ ] Verified installation from each registry
- [ ] Updated project README
- [ ] Announced release
- [ ] Created CHANGELOG entry
- [ ] All systems stable

## CRAN Submission (R)

- [ ] Downloaded ferromode_*.tar.gz from release
- [ ] Verified locally (R CMD check)
- [ ] Submitted to CRAN
- [ ] Monitoring CRAN review
- [ ] (Eventually) Published to CRAN
```

---

## 🔐 Security Notes

### Token Safety

- **Never commit tokens** to git
- **Never hardcode tokens** in workflows
- Use GitHub Secrets exclusively
- Regenerate tokens after exposure
- Rotate tokens periodically (annually)

### Release Integrity

- Sign git tags (optional but recommended):
  ```bash
  git tag -s v1.0.0 -m "Release v1.0.0"
  ```

- Verify tag before pushing:
  ```bash
  git tag -v v1.0.0  # Check signature
  ```

### Access Control

- Restrict who can create releases (GitHub Settings → Branch Protection)
- Require reviews before merging release commits
- Audit workflow logs for suspicious activity

---

## 📞 Support

If you encounter issues during publishing:

1. **Check workflow logs** — GitHub Actions shows detailed error messages
2. **Review troubleshooting** section above
3. **Check token configuration** — Most issues are auth-related
4. **Test locally first** — Run tests before triggering release workflow
5. **Open GitHub issue** — Include workflow log if stuck

---

## 📚 References

- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [crates.io Publishing](https://doc.rust-lang.org/cargo/references/publishing.html)
- [PyPI Publishing](https://packaging.python.org/en/latest/tutorials/packaging-project/)
- [npm Publishing](https://docs.npmjs.com/packages-and-modules/publishing-packages-and-modules/)
- [CRAN Submission](https://cran.r-project.org/submit.html)

---

## Summary

**Quick Start:**

1. Update versions to 1.0.0
2. Create git tag: `git tag -a v1.0.0 -m "Release v1.0.0"`
3. Push tag: `git push origin v1.0.0`
4. Watch workflows complete (5-10 minutes)
5. Verify on crates.io, PyPI, npm
6. Submit R package to CRAN manually

**Total time:** ~15 minutes for automation + 3-7 days for CRAN review

**Result:** Ferromode v1.0.0 published across 4 major registries! 🚀
