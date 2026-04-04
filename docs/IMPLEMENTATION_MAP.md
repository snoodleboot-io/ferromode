# Implementation Map
## Ferromode Project — Execution Strategy & Engineering Practices

**Created:** 2026-04-02  
**Status:** Active  
**Linked Docs:** [EXECUTION_CHECKLIST.md](./EXECUTION_CHECKLIST.md) · [SESSION_PROTOCOL.md](./SESSION_PROTOCOL.md)

---

## Engineering Practices — Non-Negotiable Standards

Every task in the execution checklist must satisfy ALL six practices before being marked DONE.

### 1. TDD (Test-Driven Development)

**Rule:** Tests are written BEFORE implementation code. The red-green-refactor cycle is mandatory.

**Workflow per task:**
1. **RED** — Write a failing test that expresses the desired behavior
2. **GREEN** — Write the minimum code to make the test pass
3. **REFACTOR** — Clean up the code while keeping tests green

**Enforcement in CI:**
- `cargo test` must pass before any merge
- New code without corresponding new tests is rejected
- Coverage minimum: 80% line, 70% branch, 90% function

**Session file update:** Code mode records which tests were written first, the red state, and the green state.

### 2. ATDD (Acceptance Test-Driven Development)

**Rule:** Acceptance criteria are defined as executable tests BEFORE the story is started.

**Workflow per story:**
1. Product owner / architect defines acceptance criteria in Given-When-Then format
2. Acceptance tests are written (may initially be marked `#[ignore]` or stubbed)
3. Implementation proceeds only after acceptance tests exist
4. Story is not done until acceptance tests pass

**Example acceptance test format:**
```rust
// Given a signal with known properties
// When decomposed with EMD
// Then the sum of IMFs + residue equals the original signal within 1e-12
#[test]
fn test_emd_reconstruction_preserves_signal() { ... }
```

**Session file update:** Architect mode records acceptance criteria; Test mode records acceptance test creation; Code mode records when acceptance tests pass.

### 3. DDD (Domain-Driven Design)

**Rule:** The Rust core models the EMD domain explicitly. No anemic data structures — types encapsulate behavior.

**Bounded Contexts:**
| Context | Module | Responsibility |
|---------|--------|----------------|
| **Signal Domain** | `types.rs` | `Signal`, `MultivariateSignal`, `ImfCollection` — rich types with validation invariants |
| **Algorithm Domain** | `algorithms/` | EMD, EEMD, CEEMDAN, MEMD, VMD — each algorithm is a distinct service |
| **Boundary Domain** | `boundary/` | Trait-based strategy pattern — each boundary condition is a distinct implementation |
| **Sifting Domain** | `sifting/` | Sifting engine and stopping criteria — the core iterative process |
| **Analysis Domain** | `hilbert/` | Hilbert transform, instantaneous frequency, marginal spectrum |
| **Configuration Domain** | `config/` | `EmdConfig`, `EnsembleConfig`, `MemdConfig` — value objects with validation |
| **Binding Layer** | `api.rs`, `ffi.rs` | Pure marshalling — zero domain logic |

**DDD Rules:**
- Domain types enforce their own invariants (e.g., `Signal::from_slice` rejects empty input)
- Value objects are immutable and equality-based
- Entities have identity (e.g., `DecompositionResult` has a unique run ID)
- Repositories are not needed — the domain is computational, not persistent
- Anti-corruption layer: the FFI boundary (`ffi.rs`) is the anti-corruption layer between domain and external languages

**Session file update:** Architect mode validates domain model alignment; Code mode notes any domain model changes.

### 4. Clean Code

**Rule:** Code must be readable, maintainable, and self-documenting.

**Standards:**
- **Naming:** Descriptive names — `compute_mean_envelope()` not `calc_me()`
- **Functions:** Max 20 lines; single responsibility; no side effects unless named for them
- **Comments:** Explain WHY, not WHAT. No commented-out code.
- **Error handling:** Typed errors with context. Never `panic!()` in library code.
- **No duplication:** DRY — extract common logic into named functions
- **Boy Scout Rule:** Leave the code cleaner than you found it

**Rust-specific:**
- Use `clippy` — all warnings must be resolved
- Use `rustfmt` — formatting is non-negotiable
- Prefer `match` over nested `if-let`
- Use `Result<T, E>` — never swallow errors
- Derive `Debug`, `Clone`, `Default` where appropriate
- Document all public items with `///` rustdoc comments

**Session file update:** Review mode flags Clean Code violations; Code mode resolves them before merge.

### 5. Clean Architecture

**Rule:** Dependencies point inward. The domain knows nothing about frameworks, bindings, or external concerns.

**Layer structure (innermost → outermost):**
```
┌─────────────────────────────────────────────────┐
│  Frameworks & Drivers (Outer)                    │
│  ├── PyO3 bindings (ferromode-py)               │
│  ├── extendr bindings (ferromode-r)             │
│  ├── WASM bindings (ferromode-wasm)             │
│  ├── Julia ccall (Ferromode.jl)                 │
│  ├── MEX bindings (ferromode-mex)               │
│  └── C++ headers (ferromode-cxx)                │
├─────────────────────────────────────────────────┤
│  Interface Adapters                              │
│  ├── api.rs — Public Rust API                   │
│  └── ffi.rs — C-ABI exports                     │
├─────────────────────────────────────────────────┤
│  Use Cases (Application Business Rules)          │
│  ├── algorithms/emd.rs — EMD decomposition      │
│  ├── algorithms/ceemdan.rs — CEEMDAN            │
│  └── algorithms/memd.rs — MEMD                  │
├─────────────────────────────────────────────────┤
│  Entities (Enterprise Business Rules) — DOMAIN   │
│  ├── types.rs — Signal, ImfCollection, etc.     │
│  ├── boundary/ — BoundaryCondition trait        │
│  ├── sifting/ — SiftingEngine                   │
│  └── error.rs — EmdError                        │
└─────────────────────────────────────────────────┘
```

**Dependency Rule:** Source code dependencies must point only inward, toward higher-level policies.
- Domain types (`types.rs`) import nothing from algorithms or bindings
- Algorithms import from domain types and boundary strategies
- API layer imports from algorithms and domain
- Bindings import ONLY from API layer — never from domain or algorithms directly

**Enforcement:**
- `cargo clippy` with custom lint rules
- Code review checks for dependency direction violations
- Any binding that contains algorithm logic is a critical violation

**Session file update:** Architect mode validates layer boundaries; Review mode enforces dependency direction.

### 6. SOLID OOP (Applied to Rust Traits & Modules)

**S — Single Responsibility Principle**
- Each module has one reason to change
- `sifting/engine.rs` handles sifting loops — not envelope computation
- `boundary/` module handles boundary strategies — not spline interpolation
- Each boundary strategy is its own file

**O — Open/Closed Principle**
- `BoundaryCondition` trait is open for extension, closed for modification
- New boundary strategies implement the trait — no changes to existing code
- `StoppingCriterion` enum is open for new variants via pattern matching

**L — Liskov Substitution Principle**
- Any `BoundaryCondition` implementation can be substituted without breaking the sifting engine
- All boundary strategies must satisfy the trait contract: return an extended signal containing the original

**I — Interface Segregation Principle**
- `BoundaryCondition` trait has one method: `extend()`
- No fat interfaces — traits are focused and minimal
- Algorithm traits (if any) are separated from configuration traits

**D — Dependency Inversion Principle**
- Algorithms depend on `BoundaryCondition` trait, not concrete implementations
- Sifting engine depends on `Spline` trait, not a specific spline implementation
- High-level `emd()` function depends on abstractions, not concrete boundary or spline types

**Session file update:** Architect mode validates SOLID compliance during design; Review mode checks during code review.

### 7. One Commit Per Task

**Rule:** Every task (T-001 through T-237) is committed as a separate, atomic git commit. No task's changes are bundled with another task's changes.

**Commit message format (Conventional Commits):**
```
<type>(<scope>): <description> — T-XXX

<body>

- Acceptance criteria: <list>
- Tests: <count> tests added/modified
- Files: <list of changed files>
```

**Type mapping:**
| Task Type | Commit Type |
|-----------|-------------|
| New algorithm/feature | `feat` |
| Bug fix | `fix` |
| Test-only task | `test` |
| Documentation task | `docs` |
| Build/CI/config task | `ci` |
| Refactoring | `refactor` |
| Dependency update | `chore` |

**Scope mapping:**
| Epic | Scope |
|------|-------|
| Epic 1: Rust Core | `core` |
| Epic 2: Python | `py` |
| Epic 3: R | `r` |
| Epic 4: Julia | `julia` |
| Epic 5: JS/TS | `js` |
| Epic 6: Docs/Validation | `docs` or `validation` |
| Epic 7: MATLAB | `mex` |
| Epic 8: C++ | `cxx` |

**Sub-scope (module-level):**
| Module | Sub-scope |
|--------|-----------|
| types.rs | `types` |
| error.rs | `error` |
| algorithms/emd.rs | `emd` |
| algorithms/ceemdan.rs | `ceemdan` |
| boundary/ | `boundary` |
| sifting/ | `sifting` |
| hilbert/ | `hilbert` |
| spline/ | `spline` |

**Examples:**
```
feat(core/types): add Signal struct with validation — T-008

- Acceptance criteria: from_slice returns Signal or EmdError, len() correct, iter() yields values
- Tests: 4 tests added
- Files: crates/ferromode/src/types.rs, crates/ferromode/src/lib.rs

test(core/spline): validate cubic spline against scipy reference — T-023

- Acceptance criteria: spline values match scipy within 1e-10 on all reference signals
- Tests: 12 tests added (uniform, non-uniform, edge cases)
- Files: crates/ferromode/src/spline/cubic.rs, crates/ferromode/tests/spline_reference.rs

docs(core/boundary): add Zeng & He 2004 citation — T-038

- Acceptance criteria: source file contains citation comment
- Tests: N/A
- Files: crates/ferromode/src/boundary/periodic.rs

ci(core): configure clippy project-wide lint rules — T-004

- Acceptance criteria: cargo clippy run with zero warnings on empty project
- Tests: N/A
- Files: clippy.toml, .github/workflows/ci.yml
```

**Enforcement rules:**
1. **One task = one commit.** Never combine multiple tasks into a single commit.
2. **No task spans multiple commits.** If a task requires multiple logical changes, they are squashed into one commit before the task moves to review.
3. **Commit message MUST include the task ID** (e.g., `T-008`) at the end of the subject line.
4. **Commit message MUST include acceptance criteria** in the body.
5. **Commit message MUST list files changed.**
6. **Review mode verifies** that the commit message matches the task description in EXECUTION_CHECKLIST.md.
7. **If a review sends a task back to code**, the fix is added as a fixup commit (`fixup! <original commit hash>`) and then squashed before re-review — the final history still shows one clean commit per task.

**Session file update:** Code mode records the commit hash and message. Review mode verifies the commit message format and task ID.

---

## Execution Waves

Tasks are grouped into execution waves. Tasks within a wave can be parallelized. Waves must complete in order.

### Wave 1: Foundation Setup (Week 1-2)
**Milestone:** M0  
**Parallelizable tasks:** T-001, T-002, T-003, T-004, T-005, T-006, T-007

```
T-001 ──┐
T-002 ──┤
T-003 ──┤
T-004 ──┼──► All must complete before Wave 2
T-005 ──┤
T-006 ──┤
T-007 ──┘
```

### Wave 2: Domain Types & Error Handling (Week 3-4)
**Milestone:** M1 (partial)  
**Dependency:** Wave 1 complete  
**Parallelizable within wave:**

```
Group A (Types):        Group B (Errors):       Group C (Spline):
T-008 ──► T-009         T-016                   T-020 ──► T-021
T-010 ──► T-011         T-018 ──► T-019         T-022 ──► T-023
T-012                   T-017 (deferred)        T-024
T-013 ──► T-014                                 T-025
T-015 (tests)
```

### Wave 3: Boundary Strategies (Week 5-8)
**Milestone:** M1 (completion)  
**Dependency:** Wave 2 complete (T-026 trait must exist)

```
T-026 (trait) ──► T-027 (enum/factory) ──► T-028 (test harness)
                      │
                      ├──► T-029, T-030, T-031 (Characteristic Wave)
                      ├──► T-032, T-033, T-034 (Mirror)
                      ├──► T-035, T-036, T-037, T-038 (Periodic)
                      ├──► T-039, T-040, T-041 (Slope)
                      ├──► T-042, T-043, T-044, T-045 (AR Model)
                      └──► T-046, T-047, T-048, T-049 (Waveform Match)
```

All boundary strategies can be implemented in parallel once the trait and test harness exist.

### Wave 4: Core EMD (Week 9-12)
**Milestone:** M2  
**Dependency:** Wave 3 complete (boundary strategies + spline must exist)

```
T-050, T-051, T-052, T-053 (Extrema) ──► T-054 (tests)
T-055 (sift_one) ──► T-056, T-057, T-058, T-059, T-060 (stopping criteria)
T-061, T-062, T-063, T-064 (Basic EMD) ──► T-065, T-066 (validation tests)
T-097, T-098, T-099, T-100, T-101 (Hilbert) ──► T-102 (validation)
```

### Wave 5: Ensemble Methods (Week 13-16)
**Milestone:** M3  
**Dependency:** Wave 4 complete (basic EMD must exist)

```
T-067, T-068, T-069, T-070, T-071 (EEMD) ──┐
T-072, T-073, T-074 (CEEMD)                ├──► Can run in parallel
T-075, T-076, T-077, T-078, T-079, T-080 (CEEMDAN)
T-081, T-082, T-083 (ICEEMDAN) ────────────┘
```

### Wave 6: Multivariate & VMD (Week 17-22)
**Milestone:** M4  
**Dependency:** Wave 5 complete

```
T-084, T-085, T-086, T-087, T-088 (Direction Sampling) ──► T-089, T-090, T-091, T-092, T-093 (MEMD)
T-094, T-095, T-096 (NA-MEMD) ──► depends on MEMD
T-103, T-104, T-105, T-106 (Post-Proc Metrics)
T-107, T-108, T-109, T-110, T-111, T-112 (VMD)
```

### Wave 7: Python Binding (Week 23-25)
**Milestone:** M5  
**Dependency:** Wave 6 complete (Rust core must have all algorithms)

```
T-113, T-114, T-115, T-116, T-117 (PyO3 setup)
T-118, T-119, T-120 (Input marshalling)
T-121, T-122, T-123 (Output marshalling)
T-124, T-125, T-126, T-127, T-128, T-129 (Function wrappers)
T-130, T-131, T-132, T-133 (Packaging)
T-184, T-185, T-186 (Reference signals) ──► T-187, T-188, T-189 (Cross-lang validation)
```

### Wave 8: R + Julia Bindings (Week 26-28)
**Milestone:** M6  
**Dependency:** Wave 7 complete (Python binding validates the approach)

```
R Binding:          Julia Binding:
T-134, T-135, T-136  T-149, T-150, T-151, T-152 (C-ABI)
T-137, T-138, T-139  T-153, T-154, T-155 (Scaffolding)
T-140, T-141, T-142  T-156, T-157, T-158, T-159, T-160, T-161 (ccall)
T-143, T-144, T-145  T-162, T-163, T-164 (Packaging)
T-146, T-147, T-148
```

R and Julia bindings can be developed in parallel.

### Wave 9: JS/TS + Documentation (Week 29-32)
**Milestone:** M7  
**Dependency:** Wave 8 complete

```
JS/TS Binding:            Documentation:
T-165, T-166, T-167, T-168  T-190, T-191, T-192, T-193 (Algo docs)
T-169, T-170, T-171         T-194, T-195, T-196, T-197 (API docs)
T-172, T-173, T-174, T-175
T-176, T-177, T-178
T-179, T-180, T-181, T-182, T-183
```

### Wave 10: MATLAB/Octave Binding (Week 33-36)
**Milestone:** M8  
**Dependency:** Wave 9 complete (C-ABI from Julia binding reused)

```
T-198, T-199, T-200, T-201 (MEX setup)
T-202, T-203, T-204, T-205 (Input marshalling)
T-206, T-207, T-208, T-209 (Output marshalling)
T-210, T-211, T-212, T-213, T-214 (Function wrappers)
T-215, T-216, T-217, T-218 (Octave compat + distribution)
```

### Wave 11: C++ Binding (Week 37-40)
**Milestone:** M9  
**Dependency:** Wave 10 complete (C-ABI header already exists)

```
T-219, T-220, T-221, T-222 (C header + build)
T-223, T-224, T-225, T-226, T-227, T-228, T-229, T-230 (C++ wrapper)
T-231, T-232, T-233 (cxx bridge)
T-234, T-235, T-236, T-237 (Testing + distribution)
```

---

## Per-Task Execution Flow

Every task follows this sequence. No shortcuts.

```
┌─────────────────────────────────────────────────────────────┐
│                    TASK EXECUTION FLOW                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. ARCHITECT MODE                                           │
│     ├── Read existing code in target module                  │
│     ├── Define acceptance criteria (ATDD)                    │
│     ├── Validate domain model alignment (DDD)               │
│     ├── Confirm SOLID compliance of design                   │
│     ├── Update session file with design decisions            │
│     └── Mark task 🔄 IN_PROGRESS (mode: architect)           │
│                                                              │
│  2. CODE MODE                                                │
│     ├── Read architect's acceptance criteria                 │
│     ├── Write FAILING test first (TDD — RED)                │
│     ├── Write minimum code to pass (TDD — GREEN)            │
│     ├── Refactor for Clean Code (TDD — REFACTOR)            │
│     ├── Verify Clean Architecture layer boundaries           │
│     ├── Run cargo test + cargo clippy                        │
│     ├── Update session file with implementation details      │
│     └── Mark task 🔄 IN_PROGRESS (mode: code)                │
│                                                              │
│  3. TEST MODE                                                │
│     ├── Verify test coverage meets targets                   │
│     ├── Verify acceptance tests pass (ATDD)                  │
│     ├── Run edge case tests                                  │
│     ├── Run property-based tests where applicable            │
│     ├── Update session file with coverage results            │
│     └── Mark task 🔍 REVIEW (if coverage passes)             │
│                                                              │
│  4. REVIEW MODE                                              │
│     ├── Check TDD compliance (tests written first?)          │
│     ├── Check ATDD compliance (acceptance criteria met?)     │
│     ├── Check DDD compliance (domain model correct?)         │
│     ├── Check Clean Code (naming, size, duplication?)        │
│     ├── Check Clean Architecture (dependency direction?)     │
│     ├── Check SOLID (SRP, OCP, LSP, ISP, DIP?)              │
│     ├── Update session file with review findings             │
│     └── Mark task ✅ DONE (if all pass) or 🔄 (if fixes needed)│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Session File Update Protocol

Each mode updates the session file with specific information. See [SESSION_PROTOCOL.md](./SESSION_PROTOCOL.md) for the complete protocol.

### Quick Reference

| Mode | Updates Session With |
|------|---------------------|
| **architect** | Acceptance criteria, domain model decisions, SOLID analysis, design trade-offs |
| **code** | Files created/modified, TDD cycle notes (red→green→refactor), Clean Code decisions |
| **test** | Coverage percentages, test counts, edge cases tested, property test results |
| **review** | Practice compliance checklist (TDD/ATDD/DDD/Clean Code/Clean Arch/SOLID), blockers, approvals |
| **debug** | Root cause analysis, fix approach, regression tests added |
| **docs** | Documentation files created/updated, cross-references added |

### Task State Transitions

```
🔲 TODO ──(architect)──► 🔄 IN_PROGRESS (architect)
                       ──(code)─────► 🔄 IN_PROGRESS (code)
                                    ──(test)─────► 🔍 REVIEW
                                                 ──(review)──► ✅ DONE
                                                              ──(review fail)──► 🔄 IN_PROGRESS (code)
```

---

## Compliance Checklist Template

When a task reaches REVIEW mode, the reviewer fills out this checklist:

```markdown
### Task T-XXX Compliance Review

- [ ] **TDD**: Tests were written before implementation
- [ ] **TDD**: Red-Green-Refactor cycle was followed
- [ ] **ATDD**: Acceptance criteria were defined before implementation
- [ ] **ATDD**: Acceptance tests pass
- [ ] **DDD**: Domain types enforce their own invariants
- [ ] **DDD**: No anemic models — types encapsulate behavior
- [ ] **Clean Code**: Function names are descriptive and intent-revealing
- [ ] **Clean Code**: No function exceeds 20 lines without justification
- [ ] **Clean Code**: No duplicated code
- [ ] **Clean Code**: All clippy warnings resolved
- [ ] **Clean Architecture**: Dependencies point inward
- [ ] **Clean Architecture**: No domain logic in binding layer
- [ ] **SOLID-S**: Module has single responsibility
- [ ] **SOLID-O**: Open for extension, closed for modification
- [ ] **SOLID-L**: Subtypes/trait implementations are substitutable
- [ ] **SOLID-I**: Interfaces are focused and minimal
- [ ] **SOLID-D**: Depends on abstractions, not concretions

**Verdict:** ✅ APPROVED / 🔄 NEEDS CHANGES
**Notes:** [specific findings]
```

---

## Risk Mitigation Through Engineering Practices

| Risk | Mitigation Through Practice |
|------|---------------------------|
| Cross-language numerical mismatch | **TDD**: Each algorithm has reference tests; **ATDD**: Acceptance criterion is numerical equivalence |
| Binding contains algorithm logic | **Clean Architecture**: Dependency direction check; **Review**: Explicit compliance checklist |
| Spline numerical instability | **TDD**: Reference tests against scipy; **DDD**: Spline is a trait with validated implementations |
| Contributor introduces breaking changes | **SOLID-OCP**: New features extend via traits; **TDD**: Existing tests catch regressions |
| Performance regression | **TDD**: Benchmark tests in CI; **Clean Code**: No unnecessary allocations in hot paths |
| Domain model becomes anemic | **DDD**: Regular domain model reviews; **SOLID-SRP**: Each type has clear responsibility |
| Test coverage drops | **TDD**: Tests written first; **CI**: Coverage gate at 80% |

---

## Quick Start for New Contributors

1. Read this document to understand the engineering practices
2. Read [EXECUTION_CHECKLIST.md](./EXECUTION_CHECKLIST.md) to see what needs to be done
3. Read [SESSION_PROTOCOL.md](./SESSION_PROTOCOL.md) to understand how to track work
4. Pick a 🔲 TODO task from the checklist
5. Follow the Per-Task Execution Flow
6. Update the session file at each step
7. Mark the task complete when the compliance checklist passes
