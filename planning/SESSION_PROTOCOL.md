# Session Protocol
## Ferromode Project — Multi-Agent Session Management & Execution Tracking

**Created:** 2026-04-02  
**Status:** Active  
**Linked Docs:** [EXECUTION_CHECKLIST.md](./EXECUTION_CHECKLIST.md) · [IMPLEMENTATION_MAP.md](./IMPLEMENTATION_MAP.md)

---

## Overview

This protocol defines how each agent mode (architect, code, test, review, debug, docs) updates the session file and execution checklist during task execution. It ensures traceability, accountability, and continuity across mode switches.

---

## Session File Structure

Each session file lives in `.promptosaurus/sessions/` and follows this format:

```yaml
---
session_id: "session_20260402_xyz123"
branch: "feat/PROJ-001-ferromode-foundation"
created_at: "2026-04-02T10:00:00Z"
current_mode: "code"
version: "1.0"
---

## Session Overview

**Branch:** feat/PROJ-001-ferromode-foundation  
**Started:** 2026-04-02 10:00 UTC  
**Current Mode:** code  
**Current Task:** T-008 — Define Signal struct  
**Current Milestone:** M0 → M1

## Mode History

| Mode | Entered | Exited | Task | Summary |
|------|---------|--------|------|---------|
| architect | 10:00 | 10:30 | T-008 | Defined acceptance criteria for Signal struct |
| code | 10:30 | - | T-008 | Writing tests then implementation |

## Execution Checklist Updates

| Timestamp | Task | Old Status | New Status | Mode | Notes |
|-----------|------|------------|------------|------|-------|
| 10:00 | T-008 | 🔲 TODO | 🔄 IN_PROGRESS | architect | Acceptance criteria defined |
| 10:30 | T-008 | 🔄 IN_PROGRESS | 🔄 IN_PROGRESS | code | Tests written (RED), implementing now |

## Actions Taken

### 2026-04-02 10:00 - architect mode
- **Task:** T-008 — Define Signal struct
- **Acceptance Criteria:**
  - Given a slice of f64 values, When Signal::from_slice is called, Then the signal stores the values and reports correct length
  - Given an empty slice, When Signal::from_slice is called, Then EmdError::EmptySignal is returned
  - Given a Signal, When .iter() is called, Then it yields the same values as the original slice
- **DDD Analysis:** Signal is a Value Object in the Signal Domain bounded context. It enforces the invariant that length > 0.
- **SOLID Analysis:** Single responsibility — represents signal data. Open for extension via methods.
- **Files to modify:** `crates/ferromode/src/types.rs` (new), `crates/ferromode/src/lib.rs` (mod declaration)

### 2026-04-02 10:30 - code mode
- **Task:** T-008 — Define Signal struct
- **TDD Cycle:**
  - **RED:** Wrote `test_signal_from_slice_returns_signal` — fails because Signal type doesn't exist
  - **RED:** Wrote `test_signal_from_empty_slice_returns_error` — fails to compile
  - **GREEN:** Created Signal struct with from_slice, len, iter methods
  - **GREEN:** Both tests pass
  - **REFACTOR:** Extracted validation into Signal::validate() private method
- **Files created:** `crates/ferromode/src/types.rs` (45 LOC)
- **Files modified:** `crates/ferromode/src/lib.rs` (added mod types)
- **Clean Code:** Extracted validate() to keep from_slice under 10 lines
- **Clean Architecture:** types.rs is in the domain layer — no dependencies on algorithms or bindings

## Context Summary

Working on M1: Spline & Boundary Foundation. Currently implementing core type system (T-008). Signal struct complete, moving to MultivariateSignal (T-009).

## Engineering Practice Compliance

| Task | TDD | ATDD | DDD | Clean Code | Clean Arch | SOLID | Status |
|------|-----|------|-----|------------|------------|-------|--------|
| T-008 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🔍 REVIEW |

## Notes

- Waiting for user review of Signal struct API before proceeding to T-009
- Consider adding sample_rate validation in a follow-up task
```

---

## Mode-Specific Session Update Requirements

### Architect Mode

**When entering:**
1. Read the task description from EXECUTION_CHECKLIST.md
2. Read the IMPLEMENTATION_MAP.md for wave dependencies
3. Read any existing session file for context

**Must record in session:**
- [ ] Acceptance criteria in Given-When-Then format (ATDD)
- [ ] Domain model alignment analysis (DDD) — which bounded context does this touch?
- [ ] SOLID compliance analysis — which principles are relevant?
- [ ] Clean Architecture layer identification — which layer does this belong to?
- [ ] List of files that will need to change
- [ ] Any design trade-offs or decisions needed
- [ ] Update EXECUTION_CHECKLIST.md: change task status from 🔲 to 🔄, set mode to `architect`

**Must NOT do:**
- Write implementation code
- Write tests
- Modify any file outside the target module

**Example session entry:**
```markdown
### 2026-04-02 10:00 - architect mode
- **Task:** T-026 — Define BoundaryCondition trait
- **Acceptance Criteria:**
  - Given a BoundaryCondition trait, When a struct implements it, Then extend() returns an ExtendedSignal
  - Given any implementation, When extend() is called, Then the extended signal contains the original signal
- **DDD:** Boundary Domain bounded context. Trait is the abstraction point.
- **SOLID-OCP:** Open for new boundary strategies, closed for modification.
- **SOLID-DIP:** Sifting engine depends on BoundaryCondition trait, not concrete implementations.
- **Clean Architecture:** Domain layer — no dependencies on algorithms, bindings, or external crates beyond std.
- **Files to create:** `crates/ferromode/src/boundary/mod.rs`, `crates/ferromode/src/boundary/traits.rs`
- **Files to modify:** `crates/ferromode/src/lib.rs`
```

### Code Mode

**When entering:**
1. Read the architect's acceptance criteria from the session file
2. Read the target module's existing code
3. Read any related tests

**Must record in session:**
- [ ] TDD cycle notes: what test was written first (RED), what code made it pass (GREEN), what refactoring was done (REFACTOR)
- [ ] Files created or modified with line counts
- [ ] Clean Code decisions — why a function was extracted, why a name was chosen
- [ ] Clean Architecture verification — confirming no layer violations
- [ ] Any deviations from the architect's plan and why
- [ ] Update EXECUTION_CHECKLIST.md: set mode to `code`
- [ ] Git commit hash and message (one commit per task, conventional commit format with task ID)

**Must NOT do:**
- Skip writing tests first
- Add algorithm logic to binding layers
- Introduce new dependencies without flagging
- Modify files outside the target module without asking

**Example session entry:**
```markdown
### 2026-04-02 10:30 - code mode
- **Task:** T-026 — Define BoundaryCondition trait
- **TDD Cycle:**
  - **RED:** Wrote `test_boundary_trait_can_be_implemented` — fails: trait doesn't exist
  - **GREEN:** Created BoundaryCondition trait with extend() method
  - **GREEN:** Test passes
  - **REFACTOR:** Added associated type ExtendedSignal for type safety
- **Files created:** `crates/ferromode/src/boundary/mod.rs` (12 LOC), `crates/ferromode/src/boundary/traits.rs` (18 LOC)
- **Clean Code:** Trait is 5 lines — minimal and focused (SOLID-I)
- **Clean Architecture:** Domain layer — no external dependencies
```

### Test Mode

**When entering:**
1. Read the architect's acceptance criteria from the session file
2. Read the implementation code
3. Read any existing tests

**Must record in session:**
- [ ] Coverage results: line %, branch %, function %
- [ ] Test count: total tests, passing, failing
- [ ] Edge cases tested — list each one
- [ ] Property-based test results (if applicable)
- [ ] Acceptance test status — do all Given-When-Then criteria pass?
- [ ] Any gaps in coverage and recommended follow-up tests
- [ ] Update EXECUTION_CHECKLIST.md: set mode to `test`

**Must NOT do:**
- Modify implementation code
- Lower coverage targets
- Mark acceptance tests as passing if they are not

**Example session entry:**
```markdown
### 2026-04-02 11:00 - test mode
- **Task:** T-026 — Define BoundaryCondition trait
- **Coverage:** 100% line, 100% branch, 100% function (trait has no branches)
- **Tests:** 3 total, 3 passing
  - `test_boundary_trait_can_be_implemented` — verifies trait contract
  - `test_extended_signal_contains_original` — verifies invariant
  - `test_extend_does_not_mutate_original` — verifies immutability
- **Acceptance Tests:** Both Given-When-Then criteria pass
- **Edge Cases:** Empty signal, single-sample signal, constant signal
- **Gaps:** None identified
```

### Review Mode

**When entering:**
1. Read the full session file for the task
2. Read the implementation code
3. Read all tests
4. Run `cargo test` and `cargo clippy`

**Must record in session:**
- [ ] Fill out the Compliance Checklist (all 16 items) from IMPLEMENTATION_MAP.md
- [ ] List any blockers found
- [ ] List any suggestions (non-blocking)
- [ ] Verdict: ✅ APPROVED or 🔄 NEEDS CHANGES
- [ ] If NEEDS CHANGES: specify exactly what must be fixed
- [ ] Update EXECUTION_CHECKLIST.md: set status to ✅ DONE or back to 🔄 IN_PROGRESS (code)

**Must NOT do:**
- Approve if any compliance item is unchecked
- Approve if any test is failing
- Approve if clippy has warnings

**Example session entry:**
```markdown
### 2026-04-02 11:30 - review mode
- **Task:** T-026 — Define BoundaryCondition trait
- **Compliance Checklist:**
  - [x] TDD: Tests were written before implementation
  - [x] TDD: Red-Green-Refactor cycle was followed
  - [x] ATDD: Acceptance criteria were defined before implementation
  - [x] ATDD: Acceptance tests pass
  - [x] DDD: Domain types enforce their own invariants
  - [x] DDD: No anemic models — trait encapsulates behavior contract
  - [x] Clean Code: Trait name is descriptive and intent-revealing
  - [x] Clean Code: Trait is 5 lines — well under 20-line limit
  - [x] Clean Code: No duplicated code
  - [x] Clean Code: All clippy warnings resolved
  - [x] Clean Architecture: Dependencies point inward (domain layer)
  - [x] Clean Architecture: No domain logic in binding layer
  - [x] SOLID-S: Trait has single responsibility (signal extension)
  - [x] SOLID-O: Open for new implementations, closed for modification
  - [x] SOLID-L: Any implementation is substitutable in sifting engine
  - [x] SOLID-I: Interface is minimal — one method
  - [x] SOLID-D: Sifting engine depends on trait, not concretions
- **Verdict:** ✅ APPROVED
- **Notes:** Clean implementation. Consider adding doc example in follow-up.
```

### Debug Mode

**When entering:**
1. Read the error report or failing test
2. Read the relevant code
3. Read the session file for context

**Must record in session:**
- [ ] Symptom description
- [ ] Root cause analysis
- [ ] Fix approach
- [ ] Regression test added (TDD — write failing test that reproduces the bug)
- [ ] Update EXECUTION_CHECKLIST.md if a new task is needed

**Example session entry:**
```markdown
### 2026-04-02 14:00 - debug mode
- **Symptom:** `test_signal_from_empty_slice_returns_error` panics instead of returning Err
- **Root Cause:** Signal::from_slice uses unwrap() on validation result instead of ?
- **Fix:** Changed unwrap() to map_err() with proper error conversion
- **Regression Test:** Added `test_signal_from_slice_panics_on_nan` to catch similar issues
- **Files modified:** `crates/ferromode/src/types.rs` (line 12)
```

### Docs Mode

**When entering:**
1. Read the implementation code to understand what needs documenting
2. Read existing documentation for style consistency
3. Read the task description from EXECUTION_CHECKLIST.md

**Must record in session:**
- [ ] Documentation files created or updated
- [ ] Cross-references added
- [ ] Any gaps in existing documentation identified
- [ ] Update EXECUTION_CHECKLIST.md: set mode to `docs`

**Example session entry:**
```markdown
### 2026-04-02 15:00 - docs mode
- **Task:** T-038 — Add citation comment for Zeng & He 2004
- **Files modified:** `crates/ferromode/src/boundary/periodic.rs` (added doc comment with citation)
- **Cross-references:** Updated `docs/algorithms/boundary_conditions.md` to include Zeng & He reference
```

---

## Execution Checklist Update Protocol

When a mode changes a task's status, it must update `docs/EXECUTION_CHECKLIST.md`:

1. Find the task row in the table
2. Update the Status column:
   - 🔲 → 🔄 when architect starts
   - 🔄 stays 🔄 when code/test/debug works on it
   - 🔄 → 🔍 when test mode completes with passing coverage
   - 🔍 → ✅ when review mode approves
   - 🔍 → 🔄 when review mode finds issues (back to code)
3. Update the Mode column to the current mode
4. Update the Session Ref column with the session file name or timestamp
5. Add notes if relevant (blockers, decisions, follow-ups)

---

## Mode Handoff Checklist

When switching from one mode to another, the outgoing mode must:

### Architect → Code
- [ ] Acceptance criteria are written in Given-When-Then format
- [ ] Domain model analysis is complete
- [ ] SOLID analysis are documented
- [ ] Target files are identified
- [ ] Session file is updated with architect entry

### Code → Test
- [ ] All tests pass (`cargo test`)
- [ ] Clippy has no warnings (`cargo clippy`)
- [ ] TDD cycle is documented in session
- [ ] Files created/modified are listed
- [ ] Session file is updated with code entry
- [ ] Task is committed with conventional commit message including task ID (one commit per task)

### Test → Review
- [ ] Coverage meets targets (80% line, 70% branch, 90% function)
- [ ] Acceptance tests pass
- [ ] Edge cases are documented
- [ ] Coverage results are in session file
- [ ] Session file is updated with test entry

### Review → Done (or back to Code)
- [ ] Compliance checklist is filled out (all 16 items)
- [ ] Verdict is recorded
- [ ] If approved: EXECUTION_CHECKLIST.md updated to ✅ DONE
- [ ] If rejected: specific issues are listed, EXECUTION_CHECKLIST.md updated to 🔄
- [ ] Session file is updated with review entry
- [ ] Commit message follows conventional commit format with task ID
- [ ] One commit per task rule was followed (no bundled tasks)

---

## Recovery Protocol

If a session is interrupted (context switch, mode change, interruption):

1. Read the session file
2. Check the `current_mode` field
3. Read the Mode History to see where work left off
4. Read the Actions Taken to understand what was done
5. Read the Context Summary for the current state
6. Resume from the last action

If the session file is missing or corrupted:
1. Check git status for uncommitted changes
2. Check which files were recently modified
3. Create a new session file with current state
4. Note in the session that recovery was performed

---

## Multi-Task Session Management

When working on multiple tasks in one session:

1. Each task gets its own Actions Taken entry
2. The Execution Checklist Updates table tracks all task status changes
3. The Context Summary reflects the state of all in-progress tasks
4. Mode History records which task was being worked on in each mode

Example:
```markdown
## Mode History

| Mode | Entered | Exited | Task | Summary |
|------|---------|--------|------|---------|
| architect | 10:00 | 10:30 | T-008 | Signal struct acceptance criteria |
| code | 10:30 | 11:00 | T-008 | Signal struct implementation |
| test | 11:00 | 11:30 | T-008 | Signal struct tests — 100% coverage |
| review | 11:30 | 11:45 | T-008 | Approved ✅ |
| architect | 11:45 | 12:00 | T-009 | MultivariateSignal acceptance criteria |
| code | 12:00 | - | T-009 | In progress |
```

---

## Quick Reference Card

| Action | What to Update | Where |
|--------|---------------|-------|
| Start a task | Status: 🔲 → 🔄, Mode: architect | EXECUTION_CHECKLIST.md + session file |
| Write tests first | TDD cycle: RED | Session file → Actions Taken |
| Make tests pass | TDD cycle: GREEN | Session file → Actions Taken |
| Refactor | TDD cycle: REFACTOR | Session file → Actions Taken |
| Finish implementation | Mode: code → test | Session file + EXECUTION_CHECKLIST.md |
| Coverage complete | Status: 🔄 → 🔍, Mode: test | EXECUTION_CHECKLIST.md + session file |
| Review complete | Status: 🔍 → ✅ or 🔄 | EXECUTION_CHECKLIST.md + session file |
| Switch modes | Update current_mode, add Mode History entry | Session file |
| End session | Update Context Summary | Session file |
| Complete a task | Conventional commit with task ID, one commit per task | git commit message + session file |
