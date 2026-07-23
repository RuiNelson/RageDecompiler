# Agent guide

Instructions for automated contributors working in `RageDecompiler`.

## Purpose and boundaries

`RageDecompiler` provides Python tools for reverse-engineering 68000 ROM code:
recursive disassembly, C++ recompilation, symbol/map comparison, iterative
coverage expansion, data filtering, and speculative code discovery.
`../StreetsOfRageRecompilation` supplies the Streets of Rage ROM-local inputs
and receives generated output.

Keep generic analysis and generation behavior here. Keep game-specific labels,
addresses, manual-function lists, ROMs, generated C++, and manuscripts in the
recompilation repository. `../Genesis-Plus-GX` is an upstream, read-only
reference and must never be edited.

Before changing files, inspect repository status and preserve unrelated work.
When an output format changes, inspect every caller and generated consumer
before implementing it.

## Repository layout

| Path | Responsibility |
| --- | --- |
| `tools/main.py` | Unified command dispatcher |
| `tools/common/` | Shared parsing and representation helpers |
| `tools/disassembler/` | Recursive-descent 68000 disassembly |
| `tools/recompiler/` | C++ generation from ROM regions |
| `tools/label_diff/` | Assembly, label, and map comparisons |
| `tools/iterative_disasm/` | Repeated coverage-expansion workflow |
| `tools/remove_data_locations/` | Removal of known data blocks from assembly |
| `tools/speculative_scan/` | Candidate discovery in unmapped ROM regions |
| `tools/tests/` and `tools/*/tests/` | Shared and feature-level tests |
| `tools/UserManual.md` | CLI and workflow documentation |

## Command surface

Use the unified module entry point:

```bash
python3 -m tools --help
python3 -m tools <command> --help
```

From `../StreetsOfRageRecompilation`, expose this sibling checkout:

```bash
PYTHONPATH=../RageDecompiler python3 -m tools --help
```

Prefer this surface over legacy direct module invocations. Preserve command
names, exit behavior, deterministic ordering, and established output formats
unless the requested change explicitly includes a migration.

## Tests

Install pytest in the active Python environment if needed, then run:

```bash
python3 -m pytest
```

Tests that use Streets of Rage fixtures default to the sibling checkout. Point
them elsewhere with:

```bash
SOR_RECOMPILATION_DIR=/path/to/StreetsOfRageRecompilation \
  python3 -m pytest
```

During development, run the smallest affected test first, followed by the full
suite. For generator changes, also regenerate into a temporary directory or a
known consumer and compare the output intentionally. Do not overwrite
checked-in consumer output merely to discover what changed.

## Analysis model

- Static disassembly follows reachable paths from known entry points.
- Indirect calls and jumps are not always resolvable statically.
- Auxiliary addresses and runtime active-disassembly data close those gaps.
- Speculative scanning proposes candidates; it is not equivalent to confirmed
  control flow.
- Labels, block boundaries, and manual-function lists are consumer inputs and
  should remain separate from generic tool policy.

Keep address-width, endianness, signedness, instruction size, and control-flow
fallthrough explicit in code and tests. Silent changes in any of these can
produce plausible but incorrect output.

## Change rules

- Keep the CLI dependency-light and usable through `python3 -m tools`.
- Prefer typed, composable transformations over ad hoc text rewriting.
- Make generated output deterministic across runs and platforms.
- Preserve useful diagnostics, including the input path and address when an
  error is tied to ROM content.
- Add a regression fixture for parser, decoder, control-flow, or codegen bugs.
- Put generated artifacts in the consumer or a temporary/build directory, not
  in this repository.
- Never commit ROM images, Python caches, pytest caches, local virtual
  environments, or transient scan output.

## Cross-repository delivery

If a tool change intentionally alters the Streets of Rage generated output,
keep the tool change and consumer regeneration reviewable as separate
repository changes, explain the expected delta, and validate the rebuilt
consumer. After validation, commit and push this repository to `main`
automatically unless the user explicitly asks not to publish. When checked out
as a submodule, publish this repository first and then update the parent
gitlink. Preserve unrelated work and never force-push or rewrite history.
