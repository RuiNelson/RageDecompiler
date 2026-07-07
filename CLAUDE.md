# CLAUDE.md

Guidance for LLM agents working in `RageDecompiler`.

## Scope

`RageDecompiler` contains Python tools for reverse-engineering Streets of Rage
ROM code. It is used as a sibling repository by
`../StreetsOfRageRecompilation`.

Do not make changes in `../Genesis-Plus-GX`; it is an upstream dependency not
owned by this project.

## Layout

- `tools/main.py` - unified CLI dispatcher.
- `tools/disassembler/` - static recursive-descent 68000 disassembler.
- `tools/recompiler/` - C++ code generator for ROM regions.
- `tools/label_diff/` - assembly/map comparison helpers.
- `tools/iterative_disasm/` - loop for expanding static coverage.
- `tools/remove_data_locations/` - strips data blocks from assembly output.
- `tools/speculative_scan/` - scans unmapped ROM regions for candidates.
- `tools/*/tests/` - Python tests.

## Commands

Run the unified CLI from this repository:

```bash
python3 -m tools --help
```

When called from `../StreetsOfRageRecompilation`, put this repository on
`PYTHONPATH`:

```bash
PYTHONPATH=../RageDecompiler python3 -m tools --help
```

Run tests:

```bash
python3 -m pytest
```

Tests that need Streets of Rage fixture data default to
`../StreetsOfRageRecompilation`. Override with:

```bash
SOR_RECOMPILATION_DIR=/path/to/StreetsOfRageRecompilation python3 -m pytest
```

## Tooling Notes

- The disassembler follows reachable code from known entry points and auxiliary
  addresses.
- Indirect jumps/calls cannot always be resolved statically; use active
  disassembly data and `iterative_disasm` workflows to close gaps.
- Keep generated output in the recompilation repository, not here.
- Prefer the unified `python3 -m tools <command>` surface over legacy module
  entry points.
