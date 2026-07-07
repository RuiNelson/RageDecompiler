# RageDecompiler

Python tools for reverse-engineering and recompiling Streets of Rage ROM code.

The unified CLI is still:

```bash
python3 -m tools <command> ...
```

When using this repository from a sibling project, put this directory on
`PYTHONPATH`:

```bash
PYTHONPATH=../RageDecompiler python3 -m tools --help
```

The Streets of Rage fixture data lives in `../StreetsOfRageRecompilation` by
default. Tests that need those fixtures can be pointed elsewhere with:

```bash
SOR_RECOMPILATION_DIR=/path/to/StreetsOfRageRecompilation python3 -m pytest
```
