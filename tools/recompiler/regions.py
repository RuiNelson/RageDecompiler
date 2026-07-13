"""Partition decoded instructions into C++ functions.

A *function* spans from a subroutine/call entry up to the next entry, so that
sequential code and intra-function branches (including back-edges) stay inside
one function — loops therefore stay as ``goto``/``while``, never recursion
through native calls.

Transfers are classified:

* **intra-function** (target in the same function)  → ``goto`` to a local label;
* **cross-function to another entry**                → native call to that fn;
* **cross-function into the middle** of a function   → native call carrying the
  target address; the callee routes it via an entry ``switch`` trampoline.
  Such mid-function targets are recorded as extra entry points of the owning
  function.
"""

import bisect
from dataclasses import dataclass, field

from tools.disassembler.instruction import FlowType


@dataclass
class Function:
    entry:    int                       # base entry address (lowest)
    addrs:    list = field(default_factory=list)   # sorted instruction addresses
    extra_entries: set = field(default_factory=set)  # cross-function mid-entries


@dataclass
class Partition:
    functions:   dict          # entry addr -> Function
    entries:     list          # sorted entry addresses
    goto_labels: set           # addresses reached by an intra-function goto
    instructions: dict         # {addr: Instruction}
    owners:       dict          # exact instruction addr -> owning function entry

    def func_of(self, addr: int) -> int:
        """Entry address of the function that owns ``addr`` (or None)."""
        if addr in self.owners:
            return self.owners[addr]
        i = bisect.bisect_right(self.entries, addr) - 1
        return self.entries[i] if i >= 0 else None

    def needs_label(self, addr: int) -> bool:
        """True if ``addr`` must carry a C label (goto target or mid-entry)."""
        if addr in self.goto_labels:
            return True
        f = self.functions.get(self.func_of(addr))
        return f is not None and addr in f.extra_entries


def partition(instructions: dict, subroutines: set,
              owner_overrides: dict | None = None) -> Partition:
    """Build the function partition from decoded instructions.

    ``subroutines`` are the disassembler's identified entries (reset/IRQ seeds,
    aux addresses, and JSR/BSR targets). Any decoded instruction is also a
    fall-back entry boundary only through those; everything else is owned by the
    nearest preceding entry unless ``owner_overrides`` assigns it to an
    overlapping speculative flow.
    """
    # Entries: known subroutines that actually decoded, always including the
    # lowest decoded address so every instruction is owned.
    entries = sorted(a for a in subroutines if a in instructions)
    if not entries or entries[0] != min(instructions):
        entries = sorted(set(entries) | {min(instructions)})

    functions = {e: Function(entry=e) for e in entries}
    owner_overrides = owner_overrides or {}
    owners = {}
    for addr in sorted(instructions):
        owner = owner_overrides.get(addr)
        if owner not in functions:
            i = bisect.bisect_right(entries, addr) - 1
            owner = entries[i]
        functions[owner].addrs.append(addr)
        owners[addr] = owner

    part = Partition(functions=functions, entries=entries,
                     goto_labels=set(), instructions=instructions,
                     owners=owners)
    next_owned = {}
    for function in functions.values():
        next_owned.update(zip(function.addrs, function.addrs[1:]))

    # Classify explicit transfers and implicit fall-through edges.  Fall-through
    # normally remains inside one numeric function, but ownership overrides for
    # overlapping speculative decodes can make it cross into another body.
    for addr, instr in instructions.items():
        fsrc = part.func_of(addr)
        explicit = list(instr.targets) if instr.flow in (
            FlowType.BRANCH, FlowType.CONDITIONAL, FlowType.CALL) else []
        fallthrough = [instr.next_address] if instr.flow in (
            FlowType.SEQUENTIAL, FlowType.CONDITIONAL, FlowType.CALL) else []
        for tgt in explicit + fallthrough:
            if tgt not in instructions:
                continue
            ftgt = part.func_of(tgt)
            if ftgt == fsrc:
                if tgt in explicit and instr.flow in (
                        FlowType.BRANCH, FlowType.CONDITIONAL):
                    part.goto_labels.add(tgt)        # intra-function goto
                elif tgt in fallthrough and next_owned.get(addr) != tgt:
                    # The same logical body resumes after an interleaved
                    # overlapping decode; codegen emits an explicit goto over
                    # the other body's numerically adjacent instructions.
                    part.goto_labels.add(tgt)
            elif tgt != ftgt:
                functions[ftgt].extra_entries.add(tgt)  # mid-function entry

    return part
