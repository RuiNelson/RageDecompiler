"""Per-flag CCR liveness for omitting dead flag updates in generated C++.

Backward dataflow over one recompiled function decides which of N/Z/V/C/X are
still observed after each instruction.  The emitter then skips CCR writes for
flags that are dead — the main readability win for otherwise faithful 68000
lowering (``move`` / ``addq`` / ``andi`` chains that nobody branches on).

Conservatism
------------
* Function exits (``rts``, tail-calls, cross-function ``jmp``/``bra``) treat
  every flag as live so the caller still sees a correct CCR.
* ``rte`` / ``rtr`` replace CCR from the stack, so prior flag writes are dead.
* Calls (``bsr`` / ``jsr``) are modelled as preserving CCR (no kill).  That may
  keep a few more flag updates than a clobber model, but never drops a write
  that a flag-preserving leaf might need after return.
* Partial writes (bit ops only touch Z; logical ops leave X alone) are modelled
  precisely so a live X is not falsely killed by ``move``/``andi``.
"""

from __future__ import annotations

from tools.disassembler.instruction import EAMode, FlowType

# Flag names used as set elements throughout the recompiler.
N, Z, V, C, X = 'N', 'Z', 'V', 'C', 'X'
ALL = frozenset({N, Z, V, C, X})
NZVC = frozenset({N, Z, V, C})
NZVCX = ALL
LOGICAL = NZVC          # set N/Z, clear V/C; X unchanged
BIT_Z = frozenset({Z})  # btst/bset/bclr/bchg

# 68000 condition → flags read (matches CPU68K::condition).
_CC_READS = {
    0:  frozenset(),           # T
    1:  frozenset(),           # F
    2:  frozenset({C, Z}),     # HI
    3:  frozenset({C, Z}),     # LS
    4:  frozenset({C}),        # CC
    5:  frozenset({C}),        # CS
    6:  frozenset({Z}),        # NE
    7:  frozenset({Z}),        # EQ
    8:  frozenset({V}),        # VC
    9:  frozenset({V}),        # VS
    10: frozenset({N}),       # PL
    11: frozenset({N}),       # MI
    12: frozenset({N, V}),     # GE
    13: frozenset({N, V}),     # LT
    14: frozenset({N, V, Z}),  # GT
    15: frozenset({N, V, Z}),  # LE
}

_CC = {
    't': 0, 'f': 1, 'hi': 2, 'ls': 3, 'cc': 4, 'cs': 5, 'ne': 6, 'eq': 7,
    'vc': 8, 'vs': 9, 'pl': 10, 'mi': 11, 'ge': 12, 'lt': 13, 'gt': 14, 'le': 15,
}

# Mnemonics that never touch CCR (data movement / address arithmetic / control).
_NO_CCR = frozenset({
    'nop', 'lea', 'pea', 'exg', 'movem', 'movep', 'movea',
    'bra', 'bsr', 'jmp', 'jsr',
})

# ALU that sets N/Z/V/C and X:=C.
_ARITH = frozenset({
    'add', 'addi', 'addq', 'sub', 'subi', 'subq', 'neg',
})

# Compare: N/Z/V/C only (X unchanged).
_CMP = frozenset({'cmp', 'cmpa', 'cmpi', 'cmpm'})

# Logical: N/Z, V=0, C=0; X unchanged.
_LOGIC = frozenset({
    'and', 'andi', 'or', 'ori', 'eor', 'eori',
    'move', 'moveq', 'tst', 'clr', 'not', 'swap', 'ext',
    'mulu', 'muls',
})

_BIT = frozenset({'btst', 'bset', 'bclr', 'bchg'})

_SHIFT = frozenset({
    'lsl', 'lsr', 'asl', 'asr', 'rol', 'ror', 'roxl', 'roxr',
})

_BCD = frozenset({'abcd', 'sbcd', 'nbcd'})


def cc_reads(cc: int) -> frozenset:
    return _CC_READS[cc & 0x0F]


def _dst_is_areg(instr) -> bool:
    return len(instr.eas) >= 2 and instr.eas[1].mode == EAMode.ADDR_REG


def _special_name(instr, index: int) -> str | None:
    if index >= len(instr.eas):
        return None
    e = instr.eas[index]
    if e.mode == EAMode.SPECIAL_REG:
        return e.special
    return None


def effects(instr) -> tuple[frozenset, frozenset]:
    """Return ``(reads, writes)`` flag sets for *instr*.

    ``writes`` is the set of flags the instruction **definitely assigns**.
    Flags not in ``writes`` are preserved (important for X vs logical ops, and
    for bit ops that only touch Z).
    """
    m = instr.mnemonic

    if m in _NO_CCR:
        return frozenset(), frozenset()

    if m == 'rts':
        # CCR escapes to the caller — model as a full read at the return.
        return ALL, frozenset()

    if m in ('rte', 'rtr'):
        # SR/CCR reloaded from the stack; prior values are dead.
        return frozenset(), ALL

    if m.startswith('db'):
        suffix = m[2:]
        cc = 1 if suffix in ('f', 'ra') else _CC[suffix]
        return cc_reads(cc), frozenset()

    if m.startswith('b') and m[1:] in _CC and m != 'bra':
        return cc_reads(_CC[m[1:]]), frozenset()

    # Scc — reads condition, does not write CCR.
    if m in {
        'st', 'sf', 'shi', 'sls', 'scc', 'scs', 'sne', 'seq',
        'svc', 'svs', 'spl', 'smi', 'sge', 'slt', 'sgt', 'sle',
    }:
        return cc_reads(_CC[m[1:]]), frozenset()

    # adda / suba never touch CCR; add/sub/cmp to An use the *a forms.
    if m in ('adda', 'suba'):
        return frozenset(), frozenset()
    if m in ('add', 'addi', 'addq', 'sub', 'subi', 'subq') and _dst_is_areg(instr):
        return frozenset(), frozenset()

    if m in _CMP:
        return frozenset(), NZVC

    if m in _ARITH:
        return frozenset(), NZVCX

    if m == 'negx':
        return frozenset({X}), NZVCX

    if m in _BCD:
        return frozenset({X}), frozenset({N, Z, C, X})

    if m in _BIT:
        return frozenset(), BIT_Z

    if m in _SHIFT:
        reads = frozenset({X}) if m in ('roxl', 'roxr') else frozenset()
        # Register-count shifts leave X unchanged when count is 0; still list X
        # as written only for forms that always update it (imm or memory ×1).
        # Conservatively treat all shifts as writing NZVCX when count is not a
        # fixed zero — immediate counts are 1..8, memory count is 1.
        return reads, NZVCX

    if m in ('divu', 'divs'):
        # X unchanged; N/Z/V/C assigned on the non-overflow path (V/C always).
        return frozenset(), NZVC

    # move / andi / ori / eori involving SR or CCR.
    if m in ('move', 'andi', 'ori', 'eori'):
        src_sp = _special_name(instr, 0)
        dst_sp = _special_name(instr, 1)
        if dst_sp in ('ccr', 'sr'):
            if m == 'move':
                return frozenset(), ALL
            # read-modify-write of CCR/SR.
            return ALL, ALL
        if src_sp in ('ccr', 'sr'):
            return ALL, frozenset()
        if m == 'move' and _dst_is_areg(instr):
            return frozenset(), frozenset()  # movea
        if m in _LOGIC or m == 'move':
            return frozenset(), LOGICAL

    if m in _LOGIC:
        return frozenset(), LOGICAL

    # Unknown / unimplemented: assume full clobber so we never drop a write.
    return frozenset(), ALL


def _successors(addr: int, instr, addr_set: set, func_of, entry: int):
    """Yield successor addresses inside this function, or ``None`` for exit.

    ``None`` means control leaves the function with CCR observable by the
    caller / callee (flags must be treated as live-out).
    """
    flow = instr.flow
    targets = list(instr.targets or [])

    def same_fn(tgt: int) -> bool:
        return tgt in addr_set and func_of(tgt) == entry

    if flow is FlowType.RETURN:
        yield None
        return

    if flow is FlowType.BRANCH:
        if not targets:
            yield None
            return
        tgt = targets[0]
        yield tgt if same_fn(tgt) else None
        return

    if flow is FlowType.CALL:
        # Fall-through after return; model call as CCR-preserving (no edge into
        # the callee for flag purposes).
        nxt = instr.next_address
        if nxt in addr_set and func_of(nxt) == entry:
            yield nxt
        else:
            yield None
        return

    if flow is FlowType.CONDITIONAL:
        for tgt in targets:
            yield tgt if same_fn(tgt) else None
        nxt = instr.next_address
        if nxt in addr_set and func_of(nxt) == entry:
            yield nxt
        else:
            yield None
        return

    # SEQUENTIAL
    nxt = instr.next_address
    if nxt in addr_set and func_of(nxt) == entry:
        yield nxt
    else:
        # Fall off the function (tail into another body, or end of ROM slice).
        yield None


def analyze(addrs: list[int], instructions: dict, func_of, entry: int
            ) -> dict[int, frozenset]:
    """Compute live-out CCR flags after each address in *addrs*.

    Returns a map ``addr -> frozenset`` of flags that must still be correct
    after the instruction at ``addr`` finishes.  Missing addresses are treated
    as fully live (safe default).
    """
    addr_set = set(addrs)
    if not addrs:
        return {}

    reads = {}
    writes = {}
    succs = {}
    for addr in addrs:
        instr = instructions[addr]
        r, w = effects(instr)
        reads[addr] = r
        writes[addr] = w
        succs[addr] = list(_successors(addr, instr, addr_set, func_of, entry))

    preds: dict[int, list[int]] = {a: [] for a in addrs}
    for addr in addrs:
        for s in succs[addr]:
            if s is not None:
                preds[s].append(addr)

    live_in = {a: frozenset() for a in addrs}
    live_out = {a: frozenset() for a in addrs}

    # Backward worklist.
    work = list(reversed(addrs))
    in_work = set(work)
    while work:
        addr = work.pop()
        in_work.discard(addr)

        out: set = set()
        for s in succs[addr]:
            if s is None:
                out |= ALL
            else:
                out |= live_in[s]
        out_f = frozenset(out)
        live_out[addr] = out_f

        new_in = frozenset(reads[addr] | (out_f - writes[addr]))
        if new_in != live_in[addr]:
            live_in[addr] = new_in
            for pred in preds[addr]:
                if pred not in in_work:
                    work.append(pred)
                    in_work.add(pred)

    return live_out


def live_for_write(live_out: frozenset | None, writes: frozenset) -> frozenset:
    """Flags this instruction should still update: live-out ∩ writes."""
    if live_out is None:
        return writes
    return frozenset(live_out) & writes
