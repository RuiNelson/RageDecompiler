"""Repeated instruction-sequence fusion for the C++ recompiler.

Classic 68000 game code was full of assembler macros that expanded to the same
multi-instruction idioms over and over (structure field init, block copy,
``moveq`` prologues, fill loops).  This module mirrors that idea in reverse:

1. Scan every subroutine for **fusible** sequential runs.
2. Rank multi-instruction **shapes** by how often they appear.
3. When emitting C++, match those repeated patterns and emit one optimised
   sequence block instead of one brace-block per instruction.

Safety (“zero danger”)
----------------------
* Never fuse across a goto label or mid-function entry (except the first PC).
* Never fuse control-flow, ``movem``, or non-sequential fall-through gaps.
* Always emit ``BEFORE_INSTRUCTION`` once per original opcode (IRQ + pace).
* Preserve per-instruction CCR liveness (no flag update moved or dropped).
* Never cache register/memory values across a ``BEFORE_INSTRUCTION`` — an IRQ
  handler may mutate any of them, so each step re-reads state after the check.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from tools.disassembler.instruction import EAMode, FlowType
from tools.recompiler import opcodes

# Longest sequence we will ever fuse (keeps matching cheap and listings readable).
MAX_SEQ_LEN = 32

# A shape must appear at least this many times across the ROM to qualify as a
# “macro” pattern.  Homogeneous optimisers (identical-op loops) ignore this.
MIN_PATTERN_COUNT = 5

# Prefer a compact C++ for-loop once an identical op repeats this many times.
IDENTICAL_LOOP_THRESHOLD = 4


# ---------------------------------------------------------------------------
# Shape keys (register-agnostic mnemonic + addressing-mode skeleton)
# ---------------------------------------------------------------------------

def _ea_shape(e) -> str:
    m = e.mode
    if m is EAMode.DATA_REG:
        return 'Dn'
    if m is EAMode.ADDR_REG:
        return 'An'
    if m is EAMode.ADDR_IND:
        return '(An)'
    if m is EAMode.ADDR_POSTINC:
        return '(An)+'
    if m is EAMode.ADDR_PREDEC:
        return '-(An)'
    if m is EAMode.ADDR_DISP:
        return 'd16(An)'
    if m is EAMode.ADDR_INDEX:
        return 'd8(An,Xn)'
    if m is EAMode.ABS_W:
        return 'abs.w'
    if m is EAMode.ABS_L:
        return 'abs.l'
    if m is EAMode.PC_DISP:
        return 'd16(PC)'
    if m is EAMode.PC_INDEX:
        return 'd8(PC,Xn)'
    if m is EAMode.IMMEDIATE:
        return '#imm'
    if m is EAMode.SPECIAL_REG:
        return e.special or 'spec'
    if m is EAMode.REG_LIST:
        return 'reglist'
    if m is EAMode.BRANCH_TARGET:
        return 'target'
    return m.name


def instr_shape(instr) -> tuple:
    """Register-agnostic shape of one instruction (mnemonic, size, EA modes)."""
    parts = [instr.mnemonic, instr.size or '-']
    parts.extend(_ea_shape(e) for e in instr.eas)
    return tuple(parts)


def _shape_label(shape: tuple) -> str:
    """Human-readable one-instruction shape, e.g. ``move.b #imm,d16(An)``."""
    mnem, size, *eas = shape
    head = mnem if size == '-' else f'{mnem}.{size}'
    if not eas:
        return head
    return f'{head} {",".join(eas)}'


def pattern_label(shapes: tuple) -> str:
    return ' ; '.join(_shape_label(s) for s in shapes)


# ---------------------------------------------------------------------------
# Fusibility
# ---------------------------------------------------------------------------

def is_data_op(instr) -> bool:
    """True if *instr* is a pure data/ALU op safe to include in a sequence."""
    m = instr.mnemonic
    if m in opcodes.FLOW_MNEMONICS or m in opcodes.GENERATOR_MNEMONICS:
        return False
    if m == 'nop':
        return True
    return True


def can_start_or_continue(instr) -> bool:
    return is_data_op(instr) and instr.flow is FlowType.SEQUENTIAL


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@dataclass
class SequenceStats:
    """Aggregate counts reported after a recompile pass."""
    pattern_counts: Counter = field(default_factory=Counter)
    fused_sequences: int = 0
    fused_instructions: int = 0
    loop_sequences: int = 0

    def note_fuse(self, n_instr: int, used_loop: bool = False) -> None:
        self.fused_sequences += 1
        self.fused_instructions += n_instr
        if used_loop:
            self.loop_sequences += 1

    def summary_lines(self, top: int = 12) -> list[str]:
        if not self.fused_sequences and not self.pattern_counts:
            return []
        lines = [
            f'[recompile] sequence macros: fused {self.fused_sequences} run(s) '
            f'covering {self.fused_instructions} instruction(s)'
            + (f', {self.loop_sequences} as loops' if self.loop_sequences else ''),
        ]
        if self.pattern_counts:
            ranked = self.pattern_counts.most_common(top)
            shown = ', '.join(
                f'{pattern_label(p)}×{c}' for p, c in ranked)
            lines.append(f'[recompile] top repeated shapes: {shown}')
        return lines


def count_pattern_shapes(eff_addrs_by_entry: dict, instructions: dict,
                         needs_label, max_len: int = 6) -> Counter:
    """Frequency table of fusible multi-instruction shapes (n-grams 2..max_len)."""
    counts: Counter = Counter()
    for addrs in eff_addrs_by_entry.values():
        if len(addrs) < 2:
            continue
        # Every start position, so overlapping idioms each count once.
        for i in range(len(addrs)):
            run = _fusible_run(addrs, i, instructions, needs_label,
                               max_len=max_len)
            if len(run) < 2:
                continue
            shapes = [instr_shape(instructions[a]) for a in run]
            for n in range(2, len(shapes) + 1):
                counts[tuple(shapes[:n])] += 1
    return counts


def discover_patterns(eff_addrs_by_entry: dict, instructions: dict,
                      needs_label, min_count: int = MIN_PATTERN_COUNT,
                      max_len: int = 6) -> tuple[frozenset, Counter]:
    """Return (macro shape set, full frequency table).

    A shape qualifies as a sequence macro when it appears >= *min_count* times
    across all subroutines.
    """
    counts = count_pattern_shapes(
        eff_addrs_by_entry, instructions, needs_label, max_len=max_len)
    known = frozenset(p for p, c in counts.items() if c >= min_count)
    return known, counts


def _fusible_run(addrs, start_idx, instructions, needs_label,
                 max_len: int = MAX_SEQ_LEN) -> list[int]:
    """Longest fusible address list starting at *start_idx* (may be length 1)."""
    if start_idx >= len(addrs):
        return []
    first = addrs[start_idx]
    instr0 = instructions[first]
    if not can_start_or_continue(instr0):
        return [first] if first in instructions else []

    run = [first]
    i = start_idx
    while i + 1 < len(addrs) and len(run) < max_len:
        cur = instructions[addrs[i]]
        if cur.flow is not FlowType.SEQUENTIAL:
            break
        nxt = addrs[i + 1]
        if cur.next_address != nxt:
            break
        if needs_label(nxt):
            break
        ninstr = instructions.get(nxt)
        if ninstr is None or not can_start_or_continue(ninstr):
            break
        run.append(nxt)
        i += 1
    return run


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

@dataclass
class SequenceMatch:
    """A concrete run of addresses to emit as one optimised sequence."""
    addrs: list[int]
    kind: str                 # 'loop' | 'pattern' | 'homogeneous'
    shapes: tuple


def find_match(addrs, start_idx, instructions, needs_label,
               known_patterns: frozenset) -> SequenceMatch | None:
    """Greedy longest match at *start_idx*, or None to fall back to 1-op emit."""
    run = _fusible_run(addrs, start_idx, instructions, needs_label)
    if len(run) < 2:
        return None

    # 1) Identical-instruction loop (strongest optimisation).
    if _all_identical(run, instructions) and len(run) >= IDENTICAL_LOOP_THRESHOLD:
        shapes = tuple(instr_shape(instructions[a]) for a in run)
        return SequenceMatch(addrs=run, kind='loop', shapes=shapes)

    # 2) Longest prefix that is a known frequent pattern.
    shapes = [instr_shape(instructions[a]) for a in run]
    best = None
    for n in range(len(shapes), 1, -1):
        key = tuple(shapes[:n])
        if key in known_patterns:
            best = SequenceMatch(addrs=run[:n], kind='pattern', shapes=key)
            break
    if best is not None:
        return best

    # 3) Homogeneous optimisable idioms even below the global frequency cut
    #    (e.g. a single 32× fill loop that only appears once).
    if _is_homogeneous_move_run(run, instructions) and len(run) >= 3:
        key = tuple(shapes)
        return SequenceMatch(addrs=run, kind='homogeneous', shapes=key)

    return None


def _all_identical(run, instructions) -> bool:
    """True when every instruction is bitwise-equivalent in operands too."""
    first = instructions[run[0]]
    for a in run[1:]:
        if not _instr_same(first, instructions[a]):
            return False
    return True


def _instr_same(a, b) -> bool:
    if a.mnemonic != b.mnemonic or a.size != b.size:
        return False
    if len(a.eas) != len(b.eas):
        return False
    for ea_a, ea_b in zip(a.eas, b.eas):
        if ea_a.mode != ea_b.mode:
            return False
        if getattr(ea_a, 'reg', None) != getattr(ea_b, 'reg', None):
            return False
        if getattr(ea_a, 'reg2', None) != getattr(ea_b, 'reg2', None):
            return False
        if getattr(ea_a, 'imm', None) != getattr(ea_b, 'imm', None):
            return False
        if getattr(ea_a, 'disp', None) != getattr(ea_b, 'disp', None):
            return False
        if getattr(ea_a, 'abs_value', None) != getattr(ea_b, 'abs_value', None):
            return False
        if getattr(ea_a, 'special', None) != getattr(ea_b, 'special', None):
            return False
        if getattr(ea_a, 'reglist', None) != getattr(ea_b, 'reglist', None):
            return False
        if getattr(ea_a, 'index_size', None) != getattr(ea_b, 'index_size', None):
            return False
        if getattr(ea_a, 'index_reg', None) != getattr(ea_b, 'index_reg', None):
            return False
        if getattr(ea_a, 'index_is_addr', False) != getattr(ea_b, 'index_is_addr', False):
            return False
    return True


def _is_homogeneous_move_run(run, instructions) -> bool:
    """Same mnemonic+size+EA-mode skeleton (regs may differ only if identical)."""
    first = instructions[run[0]]
    if first.mnemonic not in ('move', 'moveq', 'movea'):
        return False
    shape0 = instr_shape(first)
    return all(instr_shape(instructions[a]) == shape0 for a in run[1:])


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def emit_sequence(match: SequenceMatch, instructions: dict, live_out: dict,
                  emit_one, needs_label, label_name) -> tuple[list[str], bool]:
    """Lower *match* to C++ lines.

    ``emit_one(instr, live)`` must return the body statements of a single data
    op (no braces, no BEFORE_INSTRUCTION, no address comment).

    Returns ``(lines, used_loop)``.
    """
    addrs = match.addrs
    first_addr = addrs[0]
    last_addr = addrs[-1]
    n = len(addrs)

    lines: list[str] = []
    if needs_label(first_addr):
        lines.append(f'{label_name(first_addr)}:')

    desc = pattern_label(match.shapes)
    lines.append(
        f'// seq ${first_addr:06X}–${last_addr:06X} ({n}×) [{match.kind}] {desc}')

    if match.kind == 'loop' and _all_identical(addrs, instructions):
        # One outer scope holds the for-loop; body re-reads state after every
        # BEFORE_INSTRUCTION so IRQ mutation cannot leave a stale register.
        lines.append('{')
        body_lines, used_loop = _emit_identical_loop(
            addrs, instructions, live_out, emit_one)
        lines.extend(f'    {s}' for s in body_lines)
        lines.append('}')
        return lines, used_loop

    # Pre-lower every step.  When no step declares a temp, flatten into one
    # scope (the common structure-init / moveq macro case).  Otherwise keep
    # per-opcode braces so t0/t1 reuse stays valid.
    steps = []
    for addr in addrs:
        instr = instructions[addr]
        live = live_out.get(addr)  # None → emit_dataop keeps all flags
        body = emit_one(instr, live)
        steps.append((addr, instr, body))

    if all(not _body_declares_temp(body) for _, _, body in steps):
        lines.append('{')
        for addr, instr, body in steps:
            lines.append(f'    // ${addr:06X} {instr}')
            lines.append('    BEFORE_INSTRUCTION')
            for stmt in body:
                lines.append(f'    {stmt}')
        lines.append('}')
        return lines, False

    for addr, instr, body in steps:
        lines.append(f'// ${addr:06X} {instr}')
        lines.append('{')
        lines.append('    BEFORE_INSTRUCTION')
        for stmt in body:
            lines.append(f'    {stmt}')
        lines.append('}')
    return lines, False


def _body_declares_temp(body: list[str]) -> bool:
    """True if any statement introduces a C++ local (``m_byte t0 = …``)."""
    for stmt in body:
        s = stmt.lstrip()
        if s.startswith(('m_byte ', 'm_word ', 'm_long ', 'uint64_t ', 'int ')):
            return True
    return False


def _emit_identical_loop(addrs, instructions, live_out, emit_one
                         ) -> tuple[list[str], bool]:
    """Emit a for-loop for N identical data ops.

    Intermediate CCR updates that are dead stay omitted; if any intermediate
    step still has live flags, each iteration updates flags (same value ⇒
    same flags, still correct).  BEFORE_INSTRUCTION runs every iteration.
    """
    n = len(addrs)
    prototype = instructions[addrs[0]]
    # Collect which iterations need any flag write.
    flag_bodies = []
    data_bodies = []
    for addr in addrs:
        live = live_out.get(addr)
        body = emit_one(instructions[addr], live)
        # Split trailing flag updates from the data motion.  Flag helpers all
        # start with ``cpu().set`` — keep this conservative: if we cannot split
        # cleanly, fall back to unrolled emission.
        data, flags = _split_data_and_flags(body)
        data_bodies.append(data)
        flag_bodies.append(flags)

    if not all(d == data_bodies[0] for d in data_bodies):
        # emit_one produced address-dependent temps/comments — unroll instead.
        return _unrolled_from_bodies(addrs, instructions, data_bodies,
                                     flag_bodies), False

    data = data_bodies[0]
    any_mid_flags = any(flag_bodies[i] for i in range(n - 1))
    last_flags = flag_bodies[-1]

    out: list[str] = [f'for (int _seq = 0; _seq < {n}; ++_seq) {{',
                      '    BEFORE_INSTRUCTION']
    for stmt in data:
        out.append(f'    {stmt}')

    if any_mid_flags:
        # Every iteration may need flags (rare for pure block copies).
        # Use the prototype's last-live set when the final op has flags; for
        # mid iterations re-emit via a switch only if they differ.
        if all(f == flag_bodies[0] for f in flag_bodies):
            for stmt in flag_bodies[0]:
                out.append(f'    {stmt}')
        else:
            out.append('    switch (_seq) {')
            for i, flags in enumerate(flag_bodies):
                if not flags:
                    continue
                out.append(f'    case {i}:')
                for stmt in flags:
                    out.append(f'        {stmt}')
                out.append('        break;')
            out.append('    }')
    elif last_flags:
        out.append(f'    if (_seq == {n - 1}) {{')
        for stmt in last_flags:
            out.append(f'        {stmt}')
        out.append('    }')

    out.append('}')
    return out, True


def _unrolled_from_bodies(addrs, instructions, data_bodies, flag_bodies):
    out = []
    for i, addr in enumerate(addrs):
        out.append(f'// ${addr:06X} {instructions[addr]}')
        out.append('{')
        out.append('    BEFORE_INSTRUCTION')
        for stmt in data_bodies[i] + flag_bodies[i]:
            out.append(f'    {stmt}')
        out.append('}')
    return out


def _split_data_and_flags(body: list[str]) -> tuple[list[str], list[str]]:
    """Split opcode body into data-motion stmts vs trailing CCR updates."""
    data: list[str] = []
    flags: list[str] = []
    seen_flag = False
    for stmt in body:
        is_flag = (
            stmt.startswith('cpu().setNZ')
            or stmt.startswith('cpu().setFlag(')
            or stmt.startswith('cpu().setCCR(')
            or stmt.startswith('cpu().setStatus(')
        )
        if is_flag:
            seen_flag = True
            flags.append(stmt)
        elif seen_flag:
            # Non-flag after flag — treat whole body as data (don't mis-split).
            return body, []
        else:
            data.append(stmt)
    return data, flags


def emit_one_dataop(instr, live):
    """Default single-op body used by sequence emission (no flow/movem)."""
    if instr.mnemonic == 'nop':
        return ['(void)0;']
    body = opcodes.emit_dataop(instr, live_flags=live)
    if body is None:
        raise opcodes.Unsupported(instr.mnemonic)
    return body
