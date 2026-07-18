"""Hand-written multi-instruction pattern → custom C++ substitution.

Classic 68000 sources expanded assembler macros into the same instruction
sequences over and over.  This module is the reverse: each **registered**
pattern names one of those idioms and supplies a custom C++ emitter.

Workflow
--------
1. **Discover** candidates (optional report): frequent fusible shapes that no
   registered pattern handles yet — see ``suggest_unhandled_shapes``.
2. **Register** a new pattern in ``PATTERNS`` below (or via ``register``):
   a matcher that recognises the sequence, and an emitter that returns the
   replacement C++ statements.
3. Recompile — every occurrence is substituted.

Zero-danger contract (framework-enforced)
-----------------------------------------
* Matching only considers **fusible** runs: sequential data ops, no mid-run
  labels / mid-function entries, no control-flow / ``movem``.
* The custom emitter owns the body, including every
  ``BEFORE_INSTRUCTION`` (one per original opcode — IRQ + pace).
* Do not cache register/memory values across a ``BEFORE_INSTRUCTION``: an IRQ
  handler may mutate any of them.

Adding a pattern
----------------
Copy one of the handlers at the bottom of this file.  Typical skeleton::

    def _match_my_macro(instrs):
        # Return (length, data_dict) or None.
        ...

    def _emit_my_macro(m, live_out):
        # Return C++ statement lines (no outer function braces).
        # Must include BEFORE_INSTRUCTION once per consumed instruction.
        ...

    register(Pattern(
        name='my_macro',
        doc='Short description of the 68000 idiom',
        match=_match_my_macro,
        emit=_emit_my_macro,
    ))
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from tools.disassembler.instruction import EAMode, FlowType
from tools.recompiler import cpp_semantics as sem
from tools.recompiler import ea_codegen as ea
from tools.recompiler import opcodes
from tools.recompiler.ccr_liveness import ALL

# Longest run the matcher framework will feed a pattern (cheap + readable).
MAX_RUN_LEN = 64


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

MatchFn = Callable[[list], tuple[int, dict] | None]
EmitFn = Callable[['PatternHit', dict], list[str]]


@dataclass(frozen=True)
class Pattern:
    """One hand-written sequence macro."""

    name: str
    doc: str
    match: MatchFn
    emit: EmitFn


@dataclass
class PatternHit:
    """A concrete match ready to emit."""

    pattern: Pattern
    addrs: list[int]
    instrs: list
    data: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.pattern.name

    @property
    def n(self) -> int:
        return len(self.addrs)


@dataclass
class PatternStats:
    """Per-recompile counters."""

    hits: Counter = field(default_factory=Counter)       # name → match count
    instructions: Counter = field(default_factory=Counter)  # name → instrs covered
    suggestions: list[tuple[str, int]] = field(default_factory=list)

    def note(self, hit: PatternHit) -> None:
        self.hits[hit.name] += 1
        self.instructions[hit.name] += hit.n

    def summary_lines(self, top: int = 12) -> list[str]:
        lines = []
        if self.hits:
            total_hits = sum(self.hits.values())
            total_ins = sum(self.instructions.values())
            detail = ', '.join(
                f'{name}×{self.hits[name]}({self.instructions[name]} ops)'
                for name, _ in self.hits.most_common())
            lines.append(
                f'[recompile] patterns: {total_hits} hit(s), '
                f'{total_ins} instruction(s) replaced ({detail})')
        if self.suggestions:
            shown = ', '.join(f'{s}×{c}' for s, c in self.suggestions[:top])
            lines.append(
                f'[recompile] unhandled frequent shapes (candidates): {shown}')
        return lines


# Registry — append via ``register``; order is priority (first match wins when
# lengths tie; longer matches always beat shorter ones regardless of order).
PATTERNS: list[Pattern] = []


def register(pattern: Pattern) -> Pattern:
    """Append *pattern* to the global catalog and return it (decorator-friendly)."""
    PATTERNS.append(pattern)
    return pattern


# ---------------------------------------------------------------------------
# Shape helpers (discovery / suggestions)
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
    return m.name


def instr_shape(instr) -> tuple:
    parts = [instr.mnemonic, instr.size or '-']
    parts.extend(_ea_shape(e) for e in instr.eas)
    return tuple(parts)


def shape_label(shape: tuple) -> str:
    mnem, size, *eas = shape
    head = mnem if size == '-' else f'{mnem}.{size}'
    return head if not eas else f'{head} {",".join(eas)}'


def shapes_label(shapes: tuple) -> str:
    return ' ; '.join(shape_label(s) for s in shapes)


# ---------------------------------------------------------------------------
# Fusibility (shared safety gate)
# ---------------------------------------------------------------------------

def is_fusible_op(instr) -> bool:
    m = instr.mnemonic
    if m in opcodes.FLOW_MNEMONICS or m in opcodes.GENERATOR_MNEMONICS:
        return False
    return instr.flow is FlowType.SEQUENTIAL


def fusible_run(addrs, start_idx, instructions, needs_label,
                max_len: int = MAX_RUN_LEN) -> list[int]:
    """Longest fusible address list starting at *start_idx*."""
    if start_idx >= len(addrs):
        return []
    first = addrs[start_idx]
    if first not in instructions or not is_fusible_op(instructions[first]):
        return []

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
        if ninstr is None or not is_fusible_op(ninstr):
            break
        run.append(nxt)
        i += 1
    return run


# ---------------------------------------------------------------------------
# Match + emit
# ---------------------------------------------------------------------------

def try_match(addrs, start_idx, instructions, needs_label,
              catalog: list[Pattern] | None = None) -> PatternHit | None:
    """Try every registered pattern on the fusible run at *start_idx*.

    Among all matches, the **longest** wins; ties keep registry order.
    """
    run_addrs = fusible_run(addrs, start_idx, instructions, needs_label)
    if len(run_addrs) < 2:
        return None
    run_instrs = [instructions[a] for a in run_addrs]

    best: PatternHit | None = None
    for pat in (catalog if catalog is not None else PATTERNS):
        result = pat.match(run_instrs)
        if result is None:
            continue
        length, data = result
        if length < 2 or length > len(run_addrs):
            continue
        hit = PatternHit(
            pattern=pat,
            addrs=run_addrs[:length],
            instrs=run_instrs[:length],
            data=data or {},
        )
        if best is None or hit.n > best.n:
            best = hit
    return best


def emit_hit(hit: PatternHit, live_out: dict, needs_label, label_name
             ) -> list[str]:
    """Lower a match to C++ lines (label + comment + braced custom body)."""
    lines: list[str] = []
    first, last = hit.addrs[0], hit.addrs[-1]
    if needs_label(first):
        lines.append(f'{label_name(first)}:')
    lines.append(
        f'// pattern {hit.name} ${first:06X}–${last:06X} ({hit.n} ops) '
        f'— {hit.pattern.doc}')
    lines.append('{')
    body = hit.pattern.emit(hit, live_out)
    for stmt in body:
        lines.append(f'    {stmt}')
    lines.append('}')
    return lines


# ---------------------------------------------------------------------------
# Discovery: suggest shapes not yet covered by any registered pattern
# ---------------------------------------------------------------------------

def count_shapes(eff_addrs_by_entry: dict, instructions: dict, needs_label,
                 max_len: int = 4) -> Counter:
    counts: Counter = Counter()
    for addrs in eff_addrs_by_entry.values():
        for i in range(len(addrs)):
            run = fusible_run(addrs, i, instructions, needs_label,
                              max_len=max_len)
            if len(run) < 2:
                continue
            shapes = tuple(instr_shape(instructions[a]) for a in run)
            for n in range(2, len(shapes) + 1):
                counts[shapes[:n]] += 1
    return counts


def suggest_unhandled_shapes(eff_addrs_by_entry: dict, instructions: dict,
                             needs_label, min_count: int = 8, max_len: int = 4,
                             catalog: list[Pattern] | None = None,
                             top: int = 20) -> list[tuple[str, int]]:
    """Frequent shapes that no registered pattern would consume."""
    catalog = catalog if catalog is not None else PATTERNS
    counts = count_shapes(eff_addrs_by_entry, instructions, needs_label,
                          max_len=max_len)
    unhandled: Counter = Counter()
    for shapes, count in counts.items():
        if count < min_count:
            continue
        # Build a minimal fake instruction list is hard; instead walk real ROM
        # starts and mark shapes that get a hit.  Approximate: a shape is
        # "handled" if some pattern matches a run whose leading shape equals it.
        # Done in a second pass over the program.
        unhandled[shapes] = count

    # Subtract shapes that actually match at least once under the catalog.
    handled = set()
    for addrs in eff_addrs_by_entry.values():
        i = 0
        while i < len(addrs):
            hit = try_match(addrs, i, instructions, needs_label, catalog)
            if hit is None:
                i += 1
                continue
            shapes = tuple(instr_shape(ins) for ins in hit.instrs)
            handled.add(shapes)
            # Also mark every prefix — a longer registered match covers them.
            for n in range(2, len(shapes) + 1):
                handled.add(shapes[:n])
            i += hit.n

    suggestions = []
    for shapes, count in unhandled.most_common():
        if shapes in handled:
            continue
        suggestions.append((shapes_label(shapes), count))
        if len(suggestions) >= top:
            break
    return suggestions


# ---------------------------------------------------------------------------
# Shared EA / flag helpers for hand-written emitters
# ---------------------------------------------------------------------------

def _areg(n: int) -> str:
    return ea.areg(n)


def _d_long(n: int) -> str:
    return f'cpu().d[{n}]'


def _d_word(n: int) -> str:
    return f'cpu().dw({n})'


def _d_byte(n: int) -> str:
    return f'cpu().db({n})'


def _live_at(live_out: dict, addr: int):
    live = live_out.get(addr)
    return ALL if live is None else live


def _instrs_identical(instrs) -> bool:
    first = instrs[0]
    for ins in instrs[1:]:
        if not _instr_same(first, ins):
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
        for attr in ('reg', 'imm', 'disp', 'abs_value', 'special', 'reglist',
                     'index_reg', 'index_size'):
            if getattr(ea_a, attr, None) != getattr(ea_b, attr, None):
                return False
        if getattr(ea_a, 'index_is_addr', False) != getattr(ea_b, 'index_is_addr', False):
            return False
    return True


def _homogeneous_move_l_postinc_store(instrs, min_n: int = 4):
    """N× identical ``move.l Dn,(An)+`` → (n, {dn, an}) or None."""
    if len(instrs) < min_n:
        return None
    first = instrs[0]
    if first.mnemonic != 'move' or first.size != 'l' or len(first.eas) != 2:
        return None
    src, dst = first.eas
    if src.mode is not EAMode.DATA_REG or dst.mode is not EAMode.ADDR_POSTINC:
        return None
    # Consume the longest identical prefix.
    n = 1
    while n < len(instrs) and _instr_same(first, instrs[n]):
        n += 1
    if n < min_n:
        return None
    return n, {'dn': src.reg, 'an': dst.reg}


def _homogeneous_move_l_postinc_copy(instrs, min_n: int = 4):
    """N× identical ``move.l (As)+,(Ad)+`` → (n, {as_, ad}) or None."""
    if len(instrs) < min_n:
        return None
    first = instrs[0]
    if first.mnemonic != 'move' or first.size != 'l' or len(first.eas) != 2:
        return None
    src, dst = first.eas
    if src.mode is not EAMode.ADDR_POSTINC or dst.mode is not EAMode.ADDR_POSTINC:
        return None
    n = 1
    while n < len(instrs) and _instr_same(first, instrs[n]):
        n += 1
    if n < min_n:
        return None
    return n, {'as_': src.reg, 'ad': dst.reg}


def _imm_disp_store_chain(instrs, min_n: int = 2):
    """N× ``move.[bwl] #imm, d16(An)`` same An (imms/disps/sizes may differ)."""
    if len(instrs) < min_n:
        return None
    first = instrs[0]
    if first.mnemonic != 'move' or first.size not in ('b', 'w', 'l'):
        return None
    if len(first.eas) != 2:
        return None
    src0, dst0 = first.eas
    if src0.mode is not EAMode.IMMEDIATE or dst0.mode is not EAMode.ADDR_DISP:
        return None
    an = dst0.reg
    n = 0
    stores = []
    for ins in instrs:
        if ins.mnemonic != 'move' or ins.size not in ('b', 'w', 'l'):
            break
        if len(ins.eas) != 2:
            break
        src, dst = ins.eas
        if src.mode is not EAMode.IMMEDIATE or dst.mode is not EAMode.ADDR_DISP:
            break
        if dst.reg != an:
            break
        stores.append((ins.size, src.imm, dst.disp))
        n += 1
    if n < min_n:
        return None
    return n, {'an': an, 'stores': stores}


# ---------------------------------------------------------------------------
# Hand-written patterns (edit this section to add macros)
# ---------------------------------------------------------------------------

def _emit_flag_loop_tail(out: list[str], n: int, all_flag_bodies: list[list[str]]
                         ) -> None:
    """Append per-iteration / last-only CCR updates inside an open for-loop."""
    mid_flags = all_flag_bodies[:-1]
    last_flags = all_flag_bodies[-1]
    any_mid = any(mid_flags)
    if any_mid:
        if all(f == mid_flags[0] for f in mid_flags) and mid_flags[0] == last_flags:
            for stmt in last_flags:
                out.append(f'    {stmt}')
        else:
            out.append('    switch (_seq) {')
            for i, flags in enumerate(all_flag_bodies):
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


def _emit_memfill_long_postinc(m: PatternHit, live_out: dict) -> list[str]:
    """Custom C++: fill memory with Dn via post-increment An."""
    n = m.n
    dn, an = m.data['dn'], m.data['an']
    ar, dr = _areg(an), _d_long(dn)
    flag_bodies = [sem.move(dr, 'l', live=_live_at(live_out, a)) for a in m.addrs]

    out = [
        f'for (int _seq = 0; _seq < {n}; ++_seq) {{',
        '    BEFORE_INSTRUCTION',
        f'    memory().writeLong({ar}, {dr});',
        f'    {ar} += 4;',
    ]
    _emit_flag_loop_tail(out, n, flag_bodies)
    out.append('}')
    return out


register(Pattern(
    name='memfill_long_postinc',
    doc='N× move.l Dn,(An)+  — memory fill',
    match=lambda instrs: _homogeneous_move_l_postinc_store(instrs, min_n=4),
    emit=_emit_memfill_long_postinc,
))


def _emit_memcopy_long_postinc(m: PatternHit, live_out: dict) -> list[str]:
    """Custom C++: block copy via post-increment source and dest."""
    n = m.n
    as_, ad = m.data['as_'], m.data['ad']
    src_a, dst_a = _areg(as_), _areg(ad)
    # Temp ``t`` is scoped inside each iteration, after BEFORE (IRQ-safe).
    flag_bodies = [sem.move('t', 'l', live=_live_at(live_out, a)) for a in m.addrs]

    out = [
        f'for (int _seq = 0; _seq < {n}; ++_seq) {{',
        '    BEFORE_INSTRUCTION',
        f'    m_long t = memory().readLong({src_a});',
        f'    {src_a} += 4;',
        f'    memory().writeLong({dst_a}, t);',
        f'    {dst_a} += 4;',
    ]
    _emit_flag_loop_tail(out, n, flag_bodies)
    out.append('}')
    return out


register(Pattern(
    name='memcopy_long_postinc',
    doc='N× move.l (As)+,(Ad)+  — block copy',
    match=lambda instrs: _homogeneous_move_l_postinc_copy(instrs, min_n=4),
    emit=_emit_memcopy_long_postinc,
))


def _emit_imm_disp_store_chain(m: PatternHit, live_out: dict) -> list[str]:
    """Custom C++: structure / object field init via imm → d16(An) stores."""
    an = m.data['an']
    ar = _areg(an)
    write = {'b': 'writeByte', 'w': 'writeWord', 'l': 'writeLong'}
    cast = {'b': 'BYTE', 'w': 'WORD', 'l': 'LONG'}
    mask = {'b': 0xFF, 'w': 0xFFFF, 'l': 0xFFFFFFFF}
    out: list[str] = []
    for i, (size, imm, disp) in enumerate(m.data['stores']):
        addr = m.addrs[i]
        imm_hex = ea._hex(imm & mask[size])
        # Re-read An after every BEFORE (IRQ may change it).
        out.append(f'// ${addr:06X} move.{size} #{imm & mask[size]:X}, '
                   f'{disp}(a{an})')
        out.append('BEFORE_INSTRUCTION')
        out.append(
            f'memory().{write[size]}(({ar} + {disp}), {cast[size]}({imm_hex}));')
        out.extend(sem.move(f'{cast[size]}({imm_hex})', size,
                            live=_live_at(live_out, addr)))
    return out


register(Pattern(
    name='imm_disp_store_chain',
    doc='N× move.[bwl] #imm,d16(An)  — structure field init',
    match=lambda instrs: _imm_disp_store_chain(instrs, min_n=2),
    emit=_emit_imm_disp_store_chain,
))
