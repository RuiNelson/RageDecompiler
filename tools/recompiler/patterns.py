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


# ---------------------------------------------------------------------------
# Popular Streets of Rage idioms (ranked by frequency in the ROM)
# ---------------------------------------------------------------------------

def _moveq_imm_long(imm: int) -> str:
    """Sign-extended 8-bit moveq immediate as a C++ hex literal."""
    imm &= 0xFF
    return ea._hex(imm if imm < 0x80 else (imm | 0xFFFFFF00))


def _match_moveq_chain(instrs, min_n: int = 2):
    """N× consecutive ``moveq #imm,Dn`` (any Dn / imm)."""
    if len(instrs) < min_n or instrs[0].mnemonic != 'moveq':
        return None
    ops = []
    for ins in instrs:
        if ins.mnemonic != 'moveq' or len(ins.eas) != 2:
            break
        src, dst = ins.eas
        if src.mode is not EAMode.IMMEDIATE or dst.mode is not EAMode.DATA_REG:
            break
        ops.append((dst.reg, src.imm & 0xFF))
    n = len(ops)
    if n < min_n:
        return None
    return n, {'ops': ops}


def _emit_moveq_chain(m: PatternHit, live_out: dict) -> list[str]:
    """Custom C++: consecutive moveq — common multi-register prologue."""
    out: list[str] = []
    for i, (dn, imm) in enumerate(m.data['ops']):
        addr = m.addrs[i]
        val = _moveq_imm_long(imm)
        out.append(f'// ${addr:06X} moveq #{imm:02X}, d{dn}')
        out.append('BEFORE_INSTRUCTION')
        out.append(ea.write_dn(dn, 'l', val))
        out.extend(sem.move(val, 'l', live=_live_at(live_out, addr)))
    return out


register(Pattern(
    name='moveq_chain',
    doc='N× moveq #imm,Dn  — multi-register prologue',
    match=lambda instrs: _match_moveq_chain(instrs, min_n=2),
    emit=_emit_moveq_chain,
))


def _match_moveq_load_byte(instrs):
    """``moveq #imm,Dn`` ; ``move.b <ea>,Dn`` — classic zero/sign-extend byte load.

    Very common after ``moveq #0,Dn`` to clear the high bits before a byte read.
    Accepts d16(An) or (An)+ as the byte source; Dn must be the same.
    """
    if len(instrs) < 2:
        return None
    q, mb = instrs[0], instrs[1]
    if q.mnemonic != 'moveq' or mb.mnemonic != 'move' or mb.size != 'b':
        return None
    if len(q.eas) != 2 or len(mb.eas) != 2:
        return None
    q_imm, q_dn = q.eas[0], q.eas[1]
    src, dst = mb.eas
    if q_imm.mode is not EAMode.IMMEDIATE or q_dn.mode is not EAMode.DATA_REG:
        return None
    if dst.mode is not EAMode.DATA_REG or dst.reg != q_dn.reg:
        return None
    if src.mode is EAMode.ADDR_DISP:
        return 2, {
            'dn': dst.reg,
            'imm': q_imm.imm & 0xFF,
            'src_mode': 'disp',
            'an': src.reg,
            'disp': src.disp,
        }
    if src.mode is EAMode.ADDR_POSTINC:
        return 2, {
            'dn': dst.reg,
            'imm': q_imm.imm & 0xFF,
            'src_mode': 'postinc',
            'an': src.reg,
        }
    if src.mode is EAMode.ADDR_IND:
        return 2, {
            'dn': dst.reg,
            'imm': q_imm.imm & 0xFF,
            'src_mode': 'ind',
            'an': src.reg,
        }
    if src.mode is EAMode.ABS_W:
        return 2, {
            'dn': dst.reg,
            'imm': q_imm.imm & 0xFF,
            'src_mode': 'abs_w',
            'abs': src.abs_value,
        }
    return None


def _emit_moveq_load_byte(m: PatternHit, live_out: dict) -> list[str]:
    dn = m.data['dn']
    imm_val = _moveq_imm_long(m.data['imm'])
    a0, a1 = m.addrs
    out = [
        f'// ${a0:06X} moveq #{m.data["imm"]:02X}, d{dn}',
        'BEFORE_INSTRUCTION',
        ea.write_dn(dn, 'l', imm_val),
    ]
    out.extend(sem.move(imm_val, 'l', live=_live_at(live_out, a0)))

    mode = m.data['src_mode']
    if mode == 'disp':
        an, disp = m.data['an'], m.data['disp']
        src_expr = f'memory().readByte(({_areg(an)} + {disp}))'
        comment = f'move.b {disp}(a{an}), d{dn}'
    elif mode == 'postinc':
        an = m.data['an']
        step = ea.addr_step(an, 'b')
        src_expr = f'memory().readByte({_areg(an)})'
        comment = f'move.b (a{an})+, d{dn}'
    elif mode == 'ind':
        an = m.data['an']
        src_expr = f'memory().readByte({_areg(an)})'
        comment = f'move.b (a{an}), d{dn}'
    else:  # abs_w
        abs_hex = ea._hex(m.data['abs'] & 0xFFFFFF)
        src_expr = f'memory().readByte({abs_hex})'
        comment = f'move.b ({abs_hex}).w, d{dn}'

    out.append(f'// ${a1:06X} {comment}')
    out.append('BEFORE_INSTRUCTION')
    # Materialise the byte so flags and Dn write share one read (post-IRQ).
    out.append(f'm_byte t = {src_expr};')
    if mode == 'postinc':
        out.append(f'{_areg(m.data["an"])} += {ea.addr_step(m.data["an"], "b")};')
    out.append(ea.write_dn(dn, 'b', 't'))
    out.extend(sem.move('t', 'b', live=_live_at(live_out, a1)))
    return out


register(Pattern(
    name='moveq_load_byte',
    doc='moveq #imm,Dn ; move.b <ea>,Dn  — extend + load byte',
    match=_match_moveq_load_byte,
    emit=_emit_moveq_load_byte,
))


def _match_disp_word_copy_chain(instrs, min_n: int = 2):
    """N× ``move.w d16(An),d16(Am)`` — field-to-field word copies."""
    if len(instrs) < min_n:
        return None
    copies = []
    for ins in instrs:
        if ins.mnemonic != 'move' or ins.size != 'w' or len(ins.eas) != 2:
            break
        src, dst = ins.eas
        if src.mode is not EAMode.ADDR_DISP or dst.mode is not EAMode.ADDR_DISP:
            break
        copies.append((src.reg, src.disp, dst.reg, dst.disp))
    n = len(copies)
    if n < min_n:
        return None
    return n, {'copies': copies}


def _emit_disp_word_copy_chain(m: PatternHit, live_out: dict) -> list[str]:
    out: list[str] = []
    for i, (as_, sd, ad, dd) in enumerate(m.data['copies']):
        addr = m.addrs[i]
        out.append(
            f'// ${addr:06X} move.w {sd}(a{as_}), {dd}(a{ad})')
        out.append('BEFORE_INSTRUCTION')
        # Re-read both bases after BEFORE — IRQ may change either An.
        out.append(
            f'm_word t = memory().readWord(({_areg(as_)} + {sd}));')
        out.append(
            f'memory().writeWord(({_areg(ad)} + {dd}), t);')
        out.extend(sem.move('t', 'w', live=_live_at(live_out, addr)))
    return out


register(Pattern(
    name='disp_word_copy_chain',
    doc='N× move.w d16(An),d16(Am)  — structure field copy',
    match=lambda instrs: _match_disp_word_copy_chain(instrs, min_n=2),
    emit=_emit_disp_word_copy_chain,
))


def _match_lea_abs_then_moveq(instrs):
    """``lea abs.w/.l, An`` ; ``moveq #imm, Dn`` — pointer + constant setup."""
    if len(instrs) < 2:
        return None
    lea, mq = instrs[0], instrs[1]
    if lea.mnemonic != 'lea' or mq.mnemonic != 'moveq':
        return None
    if len(lea.eas) != 2 or len(mq.eas) != 2:
        return None
    src, an = lea.eas
    imm, dn = mq.eas
    if an.mode is not EAMode.ADDR_REG or dn.mode is not EAMode.DATA_REG:
        return None
    if imm.mode is not EAMode.IMMEDIATE:
        return None
    if src.mode is EAMode.ABS_W:
        abs_val = src.abs_value
        form = 'w'
    elif src.mode is EAMode.ABS_L:
        abs_val = src.abs_value
        form = 'l'
    else:
        return None
    if abs_val is None:
        return None
    return 2, {
        'an': an.reg,
        'abs': abs_val & 0xFFFFFF,
        'form': form,
        'dn': dn.reg,
        'imm': imm.imm & 0xFF,
    }


def _emit_lea_abs_then_moveq(m: PatternHit, live_out: dict) -> list[str]:
    an, dn = m.data['an'], m.data['dn']
    abs_hex = ea._hex(m.data['abs'])
    imm_val = _moveq_imm_long(m.data['imm'])
    a0, a1 = m.addrs
    # lea does not touch CCR.
    return [
        f'// ${a0:06X} lea.l ({abs_hex}).{m.data["form"]}, a{an}',
        'BEFORE_INSTRUCTION',
        f'{_areg(an)} = {abs_hex};',
        f'// ${a1:06X} moveq #{m.data["imm"]:02X}, d{dn}',
        'BEFORE_INSTRUCTION',
        ea.write_dn(dn, 'l', imm_val),
        *sem.move(imm_val, 'l', live=_live_at(live_out, a1)),
    ]


register(Pattern(
    name='lea_abs_moveq',
    doc='lea abs,An ; moveq #imm,Dn  — pointer + constant setup',
    match=_match_lea_abs_then_moveq,
    emit=_emit_lea_abs_then_moveq,
))


def _match_addw_self_chain(instrs, min_n: int = 2):
    """N× identical ``add.w Dn,Dn`` — left-shift by addition (×2 per op)."""
    if len(instrs) < min_n:
        return None
    first = instrs[0]
    if first.mnemonic != 'add' or first.size != 'w' or len(first.eas) != 2:
        return None
    src, dst = first.eas
    if (src.mode is not EAMode.DATA_REG or dst.mode is not EAMode.DATA_REG
            or src.reg != dst.reg):
        return None
    n = 1
    while n < len(instrs) and _instr_same(first, instrs[n]):
        n += 1
    if n < min_n:
        return None
    return n, {'dn': src.reg}


def _emit_addw_self_chain(m: PatternHit, live_out: dict) -> list[str]:
    from tools.recompiler.ea_codegen import TempPool

    dn = m.data['dn']
    n = m.n
    # Unrolled: each step re-reads Dn after BEFORE (IRQ-safe).  Using sem.add
    # keeps V/C/X exactly as a single add.w Dn,Dn would.
    out: list[str] = []
    for i, addr in enumerate(m.addrs):
        out.append(f'// ${addr:06X} add.w d{dn}, d{dn}')
        out.append('BEFORE_INSTRUCTION')
        tmp = TempPool(addr)
        # Operate on the word view; write back via setDw.
        dw = f'_dw{i}'
        out.append(f'm_word {dw} = {ea.read_dn(dn, "w")};')
        tmp.types[dw] = 'w'
        body = sem.add(dw, dw, 'w', tmp, live=_live_at(live_out, addr))
        out.extend(body)
        out.append(ea.write_dn(dn, 'w', dw, tmp.types))
    return out


register(Pattern(
    name='addw_self_chain',
    doc='N× add.w Dn,Dn  — shift left via add',
    match=lambda instrs: _match_addw_self_chain(instrs, min_n=2),
    emit=_emit_addw_self_chain,
))


def _match_memfill_long_ind(instrs, min_n: int = 2):
    """N× identical ``move.l Dn,(An)`` — repeated store to same address."""
    if len(instrs) < min_n:
        return None
    first = instrs[0]
    if first.mnemonic != 'move' or first.size != 'l' or len(first.eas) != 2:
        return None
    src, dst = first.eas
    if src.mode is not EAMode.DATA_REG or dst.mode is not EAMode.ADDR_IND:
        return None
    n = 1
    while n < len(instrs) and _instr_same(first, instrs[n]):
        n += 1
    if n < min_n:
        return None
    return n, {'dn': src.reg, 'an': dst.reg}


def _emit_memfill_long_ind(m: PatternHit, live_out: dict) -> list[str]:
    """Same address rewritten N times (game wait/strobe idioms + unrolled fills)."""
    n = m.n
    dn, an = m.data['dn'], m.data['an']
    ar, dr = _areg(an), _d_long(dn)
    flag_bodies = [sem.move(dr, 'l', live=_live_at(live_out, a)) for a in m.addrs]
    if n >= 4:
        out = [
            f'for (int _seq = 0; _seq < {n}; ++_seq) {{',
            '    BEFORE_INSTRUCTION',
            f'    memory().writeLong({ar}, {dr});',
        ]
        _emit_flag_loop_tail(out, n, flag_bodies)
        out.append('}')
        return out
    out = []
    for i, addr in enumerate(m.addrs):
        out.append(f'// ${addr:06X} move.l d{dn}, (a{an})')
        out.append('BEFORE_INSTRUCTION')
        out.append(f'memory().writeLong({ar}, {dr});')
        out.extend(flag_bodies[i])
    return out


register(Pattern(
    name='memfill_long_ind',
    doc='N× move.l Dn,(An)  — repeated long store',
    match=lambda instrs: _match_memfill_long_ind(instrs, min_n=2),
    emit=_emit_memfill_long_ind,
))


def _match_move_w_imm_ind_chain(instrs, min_n: int = 2):
    """N× ``move.w #imm,(An)`` same An — VDP / IO word pokes."""
    if len(instrs) < min_n:
        return None
    first = instrs[0]
    if first.mnemonic != 'move' or first.size != 'w' or len(first.eas) != 2:
        return None
    src0, dst0 = first.eas
    if src0.mode is not EAMode.IMMEDIATE or dst0.mode is not EAMode.ADDR_IND:
        return None
    an = dst0.reg
    ops = []
    for ins in instrs:
        if ins.mnemonic != 'move' or ins.size != 'w' or len(ins.eas) != 2:
            break
        src, dst = ins.eas
        if src.mode is not EAMode.IMMEDIATE or dst.mode is not EAMode.ADDR_IND:
            break
        if dst.reg != an:
            break
        ops.append(src.imm & 0xFFFF)
    n = len(ops)
    if n < min_n:
        return None
    return n, {'an': an, 'imms': ops}


def _emit_move_w_imm_ind_chain(m: PatternHit, live_out: dict) -> list[str]:
    an = m.data['an']
    ar = _areg(an)
    out: list[str] = []
    for i, imm in enumerate(m.data['imms']):
        addr = m.addrs[i]
        imm_hex = ea._hex(imm)
        out.append(f'// ${addr:06X} move.w #{imm:04X}, (a{an})')
        out.append('BEFORE_INSTRUCTION')
        out.append(f'memory().writeWord({ar}, WORD({imm_hex}));')
        out.extend(sem.move(f'WORD({imm_hex})', 'w',
                            live=_live_at(live_out, addr)))
    return out


register(Pattern(
    name='move_w_imm_ind_chain',
    doc='N× move.w #imm,(An)  — word poke chain',
    match=lambda instrs: _match_move_w_imm_ind_chain(instrs, min_n=2),
    emit=_emit_move_w_imm_ind_chain,
))


def _match_nop_run(instrs, min_n: int = 2):
    if len(instrs) < min_n:
        return None
    n = 0
    for ins in instrs:
        if ins.mnemonic != 'nop':
            break
        n += 1
    if n < min_n:
        return None
    return n, {}


def _emit_nop_run(m: PatternHit, live_out: dict) -> list[str]:
    # Still pace/IRQ once per original nop — timing-sensitive wait loops.
    out: list[str] = []
    for addr in m.addrs:
        out.append(f'// ${addr:06X} nop')
        out.append('BEFORE_INSTRUCTION')
        out.append('(void)0;')
    return out


register(Pattern(
    name='nop_run',
    doc='N× nop  — timed delay / alignment',
    match=lambda instrs: _match_nop_run(instrs, min_n=2),
    emit=_emit_nop_run,
))


def _match_lea_abs_chain(instrs, min_n: int = 2):
    """N× ``lea abs.w/.l, An`` — successive absolute pointer loads."""
    if len(instrs) < min_n:
        return None
    ops = []
    for ins in instrs:
        if ins.mnemonic != 'lea' or len(ins.eas) != 2:
            break
        src, an = ins.eas
        if an.mode is not EAMode.ADDR_REG:
            break
        if src.mode is EAMode.ABS_W:
            form = 'w'
        elif src.mode is EAMode.ABS_L:
            form = 'l'
        else:
            break
        if src.abs_value is None:
            break
        ops.append((an.reg, src.abs_value & 0xFFFFFF, form))
    n = len(ops)
    if n < min_n:
        return None
    return n, {'ops': ops}


def _emit_lea_abs_chain(m: PatternHit, live_out: dict) -> list[str]:
    out: list[str] = []
    for i, (an, abs_val, form) in enumerate(m.data['ops']):
        addr = m.addrs[i]
        abs_hex = ea._hex(abs_val)
        out.append(f'// ${addr:06X} lea.l ({abs_hex}).{form}, a{an}')
        out.append('BEFORE_INSTRUCTION')
        out.append(f'{_areg(an)} = {abs_hex};')
    return out


register(Pattern(
    name='lea_abs_chain',
    doc='N× lea abs,An  — absolute pointer chain',
    match=lambda instrs: _match_lea_abs_chain(instrs, min_n=2),
    emit=_emit_lea_abs_chain,
))


def _match_move_w_imm_then_moveq(instrs):
    """``move.w #imm,Dn`` ; ``moveq #imm,Dm`` — mixed constant setup."""
    if len(instrs) < 2:
        return None
    mw, mq = instrs[0], instrs[1]
    if mw.mnemonic != 'move' or mw.size != 'w' or mq.mnemonic != 'moveq':
        return None
    if len(mw.eas) != 2 or len(mq.eas) != 2:
        return None
    src, dst = mw.eas
    qimm, qdn = mq.eas
    if src.mode is not EAMode.IMMEDIATE or dst.mode is not EAMode.DATA_REG:
        return None
    if qimm.mode is not EAMode.IMMEDIATE or qdn.mode is not EAMode.DATA_REG:
        return None
    return 2, {
        'dn_w': dst.reg,
        'imm_w': src.imm & 0xFFFF,
        'dn_q': qdn.reg,
        'imm_q': qimm.imm & 0xFF,
    }


def _emit_move_w_imm_then_moveq(m: PatternHit, live_out: dict) -> list[str]:
    a0, a1 = m.addrs
    imm_w = ea._hex(m.data['imm_w'])
    imm_q = _moveq_imm_long(m.data['imm_q'])
    dn_w, dn_q = m.data['dn_w'], m.data['dn_q']
    out = [
        f'// ${a0:06X} move.w #{m.data["imm_w"]:04X}, d{dn_w}',
        'BEFORE_INSTRUCTION',
        ea.write_dn(dn_w, 'w', imm_w),
    ]
    out.extend(sem.move(imm_w, 'w', live=_live_at(live_out, a0)))
    out += [
        f'// ${a1:06X} moveq #{m.data["imm_q"]:02X}, d{dn_q}',
        'BEFORE_INSTRUCTION',
        ea.write_dn(dn_q, 'l', imm_q),
    ]
    out.extend(sem.move(imm_q, 'l', live=_live_at(live_out, a1)))
    return out


register(Pattern(
    name='move_w_imm_moveq',
    doc='move.w #imm,Dn ; moveq #imm,Dm  — mixed constant setup',
    match=_match_move_w_imm_then_moveq,
    emit=_emit_move_w_imm_then_moveq,
))


def _match_imm_disp_then_moveq(instrs):
    """``move.[bwl] #imm,d16(An)`` ; ``moveq #imm,Dn`` — store then constant."""
    if len(instrs) < 2:
        return None
    mv, mq = instrs[0], instrs[1]
    if mv.mnemonic != 'move' or mv.size not in ('b', 'w', 'l'):
        return None
    if mq.mnemonic != 'moveq' or len(mv.eas) != 2 or len(mq.eas) != 2:
        return None
    src, dst = mv.eas
    qimm, qdn = mq.eas
    if src.mode is not EAMode.IMMEDIATE or dst.mode is not EAMode.ADDR_DISP:
        return None
    if qimm.mode is not EAMode.IMMEDIATE or qdn.mode is not EAMode.DATA_REG:
        return None
    return 2, {
        'size': mv.size,
        'store_imm': src.imm,
        'an': dst.reg,
        'disp': dst.disp,
        'dn': qdn.reg,
        'qimm': qimm.imm & 0xFF,
    }


def _emit_imm_disp_then_moveq(m: PatternHit, live_out: dict) -> list[str]:
    size = m.data['size']
    mask = {'b': 0xFF, 'w': 0xFFFF, 'l': 0xFFFFFFFF}[size]
    write = {'b': 'writeByte', 'w': 'writeWord', 'l': 'writeLong'}[size]
    cast = {'b': 'BYTE', 'w': 'WORD', 'l': 'LONG'}[size]
    imm_s = ea._hex(m.data['store_imm'] & mask)
    imm_q = _moveq_imm_long(m.data['qimm'])
    an, disp, dn = m.data['an'], m.data['disp'], m.data['dn']
    a0, a1 = m.addrs
    out = [
        f'// ${a0:06X} move.{size} #{m.data["store_imm"] & mask:X}, '
        f'{disp}(a{an})',
        'BEFORE_INSTRUCTION',
        f'memory().{write}(({_areg(an)} + {disp}), {cast}({imm_s}));',
    ]
    out.extend(sem.move(f'{cast}({imm_s})', size, live=_live_at(live_out, a0)))
    out += [
        f'// ${a1:06X} moveq #{m.data["qimm"]:02X}, d{dn}',
        'BEFORE_INSTRUCTION',
        ea.write_dn(dn, 'l', imm_q),
    ]
    out.extend(sem.move(imm_q, 'l', live=_live_at(live_out, a1)))
    return out


register(Pattern(
    name='imm_disp_then_moveq',
    doc='move #imm,d16(An) ; moveq #imm,Dn  — store then constant',
    match=_match_imm_disp_then_moveq,
    emit=_emit_imm_disp_then_moveq,
))


def _match_abs_byte_store_chain(instrs, min_n: int = 2):
    """N× ``move.b #imm,abs.w`` — absolute byte pokes (IO / RAM flags)."""
    if len(instrs) < min_n:
        return None
    ops = []
    for ins in instrs:
        if ins.mnemonic != 'move' or ins.size != 'b' or len(ins.eas) != 2:
            break
        src, dst = ins.eas
        if src.mode is not EAMode.IMMEDIATE or dst.mode is not EAMode.ABS_W:
            break
        if dst.abs_value is None:
            break
        ops.append((src.imm & 0xFF, dst.abs_value & 0xFFFFFF))
    n = len(ops)
    if n < min_n:
        return None
    return n, {'ops': ops}


def _emit_abs_byte_store_chain(m: PatternHit, live_out: dict) -> list[str]:
    out: list[str] = []
    for i, (imm, abs_val) in enumerate(m.data['ops']):
        addr = m.addrs[i]
        imm_hex = ea._hex(imm)
        abs_hex = ea._hex(abs_val)
        out.append(f'// ${addr:06X} move.b #{imm:02X}, ({abs_hex}).w')
        out.append('BEFORE_INSTRUCTION')
        out.append(f'memory().writeByte({abs_hex}, BYTE({imm_hex}));')
        out.extend(sem.move(f'BYTE({imm_hex})', 'b',
                            live=_live_at(live_out, addr)))
    return out


register(Pattern(
    name='abs_byte_store_chain',
    doc='N× move.b #imm,abs.w  — absolute byte poke chain',
    match=lambda instrs: _match_abs_byte_store_chain(instrs, min_n=2),
    emit=_emit_abs_byte_store_chain,
))


def _match_load_word_disp_chain(instrs, min_n: int = 2):
    """N× ``move.w d16(An),Dn`` — successive word field loads into Dns."""
    if len(instrs) < min_n:
        return None
    ops = []
    for ins in instrs:
        if ins.mnemonic != 'move' or ins.size != 'w' or len(ins.eas) != 2:
            break
        src, dst = ins.eas
        if src.mode is not EAMode.ADDR_DISP or dst.mode is not EAMode.DATA_REG:
            break
        ops.append((src.reg, src.disp, dst.reg))
    n = len(ops)
    if n < min_n:
        return None
    return n, {'ops': ops}


def _emit_load_word_disp_chain(m: PatternHit, live_out: dict) -> list[str]:
    out: list[str] = []
    for i, (an, disp, dn) in enumerate(m.data['ops']):
        addr = m.addrs[i]
        out.append(f'// ${addr:06X} move.w {disp}(a{an}), d{dn}')
        out.append('BEFORE_INSTRUCTION')
        out.append(f'm_word t = memory().readWord(({_areg(an)} + {disp}));')
        out.append(ea.write_dn(dn, 'w', 't'))
        out.extend(sem.move('t', 'w', live=_live_at(live_out, addr)))
    return out


register(Pattern(
    name='load_word_disp_chain',
    doc='N× move.w d16(An),Dn  — word field load chain',
    match=lambda instrs: _match_load_word_disp_chain(instrs, min_n=2),
    emit=_emit_load_word_disp_chain,
))


def _match_bset_imm_disp_then_moveq(instrs):
    """``bset #imm,d16(An)`` ; ``moveq #imm,Dn`` — flag bit then clear/reg setup."""
    if len(instrs) < 2:
        return None
    bs, mq = instrs[0], instrs[1]
    if bs.mnemonic != 'bset' or mq.mnemonic != 'moveq':
        return None
    if len(bs.eas) != 2 or len(mq.eas) != 2:
        return None
    bit, mem = bs.eas
    qimm, qdn = mq.eas
    if bit.mode is not EAMode.IMMEDIATE or mem.mode is not EAMode.ADDR_DISP:
        return None
    if qimm.mode is not EAMode.IMMEDIATE or qdn.mode is not EAMode.DATA_REG:
        return None
    return 2, {
        'bit': bit.imm & 0xFF,
        'an': mem.reg,
        'disp': mem.disp,
        'dn': qdn.reg,
        'qimm': qimm.imm & 0xFF,
    }


def _emit_bset_imm_disp_then_moveq(m: PatternHit, live_out: dict) -> list[str]:
    """Keep bset semantics via the normal opcode lowerer for the bit op."""
    # Use the original first instruction through opcodes for exact bset CCR/Z.
    bset_instr = m.instrs[0]
    a0, a1 = m.addrs
    body = opcodes.emit_dataop(bset_instr, live_flags=_live_at(live_out, a0))
    if body is None:
        raise opcodes.Unsupported('bset')
    imm_q = _moveq_imm_long(m.data['qimm'])
    dn = m.data['dn']
    out = [f'// ${a0:06X} bset #{m.data["bit"]}, {m.data["disp"]}(a{m.data["an"]})',
           'BEFORE_INSTRUCTION', *body,
           f'// ${a1:06X} moveq #{m.data["qimm"]:02X}, d{dn}',
           'BEFORE_INSTRUCTION',
           ea.write_dn(dn, 'l', imm_q)]
    out.extend(sem.move(imm_q, 'l', live=_live_at(live_out, a1)))
    return out


register(Pattern(
    name='bset_disp_then_moveq',
    doc='bset #imm,d16(An) ; moveq #imm,Dn  — set flag bit + constant',
    match=_match_bset_imm_disp_then_moveq,
    emit=_emit_bset_imm_disp_then_moveq,
))
