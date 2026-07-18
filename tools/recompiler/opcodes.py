"""Instruction → C++ statement generation for data-processing opcodes.

Each handler emits C++ that materializes operands via ``ea_codegen`` and then
splices direct C++ statements for the opcode/CCR semantics. The generated source
is intentionally self-contained; it no longer depends on a shared per-opcode
macro header.

Control-flow instructions (bra/bcc/bsr/jsr/jmp/dbcc/rts/rte/rtr) are emitted by
``generator`` (they need region context); ``emit_dataop`` returns ``None`` for
those. ``movem`` is also expanded by ``generator`` (a memory block transfer, not
a value op). Anything else not implemented raises :class:`Unsupported` so the
generator fails loudly — the recompiler must translate 100% of what it sees.
"""

from tools.disassembler.instruction import EAMode
from tools.recompiler import ea_codegen as ea
from tools.recompiler import cpp_semantics as sem
from tools.recompiler.ccr_liveness import ALL, NZVC
from tools.recompiler.ea_codegen import EAGenError, TempPool

_SUF = {'b': 'B', 'w': 'W', 'l': 'L'}
_NBITS = {'b': 8, 'w': 16, 'l': 32}

FLOW_MNEMONICS = {
    'bra', 'bsr', 'jmp', 'jsr', 'rts', 'rte', 'rtr',
    'bhi', 'bls', 'bcc', 'bcs', 'bne', 'beq', 'bvc', 'bvs',
    'bpl', 'bmi', 'bge', 'blt', 'bgt', 'ble',
    'dbt', 'dbf', 'dbra', 'dbhi', 'dbls', 'dbcc', 'dbcs', 'dbne', 'dbeq',
    'dbvc', 'dbvs', 'dbpl', 'dbmi', 'dbge', 'dblt', 'dbgt', 'dble',
}

# Emitted by generator (memory block transfer, not a value macro).
GENERATOR_MNEMONICS = {'movem'}

# Scc — set a byte to 0xFF/0x00 per the condition. Mapped to condition numbers.
SCC = {'st': 0, 'sf': 1, 'shi': 2, 'sls': 3, 'scc': 4, 'scs': 5, 'sne': 6,
       'seq': 7, 'svc': 8, 'svs': 9, 'spl': 10, 'smi': 11, 'sge': 12,
       'slt': 13, 'sgt': 14, 'sle': 15}


class Unsupported(Exception):
    """An opcode this generator does not yet implement."""


def _sized(instr) -> str:
    return instr.size or 'w'


def _signext_to_long(expr: str, size: str) -> str:
    return ea.signext_to_long(expr, size)


def emit_dataop(instr, live_flags=None):
    """Lower *instr* to C++ statements.

    ``live_flags`` is the set of CCR flags that are still observed after this
    instruction (see ``ccr_liveness``).  ``None`` means all flags are live.
    """
    m = instr.mnemonic
    if m in FLOW_MNEMONICS or m in GENERATOR_MNEMONICS:
        return None
    if m in SCC:
        return _scc(instr, TempPool(instr.address), SCC[m])
    handler = _HANDLERS.get(m)
    if handler is None:
        raise Unsupported(m)
    return handler(instr, TempPool(instr.address), live_flags)


# ---------------------------------------------------------------------------
# Special-register helpers (move/andi/ori/eori to sr or ccr)
# ---------------------------------------------------------------------------

def _is_special(e):
    return e.mode == EAMode.SPECIAL_REG


def _special_size(e):
    """move to/from usp is long; sr/ccr are word."""
    return 'l' if e.special == 'usp' else 'w'


def _special_src_expr(e):
    if e.special == 'sr':
        return 'cpu().status()'
    if e.special == 'ccr':
        return 'cpu().ccr()'
    if e.special == 'usp':
        return 'cpu().usp'
    raise EAGenError(f'read from special register {e.special}')


def _special_write(e, value):
    if e.special == 'sr':
        return [f'cpu().setStatus({ea._cast("w", value)});']
    if e.special == 'ccr':
        return [f'cpu().setCCR({ea._cast("w", value)});']
    if e.special == 'usp':
        return [f'cpu().usp = {ea._cast("l", value)};']
    raise EAGenError(f'write to special register {e.special}')


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _move(instr, tmp, live=None):
    size = _sized(instr)
    src, dst = instr.eas[0], instr.eas[1]

    # move to/from a special register (SR / CCR / USP).
    if _is_special(dst):
        ssz = _special_size(dst)
        s_stmts, sval = (([], _special_src_expr(src)) if _is_special(src)
                         else ea.read_ea(src, ssz, tmp))
        return s_stmts + _special_write(dst, sval)
    if _is_special(src):
        ssz = _special_size(src)
        v = tmp.fresh()
        stmts = [f'{ea._CTYPE[ssz]} {v} = {_special_src_expr(src)};']
        stmts += ea.write_ea(dst, ssz, v, tmp)
        return stmts

    stmts, val = ea.read_ea(src, size, tmp)
    if dst.mode == EAMode.ADDR_REG:                # movea — sign-extend, no flags
        if size == 'b':
            raise Unsupported('movea.b')           # invalid on 68000
        return stmts + sem.movea(ea.areg(dst.reg), val, size, tmp)
    # Use the EA value directly — read_ea already materializes side effects
    # into a temp when needed; a second t1 = t0 copy only adds noise.
    stmts = list(stmts)
    stmts += ea.write_ea(dst, size, val, tmp)
    return stmts + sem.move(val, size, live=live)


def _moveq(instr, tmp, live=None):
    src, dst = instr.eas[0], instr.eas[1]
    imm = src.imm & 0xFF
    # Sign-extend the 8-bit immediate to 32 bits as a single hex literal.
    val = ea._hex(imm if imm < 0x80 else (imm | 0xFFFFFF00))
    return [ea.write_dn(dst.reg, 'l', val)] + sem.move(val, 'l', live=live)


def _arith(instr, tmp, op, live=None):
    """add / sub / cmp families (op in {'ADD','SUB','CMP'})."""
    size = _sized(instr)
    src, dst = instr.eas[0], instr.eas[1]

    if dst.mode == EAMode.ADDR_REG:               # adda / suba / cmpa
        if size == 'b':
            raise Unsupported(f'{op.lower()}a.b') # invalid on 68000
        s_stmts, sval = ea.read_ea(src, size, tmp)
        ar = ea.areg(dst.reg)
        if op == 'CMP':
            return s_stmts + sem.cmpa(ar, sval, size, tmp, live=live)
        return s_stmts + (sem.adda(ar, sval, size, tmp) if op == 'ADD'
                          else sem.suba(ar, sval, size, tmp))

    if op == 'CMP':
        # When all compare flags are dead, skip the compare entirely and only
        # preserve addressing-mode side effects (postinc / predec).
        live_set = ALL if live is None else frozenset(live)
        if not (live_set & NZVC):
            return _touch_ea(src, size, tmp) + _touch_ea(dst, size, tmp)
        s_stmts, sval = ea.read_ea(src, size, tmp)
        d_stmts, dval = ea.read_ea(dst, size, tmp)
        return s_stmts + d_stmts + sem.cmp(dval, sval, size, tmp, live=live)
    s_stmts, sval = ea.read_ea(src, size, tmp)
    pre, r, post = ea.rmw_ea(dst, size, tmp)
    op_stmts = (sem.add(r, sval, size, tmp, live=live) if op == 'ADD'
                else sem.sub(r, sval, size, tmp, live=live))
    return s_stmts + pre + op_stmts + post


def _add(instr, tmp, live=None): return _arith(instr, tmp, 'ADD', live)
def _sub(instr, tmp, live=None): return _arith(instr, tmp, 'SUB', live)
def _cmp(instr, tmp, live=None): return _arith(instr, tmp, 'CMP', live)


def _logic(instr, tmp, op, live=None):
    size = _sized(instr)
    src, dst = instr.eas[0], instr.eas[1]
    # andi/ori/eori to SR or CCR.
    if _is_special(dst):
        s_stmts, sval = ea.read_ea(src, 'w', tmp)
        cur = _special_src_expr(dst)
        cxx = {'AND': '&', 'OR': '|', 'EOR': '^'}[op]
        return s_stmts + _special_write(dst, f'({cur} {cxx} {sval})')
    s_stmts, sval = ea.read_ea(src, size, tmp)
    pre, r, post = ea.rmw_ea(dst, size, tmp)
    return s_stmts + pre + sem.logic_op(r, sval, size, op, live=live, tmp=tmp) + post


def _and(instr, tmp, live=None): return _logic(instr, tmp, 'AND', live)
def _or(instr, tmp, live=None):  return _logic(instr, tmp, 'OR', live)
def _eor(instr, tmp, live=None): return _logic(instr, tmp, 'EOR', live)


def _ea_has_side_effects(e) -> bool:
    return e.mode in (EAMode.ADDR_POSTINC, EAMode.ADDR_PREDEC)


def _touch_ea(e, size, tmp):
    """Emit only addressing-mode side effects (postinc/predec)."""
    if _ea_has_side_effects(e):
        stmts, _ = ea.read_ea(e, size, tmp)
        return stmts
    return []


def _tst(instr, tmp, live=None):
    size = _sized(instr)
    live_set = ALL if live is None else frozenset(live)
    if not (live_set & NZVC):
        return _touch_ea(instr.eas[0], size, tmp)
    stmts, val = ea.read_ea(instr.eas[0], size, tmp)
    return stmts + sem.logical(val, size, live=live)


def _clr(instr, tmp, live=None):
    size = _sized(instr)
    r = tmp.fresh()
    stmts = [f'{ea._CTYPE[size]} {r} = 0;'] + sem.clr(r, size, live=live)
    return stmts + ea.write_ea(instr.eas[0], size, r, tmp)


def _lea(instr, tmp, live=None):
    src, dst = instr.eas[0], instr.eas[1]
    setup, addr = ea.address_of(src, tmp)
    return setup + [f'{ea.areg(dst.reg)} = {ea._cast("l", addr)};']


def _pea(instr, tmp, live=None):
    setup, addr = ea.address_of(instr.eas[0], tmp)
    return setup + [
        'cpu().ssp -= 4;',
        f'memory().writeLong(cpu().ssp, {ea._cast("l", addr)});',
    ]


def _unary(instr, tmp, macro, live=None):
    """neg / not — read-modify-write a single operand via a no-arg macro."""
    size = _sized(instr)
    pre, r, post = ea.rmw_ea(instr.eas[0], size, tmp)
    op = (sem.neg(r, size, tmp, live=live) if macro == 'NEG'
          else sem.not_op(r, size, live=live, tmp=tmp))
    return pre + op + post


def _neg(instr, tmp, live=None): return _unary(instr, tmp, 'NEG', live)
def _not(instr, tmp, live=None): return _unary(instr, tmp, 'NOT', live)


def _swap(instr, tmp, live=None):
    n = instr.eas[0].reg
    r = tmp.fresh()
    return [f'm_long {r} = {ea.read_dn(n, "l")};',
            *sem.swap(r, live=live),
            ea.write_dn(n, 'l', r)]


def _ext(instr, tmp, live=None):
    n = instr.eas[0].reg
    return sem.ext(n, instr.size or 'w', tmp, live=live)


def _shift(instr, tmp, macro, live=None):
    """Shift/rotate. Forms: '<op> #cnt, Dn' / '<op> Dm, Dn' / '<op> <ea>' (×1)."""
    size = _sized(instr)
    if len(instr.eas) == 1:                       # memory shift, count = 1
        pre, r, post = ea.rmw_ea(instr.eas[0], size, tmp)
        return pre + sem.shift(r, '1', size, macro, tmp, live=live) + post
    count, dst = instr.eas[0], instr.eas[1]
    if count.mode == EAMode.IMMEDIATE:
        setup, cnt = [], str((count.imm - 1) % 8 + 1)  # immediate count is 1..8
    else:
        # Register count: Dn mod 64; a count of zero shifts nothing.
        setup, cnt = sem.reg_shift_count(count.reg, tmp)
    pre, r, post = ea.rmw_ea(dst, size, tmp)
    return setup + pre + \
        sem.shift(r, cnt, size, macro, tmp,
                  count_may_be_zero=count.mode != EAMode.IMMEDIATE,
                  live=live) + post


def _muldiv(instr, tmp, macro, live=None):
    """mulu/muls (16×16→Dn long) and divu/divs (Dn 32 ÷ src16 → Dn)."""
    src, dst = instr.eas[0], instr.eas[1]
    s_stmts, sval = ea.read_ea(src, 'w', tmp)
    if macro.startswith('MUL'):
        dexpr = ea.read_dn(dst.reg, 'w')
    else:
        dexpr = ea.read_dn(dst.reg, 'l')
    op_stmts, result = sem.muldiv(dexpr, sval, macro, tmp, live=live)
    return s_stmts + op_stmts + [ea.write_dn(dst.reg, 'l', result)]


def _bitop(instr, tmp, kind, live=None):
    bit_ea, dst = instr.eas[0], instr.eas[1]
    data_reg = dst.mode == EAMode.DATA_REG
    size = 'l' if data_reg else 'b'
    modulo = 32 if data_reg else 8
    if bit_ea.mode == EAMode.IMMEDIATE:
        bit = str(bit_ea.imm % modulo)
    else:
        bit = f'(cpu().d[{bit_ea.reg}] % {modulo})'
    if kind == 'BTST':
        stmts, val = ea.read_ea(dst, size, tmp)
        return stmts + sem.bitop(val, bit, kind, size, live=live, tmp=tmp)
    pre, r, post = ea.rmw_ea(dst, size, tmp)
    return pre + sem.bitop(r, bit, kind, size, live=live, tmp=tmp) + post


def _scc(instr, tmp, cc):
    r = tmp.fresh()
    stmts = [f'm_byte {r} = 0;', *sem.scc(r, cc)]
    return stmts + ea.write_ea(instr.eas[0], 'b', r, tmp)


def _reg_lvalue(e):
    if e.mode == EAMode.DATA_REG:
        return f'cpu().d[{e.reg}]'
    if e.mode == EAMode.ADDR_REG:
        return ea.areg(e.reg)
    raise EAGenError(f'exg operand is not a register: {e.mode}')


def _exg(instr, tmp, live=None):
    a = _reg_lvalue(instr.eas[0])
    b = _reg_lvalue(instr.eas[1])
    t = tmp.fresh()
    return [f'm_long {t} = {a};', f'{a} = {b};', f'{b} = {t};']


def _bcd(instr, tmp, macro, live=None):
    src, dst = instr.eas[0], instr.eas[1]
    s_stmts, sval = ea.read_ea(src, 'b', tmp)
    pre, r, post = ea.rmw_ea(dst, 'b', tmp)
    return s_stmts + pre + sem.bcd(r, sval, macro, tmp, live=live) + post


def _nbcd(instr, tmp, live=None):
    pre, r, post = ea.rmw_ea(instr.eas[0], 'b', tmp)
    return pre + sem.nbcd(r, tmp, live=live) + post


def _negx(instr, tmp, live=None):
    size = _sized(instr)
    pre, r, post = ea.rmw_ea(instr.eas[0], size, tmp)
    return pre + sem.negx(r, size, tmp, live=live) + post


def _movep(instr, tmp, live=None):
    """Transfer bytes between Dn and alternating memory addresses. No CCR effect."""
    size = _sized(instr)  # 'w' or 'l'
    src, dst = instr.eas[0], instr.eas[1]
    if src.mode == EAMode.DATA_REG:          # reg → mem
        dn = src.reg
        setup, base = ea.address_of(dst, tmp)
        b = tmp.fresh()
        stmts = setup + [f'm_long {b} = {base};']
        shifts = [24, 16, 8, 0] if size == 'l' else [8, 0]
        for i, sh in enumerate(shifts):
            stmts.append(f'memory().writeByte({b} + {i * 2}, '
                         f'BYTE((cpu().d[{dn}] >> {sh}) & 0xFFu));')
        return stmts
    else:                                    # mem → reg
        dn = dst.reg
        setup, base = ea.address_of(src, tmp)
        b = tmp.fresh()
        stmts = setup + [f'm_long {b} = {base};']
        if size == 'l':
            stmts.append(
                f'cpu().d[{dn}] = '
                f'(LONG(memory().readByte({b})) << 24) | '
                f'(LONG(memory().readByte({b} + 2)) << 16) | '
                f'(LONG(memory().readByte({b} + 4)) << 8) | '
                f'LONG(memory().readByte({b} + 6));'
            )
        else:
            stmts.append(
                f'cpu().d[{dn}] = (cpu().d[{dn}] & 0xFFFF0000u) | '
                f'(LONG(memory().readByte({b})) << 8) | '
                f'LONG(memory().readByte({b} + 2));'
            )
        return stmts


def _nop(instr, tmp, live=None):
    return ['(void)0;']


_HANDLERS = {
    'move': _move, 'movea': _move,
    'moveq': _moveq,
    'movep': _movep,
    'add': _add, 'adda': _add, 'addi': _add, 'addq': _add,
    'sub': _sub, 'suba': _sub, 'subi': _sub, 'subq': _sub,
    'cmp': _cmp, 'cmpa': _cmp, 'cmpi': _cmp, 'cmpm': _cmp,
    'and': _and, 'andi': _and,
    'or': _or, 'ori': _or,
    'eor': _eor, 'eori': _eor,
    'tst': _tst, 'clr': _clr, 'lea': _lea, 'pea': _pea,
    'swap': _swap, 'ext': _ext,
    'neg': _neg, 'not': _not,
    'btst': lambda i, t, live=None: _bitop(i, t, 'BTST', live),
    'bset': lambda i, t, live=None: _bitop(i, t, 'BSET', live),
    'bclr': lambda i, t, live=None: _bitop(i, t, 'BCLR', live),
    'bchg': lambda i, t, live=None: _bitop(i, t, 'BCHG', live),
    'lsl': lambda i, t, live=None: _shift(i, t, 'LSL', live),
    'lsr': lambda i, t, live=None: _shift(i, t, 'LSR', live),
    'asl': lambda i, t, live=None: _shift(i, t, 'ASL', live),
    'asr': lambda i, t, live=None: _shift(i, t, 'ASR', live),
    'rol': lambda i, t, live=None: _shift(i, t, 'ROL', live),
    'ror': lambda i, t, live=None: _shift(i, t, 'ROR', live),
    'roxl': lambda i, t, live=None: _shift(i, t, 'ROXL', live),
    'roxr': lambda i, t, live=None: _shift(i, t, 'ROXR', live),
    'mulu': lambda i, t, live=None: _muldiv(i, t, 'MULU', live),
    'muls': lambda i, t, live=None: _muldiv(i, t, 'MULS', live),
    'divu': lambda i, t, live=None: _muldiv(i, t, 'DIVU', live),
    'divs': lambda i, t, live=None: _muldiv(i, t, 'DIVS', live),
    'exg': _exg,
    'abcd': lambda i, t, live=None: _bcd(i, t, 'ABCD', live),
    'sbcd': lambda i, t, live=None: _bcd(i, t, 'SBCD', live),
    'nbcd': _nbcd,
    'negx': _negx,
    'nop': _nop,
}
