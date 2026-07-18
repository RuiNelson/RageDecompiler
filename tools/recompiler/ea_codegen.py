"""Effective-address → C++ expression generation.

Translates one structured :class:`EA` (from the disassembler) into C++ that
reads it, writes it, or computes its address — emitting **temporaries** so that
addressing-mode side effects (``(An)+`` post-increment, ``-(An)`` pre-decrement)
are sequenced correctly even when the same register appears in both operands
(``move.l (a0)+,(a0)+``).

Conventions of the emitted code
-------------------------------
* Data registers use ``cpu().d[n]`` for long and ``cpu().db/dw/setDb/setDw``
  for byte/word (merge-on-write helpers on ``CPU68K``).
* Address registers use ``cpu().a[n]`` / ``cpu().ssp`` (A7); word writes sign-
  extend to 32 bits via ``SEX_W`` / ``SEX_B`` (movea / lea rule).
* Memory is reached through ``memory()`` (``SystemMemory``):
  ``readByte/Word/Long`` and ``writeByte/Word/Long``. ``SystemMemory`` masks
  every address to 24 bits, so the generator never masks.
* Casts (``BYTE``/``WORD``/``LONG``) are emitted only when the expression's
  known type differs from the needed width — typed temps, ``db``/``dw``, and
  bare hex immediates are left alone.
"""

import re
import struct

from tools.disassembler.instruction import EA, EAMode

SIZE_BYTES = {'b': 1, 'w': 2, 'l': 4}
_READ_FN   = {'b': 'readByte',  'w': 'readWord',  'l': 'readLong'}
_WRITE_FN  = {'b': 'writeByte', 'w': 'writeWord', 'l': 'writeLong'}
_CTYPE     = {'b': 'm_byte', 'w': 'm_word', 'l': 'm_long'}
_CAST      = {'b': 'BYTE', 'w': 'WORD', 'l': 'LONG'}

_IMM_RE = re.compile(r'^(?:0x[0-9A-Fa-f]+|\d+)u?$')
_TEMP_RE = re.compile(r'^t\d+$')
_DB_RE = re.compile(r'^cpu\(\)\.db\(\d\)$')
_DW_RE = re.compile(r'^cpu\(\)\.dw\(\d\)$')
_DL_RE = re.compile(r'^cpu\(\)\.(?:d\[\d\]|a\[\d\]|ssp)$')
# Pure hex address expression as emitted by address_of / _hex.
_CONST_ADDR_RE = re.compile(r'^0x([0-9A-Fa-f]+)u?$')

# ROM image for constant-folding absolute cartridge reads (set by Generator).
_active_rom = None


def set_active_rom(rom) -> None:
    """Install the ROM image used to fold constant cartridge reads, or None."""
    global _active_rom
    _active_rom = rom


def parse_const_addr(expr: str) -> int | None:
    """If *expr* is a bare hex address literal, return its 24-bit value."""
    m = _CONST_ADDR_RE.match(expr.strip())
    if not m:
        return None
    return int(m.group(1), 16) & 0xFFFFFF


def fold_rom_read(addr: int, size: str) -> str | None:
    """Return a C++ hex literal for a ROM peek, or None if not foldable.

    Only absolute addresses that fall inside the cartridge image are folded.
    Work RAM / I/O / odd word addresses stay as runtime ``memory()`` reads.
    """
    rom = _active_rom
    if rom is None:
        return None
    addr &= 0xFFFFFF
    n = SIZE_BYTES[size]
    data = getattr(rom, '_data', None)
    if data is None or addr + n > len(data):
        return None
    # Word/long must be even on the 68000.
    if size in ('w', 'l') and (addr & 1):
        return None
    if size == 'b':
        return _hex(data[addr])
    if size == 'w':
        return _hex(struct.unpack_from('>H', data, addr)[0])
    return _hex(struct.unpack_from('>I', data, addr)[0])


class EAGenError(Exception):
    """An EA that this generator cannot translate (e.g. RAW fallback)."""


class TempPool:
    """Hands out unique temporary names within one instruction's emission.

    Names are short (``t0``, ``t1``, …).  Optional size tags feed the cast
    elider so ``setDw(0, t0)`` need not wrap an already-``m_word`` temp.
    """

    def __init__(self, addr: int = 0) -> None:
        self._n = 0
        self.types: dict[str, str] = {}

    def fresh(self, size: str | None = None) -> str:
        name = f't{self._n}'
        self._n += 1
        if size in _CTYPE:
            self.types[name] = size
        return name


def _hex(value: int) -> str:
    """Format an unsigned 32-bit constant as a compact C++ hex literal."""
    v = value & 0xFFFFFFFF
    if v <= 0xFF:
        return f'0x{v:02X}u'
    if v <= 0xFFFF:
        return f'0x{v:04X}u'
    return f'0x{v:08X}u'


def expr_size(value: str, types: dict[str, str] | None = None) -> str | None:
    """Best-effort size tag for *value*, or None if unknown."""
    e = value.strip()
    if e.startswith('BYTE('):
        return 'b'
    if e.startswith('WORD('):
        return 'w'
    if e.startswith('LONG(') or e.startswith('SEX_W(') or e.startswith('SEX_B('):
        return 'l'
    if _DB_RE.match(e):
        return 'b'
    if _DW_RE.match(e):
        return 'w'
    if _DL_RE.match(e):
        return 'l'
    if types and _TEMP_RE.match(e):
        return types.get(e)
    if _IMM_RE.match(e):
        return 'imm'
    return None


def _cast(size: str, value: str, types: dict[str, str] | None = None) -> str:
    """Apply BYTE/WORD/LONG only when the expression is not already that width.

    Immediates and same-sized temps/registers are returned unchanged — the C++
    assignment or ``setDb``/``setDw`` parameter type does the conversion.
    """
    e = value.strip()
    cast = _CAST[size]
    if e.startswith(f'{cast}('):
        return e
    known = expr_size(e, types)
    if known == size or known == 'imm':
        return e
    return f'{cast}({e})'


def areg(n: int) -> str:
    """Lvalue for address register An. A7 is the supervisor stack pointer."""
    return 'cpu().ssp' if n == 7 else f'cpu().a[{n}]'


def addr_step(reg: int, size: str) -> int:
    """Predecrement/postincrement step; byte accesses on A7 move by two."""
    if reg == 7 and size == 'b':
        return 2
    return SIZE_BYTES[size]


def read_dn(n: int, size: str) -> str:
    """Expression reading Dn at byte / word / long width."""
    if size == 'b':
        return f'cpu().db({n})'
    if size == 'w':
        return f'cpu().dw({n})'
    return f'cpu().d[{n}]'


def write_dn(n: int, size: str, value: str,
             types: dict[str, str] | None = None) -> str:
    """Statement writing ``value`` into Dn, preserving untouched high bits."""
    if size == 'b':
        return f'cpu().setDb({n}, {_cast("b", value, types)});'
    if size == 'w':
        return f'cpu().setDw({n}, {_cast("w", value, types)});'
    return f'cpu().d[{n}] = {_cast("l", value, types)};'


def write_areg_word(ar: str, value: str,
                    types: dict[str, str] | None = None) -> str:
    """Word write to An — sign-extend bit 15 (movea / lea)."""
    return f'{ar} = SEX_W({_cast("w", value, types)});'


def write_areg_long(ar: str, value: str,
                    types: dict[str, str] | None = None) -> str:
    return f'{ar} = {_cast("l", value, types)};'


def signext_to_long(expr: str, size: str,
                    types: dict[str, str] | None = None) -> str:
    if size == 'l':
        return _cast('l', expr, types)
    if size == 'w':
        return f'SEX_W({_cast("w", expr, types)})'
    return f'SEX_B({_cast("b", expr, types)})'


def _index_expr(ea: EA) -> str:
    """C++ expression for the (sign-extended) index register of an indexed EA."""
    if ea.index_is_addr:
        reg = areg(ea.index_reg)
        if ea.index_size == 'w':
            return f'SEX_W({reg} & 0xFFFFu)'
        return reg
    if ea.index_size == 'w':
        return f'SEX_W(cpu().dw({ea.index_reg}))'
    return f'cpu().d[{ea.index_reg}]'


def address_of(ea: EA, tmp: TempPool) -> tuple[list[str], str]:
    """Return (setup statements, address expression) for a memory EA."""
    if ea.mode == EAMode.ADDR_IND:
        return [], areg(ea.reg)
    if ea.mode == EAMode.ADDR_DISP:
        if not ea.disp:
            return [], areg(ea.reg)
        return [], f'({areg(ea.reg)} + {ea.disp})'
    if ea.mode == EAMode.ADDR_INDEX:
        idx = _index_expr(ea)
        if not ea.disp:
            return [], f'({areg(ea.reg)} + {idx})'
        return [], f'({areg(ea.reg)} + {ea.disp} + {idx})'
    if ea.mode == EAMode.ABS_W:
        v = ea.abs_value & 0xFFFF
        return [], _hex(v | 0xFFFF0000 if (v & 0x8000) else v)
    if ea.mode == EAMode.ABS_L:
        return [], _hex(ea.abs_value)
    if ea.mode == EAMode.PC_DISP:
        return [], _hex(ea.abs_value)
    if ea.mode == EAMode.PC_INDEX:
        return [], f'({_hex(ea.abs_value)} + {_index_expr(ea)})'
    raise EAGenError(f'address_of: mode {ea.mode} has no address')


def read_ea(ea: EA, size: str, tmp: TempPool) -> tuple[list[str], str]:
    """Return (setup statements, value expression) reading ``ea`` at ``size``."""
    if ea.mode == EAMode.DATA_REG:
        return [], read_dn(ea.reg, size)

    if ea.mode == EAMode.ADDR_REG:
        if size == 'l':
            return [], areg(ea.reg)
        if size == 'w':
            # Low 16 bits of An; keep an explicit mask (not a Dn helper).
            return [], f'WORD({areg(ea.reg)} & 0xFFFFu)'
        return [], f'BYTE({areg(ea.reg)} & 0xFFu)'

    if ea.mode == EAMode.IMMEDIATE:
        # Bare hex — callers cast only when the value must change width.
        return [], _hex(ea.imm)

    if ea.mode == EAMode.ADDR_POSTINC:
        v = tmp.fresh(size)
        step = addr_step(ea.reg, size)
        stmts = [
            f'{_CTYPE[size]} {v} = memory().{_READ_FN[size]}({areg(ea.reg)});',
            f'{areg(ea.reg)} += {step};',
        ]
        return stmts, v

    if ea.mode == EAMode.ADDR_PREDEC:
        v = tmp.fresh(size)
        step = addr_step(ea.reg, size)
        stmts = [
            f'{areg(ea.reg)} -= {step};',
            f'{_CTYPE[size]} {v} = memory().{_READ_FN[size]}({areg(ea.reg)});',
        ]
        return stmts, v

    setup, addr = address_of(ea, tmp)
    # Absolute / PC-relative cart reads with a fixed address → literal.
    const_addr = parse_const_addr(addr)
    if const_addr is not None:
        folded = fold_rom_read(const_addr, size)
        if folded is not None:
            return list(setup), folded

    v = tmp.fresh(size)
    setup = list(setup)
    setup.append(f'{_CTYPE[size]} {v} = memory().{_READ_FN[size]}({addr});')
    return setup, v


def rmw_ea(ea: EA, size: str, tmp: TempPool) -> tuple[list[str], str, list[str]]:
    """Read-modify-write access: (pre, value_temp, post)."""
    types = tmp.types
    if ea.mode == EAMode.DATA_REG:
        v = tmp.fresh(size)
        return ([f'{_CTYPE[size]} {v} = {read_dn(ea.reg, size)};'],
                v, [write_dn(ea.reg, size, v, types)])

    if ea.mode == EAMode.ADDR_REG:
        # No 68000 opcode does a byte/word read-modify-write on An.
        if size != 'l':
            raise EAGenError(f'{size}-size read-modify-write on an address register')
        v = tmp.fresh('l')
        return ([f'm_long {v} = {areg(ea.reg)};'],
                v, [write_areg_long(areg(ea.reg), v, types)])

    bytes_ = (
        addr_step(ea.reg, size)
        if ea.mode in (EAMode.ADDR_POSTINC, EAMode.ADDR_PREDEC)
        else SIZE_BYTES[size]
    )
    addr = tmp.fresh('l')
    v = tmp.fresh(size)
    if ea.mode == EAMode.ADDR_POSTINC:
        pre = [f'm_long {addr} = {areg(ea.reg)};',
               f'{_CTYPE[size]} {v} = memory().{_READ_FN[size]}({addr});']
        post = [f'memory().{_WRITE_FN[size]}({addr}, {v});',
                f'{areg(ea.reg)} += {bytes_};']
        return pre, v, post
    if ea.mode == EAMode.ADDR_PREDEC:
        pre = [f'{areg(ea.reg)} -= {bytes_};',
               f'm_long {addr} = {areg(ea.reg)};',
               f'{_CTYPE[size]} {v} = memory().{_READ_FN[size]}({addr});']
        post = [f'memory().{_WRITE_FN[size]}({addr}, {v});']
        return pre, v, post

    setup, aexpr = address_of(ea, tmp)
    const_addr = parse_const_addr(aexpr)
    folded = (fold_rom_read(const_addr, size)
              if const_addr is not None else None)
    if folded is not None:
        # Read side is a cart constant; write still goes through memory()
        # (unusual for ROM, but keep store semantics exact).
        pre = list(setup) + [f'{_CTYPE[size]} {v} = {folded};']
        post = [f'memory().{_WRITE_FN[size]}({aexpr}, {v});']
        return pre, v, post

    pre = setup + [f'm_long {addr} = {aexpr};',
                   f'{_CTYPE[size]} {v} = memory().{_READ_FN[size]}({addr});']
    post = [f'memory().{_WRITE_FN[size]}({addr}, {v});']
    return pre, v, post


def write_ea(ea: EA, size: str, value: str, tmp: TempPool) -> list[str]:
    """Return statements writing ``value`` into ``ea`` at ``size``."""
    types = tmp.types
    if ea.mode == EAMode.DATA_REG:
        return [write_dn(ea.reg, size, value, types)]

    if ea.mode == EAMode.ADDR_REG:
        # No 68000 opcode writes a byte to An; word writes sign-extend (movea).
        if size == 'b':
            raise EAGenError('byte write to an address register')
        if size == 'l':
            return [write_areg_long(areg(ea.reg), value, types)]
        return [write_areg_word(areg(ea.reg), value, types)]

    if ea.mode == EAMode.ADDR_POSTINC:
        step = addr_step(ea.reg, size)
        return [
            f'memory().{_WRITE_FN[size]}({areg(ea.reg)}, '
            f'{_cast(size, value, types)});',
            f'{areg(ea.reg)} += {step};',
        ]

    if ea.mode == EAMode.ADDR_PREDEC:
        step = addr_step(ea.reg, size)
        return [
            f'{areg(ea.reg)} -= {step};',
            f'memory().{_WRITE_FN[size]}({areg(ea.reg)}, '
            f'{_cast(size, value, types)});',
        ]

    if ea.mode in (EAMode.IMMEDIATE, EAMode.PC_DISP, EAMode.PC_INDEX):
        raise EAGenError(f'{ea.mode} is not a writable destination')

    setup, addr = address_of(ea, tmp)
    return setup + [
        f'memory().{_WRITE_FN[size]}({addr}, {_cast(size, value, types)});'
    ]
