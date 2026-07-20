"""Emit a RecompilationEnvironment subclass from the decoded ROM.

Ties together region partitioning (``regions``), data-op codegen (``opcodes``)
and EA codegen (``ea_codegen``), adding the control-flow lowering that needs
region context: intra-function ``goto``, cross-function calls, the indirect
``dispatch`` table, and the JSR/RTS emulated-stack handling.

Output: a ``Sor.hpp`` / ``Sor.cpp`` pair. Generation is total — an opcode that
cannot be translated is a hard error, never a silent stub; coverage statistics
are reported by ``main``. Function names and intra-function ``goto`` labels
come from the labels CSV when present (falling back to ``sub_XXXXXX`` /
``L{addr}``).
"""

import bisect
import re
from collections import deque

from tools.disassembler.instruction import FlowType, EAMode
from tools.recompiler import cpp_semantics as sem
from tools.recompiler import ea_codegen as ea
from tools.recompiler import opcodes
from tools.recompiler import ccr_liveness
from tools.recompiler.opcodes import Unsupported
from tools.recompiler.ea_codegen import EAGenError, TempPool
from tools.recompiler.regions import partition

# Condition-code numbers.
_CC = {
    't': 0, 'f': 1, 'hi': 2, 'ls': 3, 'cc': 4, 'cs': 5, 'ne': 6, 'eq': 7,
    'vc': 8, 'vs': 9, 'pl': 10, 'mi': 11, 'ge': 12, 'lt': 13, 'gt': 14, 'le': 15,
}


def _fn(addr: int) -> str:
    return f'sub_{addr:06x}'


def _default_label(addr: int) -> str:
    return f'L{addr:06x}'


# Identifiers a generated function must not take (Sor / RecompilationEnvironment
# members and a few obvious C++ keywords); such a label falls back to sub_….
_RESERVED = {
    'run', 'dispatch', 'unhandledDispatch', 'boot', 'powerOn', 'powerOff',
    'onPowerOn', 'cpuInterruptMask', 'dumpUnhandledDispatchCpuState',
    'vSync', 'hSync', 'cpu', 'memory', 'vdp', 'controllers', 'z80', 'sound',
    'loadROM', 'runVDPInterrupts', 'shouldQuit', 'main',
    'int', 'char', 'void', 'for', 'do', 'if', 'else', 'while', 'switch',
    'case', 'default', 'return', 'class', 'struct', 'new', 'delete', 'this',
    'and', 'or', 'not', 'xor', 'auto', 'const', 'static', 'goto',
}


def _sanitize(name: str) -> str | None:
    """Turn a CSV label into a valid C++ identifier, or None if unusable."""
    ident = re.sub(r'[^0-9A-Za-z_]', '_', name.strip())
    if not ident or ident[0].isdigit():
        return None
    return ident


def _speculative_owner_overrides(instructions, speculative_addrs,
                                 speculative_scope):
    """Assign Phase-2 instructions to the speculative flow that reached them.

    Numeric partitioning alone is insufficient for overlapping 68000 decodes.
    For example, a speculative instruction at $014EA0 falls through to $014EA4
    while a baseline four-byte instruction starts at $014EA2.  Addresses after
    $014EA2 still belong to the $014EA0 speculative flow even though their
    nearest preceding subroutine entry is the baseline one.

    Walk each speculative function entry concurrently so roots claim themselves
    before their successors. Calls keep their fall-through in the caller while
    call targets are owned by their own disassembler-created subroutine root.
    """
    speculative_addrs = set(speculative_addrs or [])
    roots = sorted(e for e in set(speculative_scope or []) if e in instructions)
    if not speculative_addrs or not roots:
        return {}

    root_set = set(roots)
    owners = {}
    pending = deque((entry, entry) for entry in roots)
    while pending:
        addr, owner = pending.popleft()
        if addr not in speculative_addrs or addr in owners:
            continue
        if addr in root_set and addr != owner:
            continue
        owners[addr] = owner
        instr = instructions[addr]
        if instr.flow is FlowType.RETURN:
            successors = []
        elif instr.flow is FlowType.BRANCH:
            successors = list(instr.targets)
        elif instr.flow is FlowType.CALL:
            successors = [instr.next_address]
        elif instr.flow is FlowType.CONDITIONAL:
            successors = list(instr.targets) + [instr.next_address]
        else:
            successors = [instr.next_address]
        for successor in successors:
            if successor not in root_set or successor == owner:
                pending.append((successor, owner))

    # All Phase-2 instructions should be reachable from a Phase-2 subroutine
    # root. Keep generation total if a malformed graph says otherwise by using
    # the nearest speculative root rather than falling back to a baseline owner.
    for addr in speculative_addrs - owners.keys():
        i = bisect.bisect_right(roots, addr) - 1
        owners[addr] = roots[i] if i >= 0 else roots[0]
    return owners


def _tail_owner_overrides(instructions, subroutines, owner_overrides,
                          excluded_entries):
    """Keep 68000 tail branches inside one native function.

    A label may also be a callable entry, but a ``bra`` to it is still a jump,
    not a C++ call. Group entries joined by branches so back-edges remain gotos.
    """
    initial = partition(instructions, subroutines, owner_overrides)
    parent = {entry: entry for entry in initial.entries}

    def find(entry):
        while parent[entry] != entry:
            parent[entry] = parent[parent[entry]]
            entry = parent[entry]
        return entry

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    excluded_entries = set(excluded_entries or ())
    for addr, instr in instructions.items():
        if instr.flow not in (FlowType.BRANCH, FlowType.CONDITIONAL):
            continue
        source = initial.func_of(addr)
        for target in instr.targets:
            if target not in instructions:
                continue
            destination = initial.func_of(target)
            if source != destination and not ({source, destination} & excluded_entries):
                union(source, destination)

    aliases = {entry for entry in initial.entries if find(entry) != entry}
    overrides = dict(owner_overrides)
    for entry in aliases:
        owner = find(entry)
        overrides.update({addr: owner for addr in initial.functions[entry].addrs})
    return set(initial.entries) - aliases, aliases, overrides


class Stats:
    def __init__(self):
        self.handled = 0
        self.stubbed = 0
        self.stub_mnemonics = {}

    def stub(self, mnem):
        self.stubbed += 1
        self.stub_mnemonics[mnem] = self.stub_mnemonics.get(mnem, 0) + 1


class Generator:
    def __init__(self, instructions, subroutines, rom_path='rom/StreetsOfRage.bin',
                 names=None, speculative_addrs=None, speculative_scope=None,
                 baseline_instrs=None, manual_functions=None,
                 confirm_addrs=None, rom=None, log_labels=None):
        self.ins = instructions
        self.rom_path = rom_path
        # Optional ROM image: absolute/PC-relative cartridge peeks are folded
        # into C++ literals when the address is a compile-time constant.
        self.rom = rom
        self.stats = Stats()
        self._names = names or {}
        # labels.csv only: subroutine entries that should log on entry.
        self._log_labels = dict(log_labels or {})
        # Phase-2 addresses determine overlapping-flow ownership. Confirmable
        # addresses are broader in discovery builds: they also include baseline
        # instruction boundaries which are not already function entries, since
        # an indirect call may land in the middle of known code.
        phase2_addrs = set(speculative_addrs or [])
        self._speculative = set(
            phase2_addrs if confirm_addrs is None else confirm_addrs)
        # _speculative_scope: all Phase-2-derived functions (seeds + derivatives)
        # — these get their full address list instead of the baseline-filtered one.
        self._speculative_scope = set(
            phase2_addrs if speculative_scope is None else speculative_scope)
        self._manual_functions = set(manual_functions or [])
        speculative_owners = _speculative_owner_overrides(
            instructions, phase2_addrs, self._speculative_scope)
        body_entries, self._entry_aliases, owners = _tail_owner_overrides(
            instructions, subroutines, speculative_owners,
            self._manual_functions | self._speculative_scope)
        self._all_entries = set(subroutines)
        self.part = partition(instructions, body_entries,
                              owner_overrides=owners)
        # Functions implemented by hand retain generated declarations, calls,
        # and dispatch entries, but their C++ bodies are omitted.
        # Effective instruction addresses per function.
        # Baseline functions are restricted to Phase-1 addresses so that phantom
        # instructions injected by overlapping speculative decodes are excluded.
        bl = set(baseline_instrs) if baseline_instrs is not None else None
        self._eff_addrs = {
            e: (f.addrs if (bl is None or e in self._speculative_scope)
                else [a for a in f.addrs if a in bl])
            for e, f in self.part.functions.items()
        }
        # Set version of _eff_addrs for O(1) membership in _transfer.
        self._addrs_sets = {e: set(addrs) for e, addrs in self._eff_addrs.items()}
        # Keep only addresses that have an emitted owner/body.  In particular,
        # this excludes phantom instructions introduced when a speculative
        # decode overlaps a baseline function and baseline filtering removes
        # them from that function's effective address list.
        self._speculative = {
            addr for addr in self._speculative
            if addr in self.ins
            and self.part.func_of(addr) in self._addrs_sets
            and addr in self._addrs_sets[self.part.func_of(addr)]
        }
        # Mid-function speculative addresses use the existing entry_ switch and
        # local-label mechanism.  A small wrapper per address calls the owning
        # grouped body, preserving intra-routine gotos and 68000 loops.
        for addr in self._speculative | self._entry_aliases:
            owner = self.part.func_of(addr)
            if addr != owner:
                self.part.functions[owner].extra_entries.add(addr)
        # Rejected entries (speculative-scope with invalid opcodes): populated in
        # emit_source before the second pass; _transfer skips direct calls to
        # rejected functions.
        self._rejected: set = set()
        self._build_fn_names(self._names)
        self._build_speculative_fn_names()
        self._build_label_names(self._names)

    def _build_fn_names(self, names):
        """Map each function entry to a C++ identifier — the labels-CSV name
        when it sanitizes to a free identifier, otherwise ``sub_XXXXXX``."""
        self._fnname = {}
        used = set()
        for e in sorted(self._all_entries):
            ident = _sanitize(names.get(e, '')) if names.get(e) else None
            if not ident or ident in _RESERVED or ident in used:
                ident = _fn(e)
            if ident in used:                  # extremely unlikely collision
                ident = f'{ident}_{e:06x}'
            used.add(ident)
            self._fnname[e] = ident

    def _build_label_names(self, names):
        """Map goto / mid-entry targets to C++ label identifiers."""
        self._labelname = {}
        used = set(self._fnname.values()) | set(self._spec_fnname.values()) | _RESERVED
        label_addrs = set(self.part.goto_labels)
        for func in self.part.functions.values():
            label_addrs |= func.extra_entries
        for addr in sorted(label_addrs):
            ident = _sanitize(names.get(addr, '')) if names.get(addr) else None
            if not ident or ident in used:
                ident = _default_label(addr)
            if ident in used:
                ident = f'{ident}_{addr:06x}'
            used.add(ident)
            self._labelname[addr] = ident

    def _build_speculative_fn_names(self):
        """Name the lightweight functions for speculative mid-entries.

        Base entries already have a normal generated function.  Every other
        speculative instruction gets a ``sub_AAAAAA`` wrapper unless that name
        was claimed by a labelled base function, in which case use an explicit
        ``spec_entry_AAAAAA`` fallback.
        """
        self._spec_fnname = {}
        used = set(self._fnname.values()) | _RESERVED
        for addr in sorted(self._speculative):
            if addr in self.part.functions:
                continue
            ident = _fn(addr)
            if ident in used:
                ident = f'spec_entry_{addr:06x}'
            used.add(ident)
            self._spec_fnname[addr] = ident

    def fn(self, entry):
        """C++ function name for a function entry address."""
        return self._fnname.get(entry, _fn(entry))

    def label(self, addr):
        """C++ label for an intra-function branch target."""
        return self._labelname.get(addr, _default_label(addr))

    def speculative_fn(self, addr):
        """C++ callable entry for a speculative instruction address."""
        if addr in self.part.functions:
            return self.fn(addr)
        return self._spec_fnname[addr]

    # -- control-flow lowering ------------------------------------------------

    def _transfer(self, src_addr, tgt):
        """Unconditional transfer to absolute ``tgt`` (goto or cross-fn call)."""
        if tgt not in self.ins:
            return [f'traceEnter({ea._hex(src_addr)});',
                    f'dispatch({ea._hex(tgt)}); return;']
        src_fn = self.part.func_of(src_addr)
        tgt_fn = self.part.func_of(tgt)
        if tgt_fn == src_fn and tgt in self._addrs_sets[src_fn]:
            # tgt is actually decoded as part of this function — safe goto.
            return [f'goto {self.label(tgt)};']
        # If the owning function was rejected (invalid speculative code), fall
        # through to dispatch so the runtime can handle it at run time.
        if tgt_fn in self._rejected:
            return [f'traceEnter({ea._hex(src_addr)});',
                    f'dispatch({ea._hex(tgt)}); return;']
        owner = tgt_fn
        return [f'{self.fn(owner)}({ea._hex(tgt)}); return;']

    def _cc_of(self, mnem, prefix):
        return _CC[mnem[len(prefix):]]

    def _emit_flow(self, instr):
        m = instr.mnemonic
        a = instr.address
        nxt = instr.next_address

        if m == 'nop':
            return ['(void)0;']

        if m == 'bra':
            return self._transfer(a, instr.targets[0])

        if m.startswith('b') and m[1:] in _CC and m not in ('bra',):
            cc = _CC[m[1:]]
            body = self._transfer(a, instr.targets[0])
            return [f'if ({sem.cc_expr(cc)}) {{'] + \
                   [f'    {s}' for s in body] + ['}']

        if m in ('jmp',):
            if instr.indirect or not instr.targets:
                setup, addr = self._jump_address(instr)
                owner = self.part.func_of(a)
                local_targets = sorted(
                    target for target in self.part.functions[owner].extra_entries
                    if target in self._addrs_sets[owner])
                if local_targets:
                    target = f'jump_target_{a:06x}'
                    setup += [f'm_long {target} = {addr};', f'switch ({target}) {{']
                    setup += [f'    case {ea._hex(entry)}: goto {self.label(entry)};'
                              for entry in local_targets]
                    setup += ['    default: break;', '}']
                    addr = target
                return setup + [f'traceEnter({ea._hex(a)});',
                                f'dispatch({addr}); return;']
            return self._transfer(a, instr.targets[0])

        if m in ('bsr', 'jsr'):
            ret = ea._hex(nxt)
            if m == 'jsr' and (instr.indirect or not instr.targets):
                setup, addr = self._jump_address(instr)
                # Indirect: evaluate EA then CALL_DISPATCH.
                return setup + [
                    f'traceEnter({ea._hex(a)});',
                    f'CALL_DISPATCH({addr}, {ret});',
                ]
            tgt = instr.targets[0]
            if tgt in self.ins:
                owner = self.part.func_of(tgt)
                if owner not in self._rejected:
                    if owner == tgt:
                        return [f'CALL({self.fn(owner)}, {ret});']
                    return [f'CALL_ENTRY({self.fn(owner)}, {ea._hex(tgt)}, {ret});']
            return [
                f'traceEnter({ea._hex(a)});',
                f'CALL_DISPATCH({ea._hex(tgt)}, {ret});',
            ]

        if m == 'rts':
            return ['RETURN_68K();']

        if m == 'rte':
            return ['cpu().setStatus(memory().readWord(cpu().ssp));',
                    'cpu().ssp += 6;',
                    'return;']

        if m == 'rtr':
            return ['cpu().setCCR(memory().readWord(cpu().ssp));',
                    'cpu().ssp += 6;',
                    'return;']

        if m.startswith('db'):
            suffix = m[2:]
            cc = 1 if suffix in ('f', 'ra') else _CC[suffix]
            reg = instr.eas[0].reg
            body = self._transfer(a, instr.targets[0])
            ctr = f'dbcc_{a:06x}'
            # DBcc: if condition false, decrement and maybe branch.
            return [f'if (!{sem.cc_expr(cc)}) {{',
                    f'    m_word {ctr} = WORD((cpu().dw({reg}) - 1) & 0xFFFFu);',
                    f'    cpu().setDw({reg}, {ctr});',
                    f'    if ({ctr} != 0xFFFFu) {{'] + \
                   [f'        {s}' for s in body] + ['    }', '}']

        raise Unsupported(m)

    def _jump_address(self, instr):
        """(setup, expr) for the effective target address of an indirect jmp/jsr."""
        if not instr.eas:
            return [], 'cpu().pc'
        e = instr.eas[0]
        if e.mode == EAMode.ADDR_IND:
            return [], ea.areg(e.reg)
        try:
            return ea.address_of(e, TempPool(instr.address))
        except EAGenError:
            return [], ea._hex(e.abs_value or 0)

    # -- per-instruction emission --------------------------------------------

    def _emit_instr(self, instr, live_out=None):
        try:
            if instr.mnemonic in opcodes.FLOW_MNEMONICS:
                body = self._emit_flow(instr)
            elif instr.mnemonic == 'movem':
                body = self._emit_movem(instr)
            else:
                body = opcodes.emit_dataop(instr, live_flags=live_out)
                if body is None:
                    raise Unsupported(instr.mnemonic)
        except (Unsupported, EAGenError) as exc:
            raise RuntimeError(
                f'cannot translate {instr} at ${instr.address:06X}: '
                f'{type(exc).__name__}: {exc}') from exc
        self.stats.handled += 1
        lines = []
        if self.part.needs_label(instr.address):
            lines.append(f'{self.label(instr.address)}:')
        lines.append(f'// ${instr.address:06X} {instr}')
        # One block per instruction so short temps (t0/t1) can be reused safely.
        # Always braced for a uniform listing — the C++ compiler optimizes this.
        lines.append('{')
        lines.append('    BEFORE_INSTRUCTION')
        for stmt in body:
            lines.append(f'    {stmt}')
        lines.append('}')
        return lines

    # -- movem (register-list memory block transfer) --------------------------

    @staticmethod
    def _reg_index(tok: str) -> int:
        """Map ``dN`` / ``aN`` / ``sp`` to a unified 0..15 index (d0..d7, a0..a7)."""
        tok = tok.strip()
        if tok == 'sp':
            return 15
        if tok[0] == 'd':
            return int(tok[1])
        return 8 + int(tok[1])

    @staticmethod
    def _index_to_reg(i: int) -> tuple[bool, int]:
        if i < 8:
            return (False, i)
        n = i - 8
        return (True, 7 if n == 7 else n)

    @staticmethod
    def _parse_reglist(text):
        """'d0-d7/a0-a5' or 'd5-a4' → [(is_addr, n), …] in canonical order."""
        regs = []
        for group in text.split('/'):
            if '-' in group:
                lo, hi = group.split('-', 1)
                i0 = Generator._reg_index(lo)
                i1 = Generator._reg_index(hi)
                if i0 > i1:
                    i0, i1 = i1, i0
                for i in range(i0, i1 + 1):
                    regs.append(Generator._index_to_reg(i))
            else:
                regs.append(Generator._index_to_reg(Generator._reg_index(group)))
        return sorted(set(regs), key=lambda r: (r[0], r[1]))

    @staticmethod
    def _movem_regs(reg_ea):
        """Register list for a movem operand (REG_LIST, or a single Dn / An)."""
        if reg_ea.mode == EAMode.REG_LIST:
            return Generator._parse_reglist(reg_ea.reglist)
        if reg_ea.mode == EAMode.DATA_REG:
            return [(False, reg_ea.reg)]
        if reg_ea.mode == EAMode.ADDR_REG:
            return [(True, reg_ea.reg)]
        raise ValueError(f'movem register operand has mode {reg_ea.mode}')

    def _movem_reg_read(self, is_addr, n, size):
        if is_addr:
            ar = ea.areg(n)
            if size == 'w':
                return f'WORD({ar} & 0xFFFFu)'
            return ar
        return ea.read_dn(n, size)

    def _movem_reg_write(self, is_addr, n, size, value, types=None):
        if is_addr:
            ar = ea.areg(n)
            if size == 'l':
                return ea.write_areg_long(ar, value, types)
            return ea.write_areg_word(ar, value, types)
        if size == 'l':
            return ea.write_dn(n, 'l', value, types)
        # MOVEM memory→register sign-extends each word to 32 bits — Dn too.
        return ea.write_dn(n, 'l', ea.signext_to_long(value, 'w', types), types)

    def _emit_movem(self, instr):
        size = instr.size or 'w'
        loads = {'b': 'memory().readByte', 'w': 'memory().readWord',
                 'l': 'memory().readLong'}[size]
        storem = {'b': 'memory().writeByte', 'w': 'memory().writeWord',
                  'l': 'memory().writeLong'}[size]
        nbytes = 4 if size == 'l' else 2
        tmp = TempPool(instr.address)
        reg_size = 'l' if size == 'l' else 'w'

        # The memory side is the operand with a memory addressing mode; the
        # other operand is the register list (possibly a single register, which
        # the decoder classifies as DATA_REG/ADDR_REG rather than REG_LIST).
        mem_modes = {EAMode.ADDR_IND, EAMode.ADDR_POSTINC, EAMode.ADDR_PREDEC,
                     EAMode.ADDR_DISP, EAMode.ADDR_INDEX, EAMode.ABS_W,
                     EAMode.ABS_L, EAMode.PC_DISP, EAMode.PC_INDEX}
        if instr.eas[0].mode in mem_modes:
            mem, reg_ea, store = instr.eas[0], instr.eas[1], False
        else:
            mem, reg_ea, store = instr.eas[1], instr.eas[0], True
        regs = self._movem_regs(reg_ea)

        out = []
        if store and mem.mode == EAMode.ADDR_PREDEC:
            ar = ea.areg(mem.reg)
            init = None
            if (True, mem.reg) in regs:
                # 68000: when the base An is in the list, its *initial* value
                # is stored, not the partially-decremented one.
                init = tmp.fresh('l')
                out.append(f'm_long {init} = {ar};')
            for is_addr, n in reversed(regs):       # predec stores high→low
                out.append(f'{ar} -= {nbytes};')
                if init is not None and is_addr and n == mem.reg:
                    val = f'WORD({init} & 0xFFFFu)' if size == 'w' else init
                else:
                    val = self._movem_reg_read(is_addr, n, size)
                out.append(f'{storem}({ar}, {val});')
            return out
        if not store and mem.mode == EAMode.ADDR_POSTINC:
            ar = ea.areg(mem.reg)
            for is_addr, n in regs:
                v = tmp.fresh(reg_size)
                out.append(f'm_{"long" if size == "l" else "word"} {v} '
                           f'= {loads}({ar});')
                out.append(f'{ar} += {nbytes};')
                out.append(self._movem_reg_write(is_addr, n, size, v, tmp.types))
            return out

        # Control addressing modes: sequential offsets from a fixed base.
        setup, addr = ea.address_of(mem, tmp)
        base = tmp.fresh('l')
        out += setup + [f'm_long {base} = {addr};']
        for off, (is_addr, n) in enumerate(regs):
            ea_expr = f'{base} + {off * nbytes}'
            if store:
                out.append(f'{storem}({ea_expr}, {self._movem_reg_read(is_addr, n, size)});')
            else:
                v = tmp.fresh(reg_size)
                out.append(f'm_{"long" if size == "l" else "word"} {v} '
                           f'= {loads}({ea_expr});')
                out.append(self._movem_reg_write(is_addr, n, size, v, tmp.types))
        return out

    # -- function emission ----------------------------------------------------

    def _emit_function(self, func):
        out = [f'void Sor::{self.fn(func.entry)}(m_long entry_) {{']
        out.append('    traceEnter(entry_);')
        return self._emit_function_body(func, out)

    def _emit_function_body(self, func, out):
        addrs = self._eff_addrs[func.entry]
        eff_set = self._addrs_sets[func.entry]
        # Only emit extra-entry goto cases for addresses that will actually be
        # emitted — phantom entries from speculative contamination are excluded.
        eff_extras = [t for t in sorted(func.extra_entries) if t in eff_set]
        if eff_extras:
            out.append('    switch (entry_) {')
            for t in eff_extras:
                out.append(f'        case {ea._hex(t)}: goto {self.label(t)};')
            out.append('        default: break;')
            out.append('    }')
        else:
            out.append('    (void)entry_;')
        # Omit CCR updates for flags that no later reader observes.
        live_out = ccr_liveness.analyze(
            addrs, self.ins, self.part.func_of, func.entry)
        falls_through = (FlowType.SEQUENTIAL, FlowType.CONDITIONAL, FlowType.CALL)
        for index, addr in enumerate(addrs):
            live = live_out.get(addr, ccr_liveness.ALL)
            out += [f'    {ln}' for ln in self._emit_instr(self.ins[addr], live)]
            instr = self.ins[addr]
            next_emitted = addrs[index + 1] if index + 1 < len(addrs) else None
            if (next_emitted is not None and instr.flow in falls_through
                    and instr.next_address != next_emitted):
                out += [f'    {ln}' for ln in self._transfer(
                    addr, instr.next_address)]
        # A function whose last instruction falls through (no RTS/RTE/BRA/JMP)
        # is hand-optimized 68000 code sharing a tail with whatever comes next
        # in ROM order — real hardware just keeps executing into it. Tail-call
        # the owning function instead of returning, so the original caller's
        # pushed return address gets popped by *that* function's eventual
        # rts/rte rather than leaking 4 bytes off the emulated 68k stack.
        last = self.ins[addrs[-1]]
        if last.flow in falls_through \
                and last.next_address in self.ins:
            out += [f'    {ln}' for ln in self._transfer(addrs[-1], last.next_address)]
        else:
            out.append('    return;')
        out.append('}')
        return out

    def _emit_speculative_wrapper(self, addr):
        """Emit one exact-address entry function without duplicating its body."""
        owner = self.part.func_of(addr)
        return '\n'.join([
            f'void Sor::{self.speculative_fn(addr)}() {{',
            f'    {self.fn(owner)}({ea._hex(addr)});',
            '}',
        ])

    def _emit_entry_alias_wrapper(self, addr):
        """Preserve names used by hand-written code for grouped entries."""
        owner = self.part.func_of(addr)
        return '\n'.join([
            f'void Sor::{self.fn(addr)}(m_long entry_) {{',
            f'    {self.fn(owner)}(entry_);',
            '}',
        ])


    # -- whole-program emission ----------------------------------------------

    def emit_header(self):
        decls = [f'    void {self.fn(e)}(m_long entry_ = {ea._hex(e)});'
                 for e in self.part.entries if e not in self._rejected]
        decls += [f'    void {self.fn(e)}(m_long entry_ = {ea._hex(e)});'
                  for e in sorted(self._entry_aliases)
                  if self.part.func_of(e) not in self._rejected]
        decls += [f'    void {self.speculative_fn(addr)}();'
                  for addr in sorted(self._speculative)
                  if addr not in self.part.functions
                  and self.part.func_of(addr) not in self._rejected]
        return _HEADER_TEMPLATE.format(decls='\n'.join(decls))

    def emit_source(self):
        ea.set_active_rom(self.rom)
        try:
            return self._emit_source_body()
        finally:
            ea.set_active_rom(None)

    def _emit_labelled_entry_log(self):
        """Switch for incomplete labels.csv subroutine entries → async Logger.

        Only addresses present in labels.csv whose description does **not**
        contain ``100%`` (still under investigation), and that became real
        function entries (not anonymous sub_XXXXXX fallbacks), are logged.
        Fully known (100%) routines and addresses.csv-only names stay silent.
        """
        named = []
        for e in sorted(self._all_entries):
            if e not in self._log_labels:
                continue
            name = self._fnname.get(e, _fn(e))
            if name == _fn(e):
                continue  # label unusable / reserved — silent sub_XXXXXX
            named.append((e, name))
        lines = [
            'void labelledEntryLog(m_long addr) {',
            '    switch (addr & 0x00FFFFFFu) {',
        ]
        for e, name in named:
            lines.append(
                f'        case {ea._hex(e)}: '
                f'Logger::log("[68k] {name} ($%06X)", '
                f'static_cast<unsigned>(addr & 0x00FFFFFFu)); return;'
            )
        lines += [
            '        default: return;',
            '    }',
            '}',
        ]
        return '\n'.join(lines)

    def _emit_source_body(self):
        boot = self.fn(self.part.func_of(0x000200))
        parts = [_SOURCE_PREAMBLE.format(cast_macros=sem.CAST_MACROS.strip(),
                                         rom_path=self.rom_path,
                                         boot_fn=boot,
                                         labelled_entry_log=self._emit_labelled_entry_log())]

        # Pre-translate all functions; speculative-scope entries that fail are
        # excluded entirely — they decoded data as code and are not valid entry
        # points.
        # Non-speculative failures are real recompiler bugs and re-raise.
        bodies = {}
        rejected = set()
        for e in self.part.entries:
            if e in self._manual_functions:
                continue
            try:
                bodies[e] = '\n'.join(self._emit_function(self.part.functions[e]))
            except Exception:
                if e in self._speculative_scope:
                    rejected.add(e)
                else:
                    raise

        if rejected:
            import sys
            print(f'[recompile] {len(rejected)} speculative entry(ies) rejected '
                  f'(invalid opcodes — treated as data, not code)', file=sys.stderr)

        # Expose rejected set so _transfer can route calls through dispatch instead
        # of generating direct C++ function calls to non-existent bodies.
        self._rejected = rejected
        # Re-translate: _transfer now knows which targets are rejected and emits
        # dispatch() for them so their callers still link correctly.  Reset
        # counters so the probe pass above does not double-count.
        self.stats = Stats()
        bodies = {e: '\n'.join(self._emit_function(self.part.functions[e]))
                  for e in self.part.entries
                  if e not in rejected and e not in self._manual_functions}

        disp = ['void Sor::dispatch(m_long addr) {', '    switch (addr) {']
        dispatch_entries = sorted(self._all_entries | self._speculative)
        for e in dispatch_entries:
            owner = self.part.func_of(e)
            if owner not in rejected:
                if e in self._speculative:
                    disp.append(
                        f'        case {ea._hex(e)}: '
                        f'confirmSpeculative({ea._hex(e)}); '
                        f'{self.speculative_fn(e)}(); return;')
                else:
                    entry = '' if e == owner else ea._hex(e)
                    disp.append(f'        case {ea._hex(e)}: '
                                f'{self.fn(owner)}({entry}); return;')
        disp += ['        default: unhandledDispatch(addr); return;', '    }', '}']
        parts.append('\n'.join(disp))

        for e in self.part.entries:
            if e not in rejected and e not in self._manual_functions:
                parts.append(bodies[e])

        for addr in sorted(self._entry_aliases):
            if self.part.func_of(addr) not in rejected:
                parts.append(self._emit_entry_alias_wrapper(addr))

        for addr in sorted(self._speculative):
            owner = self.part.func_of(addr)
            if addr not in self.part.functions and owner not in rejected:
                parts.append(self._emit_speculative_wrapper(addr))

        return '\n\n'.join(parts) + '\n'


_HEADER_TEMPLATE = '''\
#pragma once

// Generated by tools/recompiler — do not edit by hand.
// Recompiled Streets of Rage cartridge as a RecompilationEnvironment subclass.

#include "RecompilationEnvironment.hpp"
#include <cstdint>
#include <string>

class Sor : public RecompilationEnvironment {{
    public:
    explicit Sor(const std::string &romPath,
                 VDP::Synchronization sync    = VDP::VSync,
                 VDP::Scaling         scaling = VDP::Integer,
                 std::uint16_t        remoteAccessPort = 6969);

    protected:
    void run() override;

    private:
    // Indirect-jump dispatch (jmp (an), computed jumps).
    void dispatch(m_long addr);
    void unhandledDispatch(m_long addr);

    // Cooperative interrupt delivery: 68000 exception entry + handler dispatch,
    // invoked before an instruction when an unmasked IRQ is pending.
    void serviceIRQ();

    // Recompiled subroutines plus exact speculative instruction entries.
{decls}
}};
'''

_SOURCE_PREAMBLE = '''\
// Generated by tools/recompiler — do not edit by hand.

#include "Sor.hpp"
#include "Logger.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

{cast_macros}

{labelled_entry_log}

Sor::Sor(const std::string &romPath, VDP::Synchronization sync,
         VDP::Scaling scaling, std::uint16_t remoteAccessPort)
    : RecompilationEnvironment(sync, scaling, VDP::HardwareSpriteLimit, remoteAccessPort) {{
    loadROM(romPath);
}}

void Sor::unhandledDispatch(m_long addr) {{
    // Delegated to the base class so the behaviour (abort, or record-and-exit
    // when --auxAddrFile is set) lives in hand-written, non-generated code.
    reportUnhandledDispatch(addr);
}}

void Sor::serviceIRQ() {{
    // Slow path: an unmasked interrupt is pending. Perform the 68000 autovector
    // exception entry, then dispatch the recompiled handler; its `rte` restores
    // the saved SR (and the previous IPL) and balances the stack.
    int level = irqLevel();
    if (level <= cpu().interruptMask())
        return; // raced with another service / masked
    clearInterrupt(level);
    m_word oldSR = cpu().status();
    cpu().ssp -= 4;
    memory().writeLong(cpu().ssp, 0); // return PC slot (native return drives flow)
    cpu().ssp -= 2;
    memory().writeWord(cpu().ssp, oldSR);
    // Stay in supervisor mode and raise the interrupt mask to this level.
    cpu().enterInterrupt(level);
    dispatch(memory().readLong(0x60 + LONG(level) * 4));
    if (level == 6) {{
        sound().endFrame();
    }}
}}

void Sor::run() {{
    // Reset: load the supervisor stack pointer from the reset vector, then
    // enter the ROM bootstrap. (The boot code itself reloads SR/registers.)
    cpu().ssp = memory().readLong(0x000000);
    {boot_fn}();
}}'''
