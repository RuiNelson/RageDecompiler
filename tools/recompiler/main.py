"""CLI: recompile the ROM into a Sor.hpp / Sor.cpp RecompilationEnvironment subclass.

    python3 -m tools.recompiler rom/StreetsOfRage.bin -o src/generated [--aux aux_addresses.txt]

Loads the ROM, runs the recursive-descent disassembler (seeded with the
reset/IRQ vectors and any auxiliary entry points), then emits the recompiled
C++. Prints coverage: how many instructions were translated vs. emitted as
compile-safe stubs (unimplemented opcodes), broken down by mnemonic.
"""

import argparse
import bisect
import os
import sys

from tools.disassembler.disassembler import Disassembler
from tools.disassembler.instruction import EAMode
from tools.disassembler.main import _load_csv_addresses, _load_labels_csv
from tools.disassembler.rom import ROM
from tools.common.addresses import parse_address_lines
from tools.recompiler.generator import Generator


def _discover_table_targets(disasm, rom) -> set:
    found = _discover_bra_tables(disasm, rom)
    found |= _discover_word_jump_tables(disasm, rom)
    found |= _discover_shared_dispatcher_tables(disasm, rom)
    return found


def _disassemble_to_fixpoint(rom, seeds, verbose=False):
    seeds = set(seeds)
    while True:
        disasm = Disassembler(rom, aux_addresses=sorted(seeds),
                              verbose=verbose)
        disasm.disassemble()
        discovered = _discover_table_targets(disasm, rom)
        if discovered <= seeds:
            return disasm, seeds
        seeds |= discovered


def _discover_bra_tables(disasm, rom) -> set:
    """Find PC-relative `bra` jump-table entries.

    A `jmp d(pc,Dn)` lands on a contiguous table of `bra` entries. Those entries
    are reachable only through the computed jump, so static following never
    decodes them. Return every entry address so they can be seeded as code.

    The table is uniform in one of the three `bra` widths; the stride is taken
    from the first entry:
      * `bra.s` — opcode 0x60xx with xx ∉ {0x00, 0xFF} (2-byte entries)
      * `bra.w` — opcode 0x6000 then a 16-bit displacement (4-byte entries)
      * `bra.l` — opcode 0x60FF then a 32-bit displacement (6-byte entries)
    """
    found = set()
    for instr in disasm.instructions.values():
        if instr.mnemonic not in ('jmp', 'jsr') or not instr.indirect:
            continue
        if not instr.eas or instr.eas[0].mode != EAMode.PC_INDEX:
            continue
        base = instr.eas[0].abs_value
        if base is None or (base & 1) or not rom.in_bounds(base, 2):
            continue
        first = rom.read_word(base)
        if first == 0x6000:                                      # bra.w
            stride = 4
        elif first == 0x60FF:                                    # bra.l
            stride = 6
        elif (first >> 8) == 0x60 and (first & 0xFF) not in (0x00, 0xFF):
            stride = 2                                           # bra.s
        else:
            continue
        addr = base
        while rom.in_bounds(addr, stride):
            word = rom.read_word(addr)
            # Every entry must be the same `bra` width as the first.
            if stride == 4 and word != 0x6000:
                break
            if stride == 6 and word != 0x60FF:
                break
            if stride == 2 and ((word >> 8) != 0x60 or (word & 0xFF) in (0x00, 0xFF)):
                break
            found.add(addr)
            addr += stride
            if addr - base > 0x400:  # sanity cap; no real table is this large
                break
    return found


# Word jump-table entries hold a 16-bit absolute address, so they can only
# reach code below 64 KiB; valid targets sit above the vector table / header.
_CODE_LO, _CODE_HI = 0x000200, 0x010000


def _read_word_table(rom, base, high_prefix=0, allow_backward=False) -> set:
    """Read a forward, self-bounded word jump table starting at *base*.

    Each entry is a 16-bit absolute address. The table is accepted only if it is
    clean: every entry is an even, in-range code address and the entries occupy
    exactly `[base, earliest-target)` — the storage cannot overlap its own first
    handler. Returns the set of target addresses, or empty if it is not a table.
    """
    if high_prefix:
        return _read_prefixed_word_table(rom, base, high_prefix)
    if allow_backward:
        return _read_backward_tolerant_word_table(rom, base)

    targets, addr, min_target = [], base, None
    while rom.in_bounds(addr, 2):
        if min_target is not None and addr >= min_target:
            break  # the table cannot extend into its own earliest target
        word = rom.read_word(addr)
        if (word & 1) or not (_CODE_LO <= word < _CODE_HI):
            break  # not a plausible even, in-range code address
        targets.append(word)
        min_target = word if min_target is None else min(min_target, word)
        addr += 2
        if addr - base > 0x100:  # sanity cap; no real table is this large
            break
    if targets and min_target is not None and min_target > base and addr == min_target:
        return set(targets)
    return set()


def _read_backward_tolerant_word_table(rom, base) -> set:
    """Read a word table that may branch back to already-known handlers.

    Some shared-dispatcher tables are inline before their first new handler, but
    early entries point backward into handlers discovered from other tables. The
    table is therefore bounded by the earliest target greater than the table
    base, not by the absolute minimum target.
    """
    if (base & 1) or not rom.in_bounds(base, 2):
        return set()

    targets, addr, end = [], base, None
    while rom.in_bounds(addr, 2):
        if end is not None and addr >= end:
            break
        word = rom.read_word(addr)
        if (word & 1) or not (_CODE_LO <= word < _CODE_HI):
            break
        targets.append(word)
        if word > base:
            end = word if end is None else min(end, word)
        addr += 2
        if addr - base > 0x100:  # sanity cap; no real table is this large
            break
    if targets and end is not None and addr == end:
        return set(targets)
    return set()


def _read_prefixed_word_table(rom, base, high_prefix) -> set:
    """Read a word table whose entries inherit the destination register's high word.

    SoR uses a shared dispatcher shape like:

        moveq   #bank, Dn
        swap    Dn              ; Dn high word now selects a 64K ROM bank
        ...
        move.w  (An,Dn.w), Dn   ; low word comes from the table
        movea.l Dn, Am
        jmp     (Am)

    The table's first target usually starts immediately after the table, but
    later entries may point backward into already-known handlers. That makes the
    stricter minimum-target bound used by plain 16-bit tables invalid here, so
    the first entry defines the table end instead.
    """
    if (base & 1) or not rom.in_bounds(base, 2):
        return set()
    first = high_prefix | rom.read_word(base)
    if (first & 1) or first <= base or not rom.in_bounds(first, 2):
        return set()

    targets, addr = [], base
    while addr < first and rom.in_bounds(addr, 2):
        word = rom.read_word(addr)
        target = high_prefix | word
        if (target & 1) or target < _CODE_LO or not rom.in_bounds(target, 2):
            return set()
        targets.append(target)
        addr += 2
        if addr - base > 0x100:  # sanity cap; no real table is this large
            return set()
    return set(targets) if addr == first else set()


def _feeds_indirect_jump(disasm, instr) -> bool:
    """True if a register-indirect `jmp`/`jsr` follows *instr* within a few steps."""
    nxt = disasm.instructions.get(instr.next_address)
    for _ in range(3):
        if nxt is None:
            return False
        if nxt.mnemonic in ('jmp', 'jsr') and nxt.indirect:
            return True
        nxt = disasm.instructions.get(nxt.next_address)
    return False


def _dreg_high_prefix(ordered, index_of, move_instr, dreg):
    """Return a preserved high-word prefix for a table load into Dn, if known."""
    k = index_of.get(move_instr.address)
    if k is None:
        return 0
    for j in range(k - 1, max(k - 9, -1), -1):
        prev = ordered[j]
        if (prev.mnemonic == 'swap' and prev.eas
                and prev.eas[0].mode == EAMode.DATA_REG
                and prev.eas[0].reg == dreg):
            for between in ordered[j + 1:k]:
                if (between.eas and between.eas[-1].mode == EAMode.DATA_REG
                        and between.eas[-1].reg == dreg
                        and (between.size == 'l' or between.mnemonic in ('moveq', 'swap'))):
                    return 0
            for i in range(j - 1, max(j - 5, -1), -1):
                seed = ordered[i]
                if (seed.mnemonic == 'moveq' and len(seed.eas) >= 2
                        and seed.eas[0].mode == EAMode.IMMEDIATE
                        and seed.eas[1].mode == EAMode.DATA_REG
                        and seed.eas[1].reg == dreg):
                    return (seed.eas[0].imm & 0xFFFF) << 16
            return 0
    return 0


def _discover_word_jump_tables(disasm, rom) -> set:
    """Find word-address jump/call tables and return their target addresses.

    A common dispatcher loads a 16-bit absolute code address out of a PC-relative
    table and jumps/calls through it, in two shapes:

        move.w  d(pc,Dn.w), Dm     ; (a) table base is PC-relative in the move
        movea.l Dm, An
        jmp     (An)

        lea     table(pc), An      ; (b) table base set up by a preceding lea
        move.w  (An,Dn.w), Dm
        movea.l Dm, Am
        jsr     (Am)

    Unlike `bra` tables the entries are raw word addresses and the jump is
    register-indirect, so `_discover_bra_tables` never sees them. The entries are
    reachable only through the computed jump, so static following never decodes
    them. Return every target so they can be seeded as code.

    Only forward, self-bounded tables are accepted: the entries occupy
    `[base, earliest-target)` (the storage cannot overlap its own first handler),
    every entry is an even address inside the word-addressable code range, and a
    register-indirect jmp/jsr must follow within a few instructions. These guards
    keep PC-relative *data* word reads from being mistaken for jump tables.
    """
    found = set()
    ordered = sorted(disasm.instructions.values(), key=lambda i: i.address)
    index_of = {ins.address: k for k, ins in enumerate(ordered)}

    def _lea_base(move_instr, an_reg):
        """Resolve `lea d(pc),A{an_reg}` set up shortly before `move_instr`."""
        k = index_of.get(move_instr.address)
        if k is None:
            return None
        for j in range(k - 1, max(k - 9, -1), -1):
            prev = ordered[j]
            if prev.mnemonic != 'lea' or len(prev.eas) < 2:
                continue
            src, dst = prev.eas[0], prev.eas[1]
            if (dst.mode == EAMode.ADDR_REG and dst.reg == an_reg
                    and src.mode in (EAMode.PC_DISP, EAMode.PC_INDEX)):
                return src.abs_value
        return None

    for instr in disasm.instructions.values():
        if instr.mnemonic != 'move' or instr.size != 'w':
            continue
        if (len(instr.eas) < 2 or instr.eas[1].mode != EAMode.DATA_REG):
            continue
        dst_reg = instr.eas[1].reg
        src = instr.eas[0]
        if src.mode == EAMode.PC_INDEX:
            base = src.abs_value                       # shape (a)
        elif src.mode == EAMode.ADDR_INDEX and src.reg is not None:
            base = _lea_base(instr, src.reg)           # shape (b)
        else:
            continue
        if base is None:
            continue

        # Confirm the loaded word feeds a register-indirect jmp/jsr shortly after.
        if not _feeds_indirect_jump(disasm, instr):
            continue

        found |= _read_word_table(
            rom, base, _dreg_high_prefix(ordered, index_of, instr, dst_reg))
    return found


def _discover_shared_dispatcher_tables(disasm, rom) -> set:
    """Find word jump tables reached through a *shared* register-indirect dispatcher.

    Many object routines set a table base then tail-jump to a common dispatcher:

        lea     table(pc), An      ; in the caller routine
        jmp     dispatcher         ; (or bra/jsr)
      dispatcher:
        move.w  (An,Dn.w), Dm      ; An is the incoming table base
        movea.l Dm, Ak
        jmp     (Ak)

    `_discover_word_jump_tables` misses these because the `lea` (caller) and the
    indexed `move.w` (dispatcher) live in different functions, so its same-flow
    `lea` lookback never sees the base. Here we recognise the dispatcher by the
    fact that its index register An is *not* written between the function entry
    and the `move.w` (i.e. An arrives as a parameter), then resolve a table base
    from every caller that `lea`s one immediately before jumping to the entry.
    """
    found = set()
    ordered = sorted(disasm.instructions.values(), key=lambda i: i.address)
    index_of = {ins.address: k for k, ins in enumerate(ordered)}
    subs = sorted(disasm.subroutines)

    # caller index: target address -> instructions that branch/jump/call to it.
    callers: dict[int, list] = {}
    for ins in ordered:
        for tgt in (ins.targets or ()):
            callers.setdefault(tgt, []).append(ins)

    def _writes_areg(ins, an_reg):
        return bool(ins.eas) and ins.eas[-1].mode == EAMode.ADDR_REG \
            and ins.eas[-1].reg == an_reg

    def _lea_base_at(call_instr, an_reg):
        """Resolve `lea d(pc),A{an_reg}` shortly before *call_instr* (caller side)."""
        k = index_of.get(call_instr.address)
        if k is None:
            return None
        for j in range(k - 1, max(k - 9, -1), -1):
            prev = ordered[j]
            if prev.mnemonic == 'lea' and len(prev.eas) >= 2:
                src, dst = prev.eas[0], prev.eas[1]
                if (dst.mode == EAMode.ADDR_REG and dst.reg == an_reg
                        and src.mode in (EAMode.PC_DISP, EAMode.PC_INDEX)):
                    return src.abs_value
        return None

    for instr in disasm.instructions.values():
        if instr.mnemonic != 'move' or instr.size != 'w':
            continue
        if (len(instr.eas) < 2 or instr.eas[1].mode != EAMode.DATA_REG):
            continue
        src = instr.eas[0]
        if src.mode != EAMode.ADDR_INDEX or src.reg is None:
            continue
        an_reg = src.reg

        # The dispatcher may be the function entry or an extra entry inside a
        # larger routine. For each concrete caller target before the indexed
        # move, An must arrive untouched from that target (a parameter, not
        # local setup).
        i = bisect.bisect_right(subs, instr.address) - 1
        if i < 0:
            continue
        entry = subs[i]
        k_entry, k_here = index_of.get(entry), index_of.get(instr.address)
        if k_entry is None or k_here is None:
            continue
        if not _feeds_indirect_jump(disasm, instr):
            continue

        high_prefix = _dreg_high_prefix(ordered, index_of, instr, instr.eas[1].reg)
        for j in range(k_entry, k_here + 1):
            target_callers = callers.get(ordered[j].address, ())
            if not target_callers:
                continue
            if any(_writes_areg(ordered[k], an_reg) for k in range(j, k_here)):
                continue  # An set locally after this entry point
            for caller in target_callers:
                base = _lea_base_at(caller, an_reg)
                if base is not None:
                    found |= _read_word_table(
                        rom, base, high_prefix, allow_backward=True)
    return found


def _load_aux(path):
    return parse_address_lines(path, code_only=True, warn_prefix='[recompile]')


def _expand_speculative_entries(rom, baseline_disasm, candidates) -> set:
    """Return every valid aligned entry inside each speculative candidate run.

    ``speculative_scan`` deliberately records one address per plausible routine.
    An indirect call can nevertheless land at another word-aligned decode inside
    that byte range.  For example, the primary stream has an instruction at
    $012B92 whose next address is $012B96, while $012B94 is also a valid entry
    with a different overlapping decode.  Validate every even address up to the
    candidate's straight-line terminator so discovery compiles both streams.
    """
    candidates = set(candidates or [])
    if not candidates:
        return set()

    # Lazy import avoids a module-import cycle: speculative_scan uses the
    # recompiler's fixpoint table discovery, while the CLI needs its validator
    # only after both modules have been loaded.
    from tools.disassembler.rom_map import RomMap
    from tools.speculative_scan.main import Validator

    rom_map = RomMap(rom.size, baseline_disasm.instructions,
                     baseline_disasm.subroutines,
                     baseline_disasm.labels).format()
    validator = Validator(rom, rom_map)
    baseline_boundaries = frozenset(ord(c) for c in 'SsLC')
    entries = set()
    for start in sorted(candidates):
        if not rom.in_bounds(start, 2) or (start & 1):
            continue
        if not validator.terminates(start):
            continue
        stop = validator.run_end(start)
        if stop <= start:
            stop = start + 2
        for addr in range(start, stop, 2):
            if not rom.in_bounds(addr, 2):
                break
            # A known instruction boundary already has a baseline dispatch
            # owner. Continuation bytes and unknown bytes remain valid alternate
            # speculative starts and must be checked.
            if rom_map[addr] in baseline_boundaries:
                continue
            if validator.terminates(addr):
                entries.add(addr)
    return entries


def main(argv=None):
    ap = argparse.ArgumentParser(description='Recompile the SoR ROM to C++.')
    ap.add_argument('rom', help='path to the ROM binary')
    ap.add_argument('-o', '--out-dir', default='src/generated',
                    help='output directory for Sor.hpp / Sor.cpp')
    ap.add_argument('--aux', default='code-analysis/aux_addresses.txt',
                    help='auxiliary entry-point address file: indirect-jump '
                         'targets from the active disassembly (optional)')
    ap.add_argument('--speculative', default='',
                    help='speculative candidate file: compile every valid '
                         'aligned address in those candidate ranges as an exact '
                         'entry; '
                         'confirm indirect dispatches so the runtime can promote '
                         'the actual target to the aux file. Disabled unless '
                         'this option is provided.')
    ap.add_argument('--labels-csv', default='code-analysis/labels.csv',
                    help='CSV of code-segment names (address,label,comment)')
    ap.add_argument('--addresses-csv', default='code-analysis/addresses.csv',
                    help='CSV of address labels (address,label,comment)')
    ap.add_argument('--manual-functions', default='',
                    help='address file listing functions declared and dispatched '
                         'normally but implemented in hand-written C++')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args(argv)

    rom = ROM.from_file(args.rom)

    # Phase 1: baseline disassembly (aux only) with iterative table discovery.
    seeds = set(_load_aux(args.aux))
    disasm, seeds = _disassemble_to_fixpoint(rom, seeds, args.verbose)
    baseline_entries = set(disasm.subroutines)
    baseline_instrs = set(disasm.instructions.keys())

    # Phase 2: merge speculative seeds and re-run table discovery to convergence.
    # The _transfer fix (addrs_set check) ensures that speculative entries which
    # straddle baseline-instruction ranges don't generate invalid cross-fn gotos.
    speculative_raw = set(_load_aux(args.speculative)) - seeds  # skip already-known
    speculative_decode_entries = _expand_speculative_entries(
        rom, disasm, speculative_raw) - seeds
    if speculative_decode_entries:
        disasm, seeds = _disassemble_to_fixpoint(
            rom, seeds | speculative_decode_entries, args.verbose)
    # The aligned alternatives are decode seeds, not separate body owners. Keep
    # the scanner's original starts (plus genuinely derived call/table targets)
    # as generated function roots; exact alternatives become lightweight entry
    # wrappers into those grouped bodies.
    alternate_decode_seeds = speculative_decode_entries - speculative_raw
    generated_subroutines = set(disasm.subroutines) - alternate_decode_seeds
    speculative_derived = generated_subroutines - baseline_entries
    # Every instruction decoded only after adding the speculative seeds is a
    # possible run-time entry.  Function-pointer tables are allowed to point
    # into the middle of a routine, not just at the candidate start selected by
    # speculative_scan, so expose every one of these instruction boundaries to
    # dispatch.  Generator keeps their bodies grouped and emits lightweight
    # entry wrappers, avoiding code duplication and recursive native calls for
    # 68000 loops.
    speculative_entries = set(disasm.instructions) - baseline_instrs
    # Discovery must also cover indirect calls into the middle of code already
    # decoded by the baseline pass. Those addresses are not speculative code and
    # therefore can never be found by speculative_scan, but they still need an
    # exact dispatch wrapper so the active run can confirm them without exit 42.
    confirm_entries = (set(disasm.instructions) - baseline_entries
                       if args.speculative else set())

    if args.verbose:
        print(f'[recompile] {len(seeds)} seed addresses, '
              f'{len(speculative_decode_entries)} speculative decode entries, '
              f'{len(speculative_derived)} speculative-derived entries, '
              f'{len(speculative_entries)} speculative instruction entries, '
              f'{len(confirm_entries)} confirmable exact entries',
              file=sys.stderr)

    # Function names and intra-function goto labels: code-segment labels
    # (labels.csv) take precedence over the general address labels
    # (addresses.csv). Names at non-code addresses are harmlessly ignored.
    names = {}
    if os.path.exists(args.addresses_csv):
        names.update({a: x.label for a, x in
                      _load_csv_addresses(args.addresses_csv).items()})
    if os.path.exists(args.labels_csv):
        names.update(_load_labels_csv(args.labels_csv)[0])

    manual_functions = set(_load_aux(args.manual_functions))
    unknown_manual = manual_functions - set(disasm.subroutines)
    if unknown_manual:
        formatted = ', '.join(f'${address:06X}' for address in sorted(unknown_manual))
        ap.error(f'manual function address(es) are not subroutine entries: {formatted}')

    gen = Generator(disasm.instructions, generated_subroutines,
                    rom_path=args.rom, names=names,
                    speculative_addrs=speculative_entries,
                    speculative_scope=speculative_derived,
                    baseline_instrs=baseline_instrs,
                    manual_functions=manual_functions,
                    confirm_addrs=confirm_entries,
                    rom=rom)
    source = gen.emit_source()   # must run first — populates self._rejected
    header = gen.emit_header()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'Sor.hpp'), 'w') as f:
        f.write(header)
    with open(os.path.join(args.out_dir, 'Sor.cpp'), 'w') as f:
        f.write(source)

    s = gen.stats
    total = s.handled + s.stubbed
    print(f'[recompile] {len(disasm.instructions)} instructions, '
          f'{len(gen.part.entries)} functions')
    if manual_functions:
        print(f'[recompile] {len(manual_functions)} function(s) supplied by hand-written C++')
    print(f'[recompile] translated {s.handled}/{total} '
          f'({100 * s.handled / total:.1f}%), {s.stubbed} stubbed')
    if s.stub_mnemonics:
        top = sorted(s.stub_mnemonics.items(), key=lambda kv: -kv[1])
        print('[recompile] stubbed opcodes: ' +
              ', '.join(f'{m}×{n}' for m, n in top))
    print(f'[recompile] wrote {args.out_dir}/Sor.hpp, {args.out_dir}/Sor.cpp')
    return 0


if __name__ == '__main__':
    sys.exit(main())
