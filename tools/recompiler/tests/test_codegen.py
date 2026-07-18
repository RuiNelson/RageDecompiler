"""Unit tests for the recompiler codegen (EA expressions + opcode lowering)."""

from pathlib import Path
import os
import re
import subprocess
import sys

from tools.disassembler.instruction import EA, EAMode, FlowType, Instruction
from tools.disassembler.rom import ROM
from tools.recompiler import ea_codegen as ea
from tools.recompiler import main as recompiler_main
from tools.recompiler import opcodes
from tools.recompiler.ea_codegen import TempPool
from tools.recompiler.generator import Generator
from tools.recompiler.main import _expand_speculative_entries, _load_aux
from tools.recompiler.opcodes import Unsupported
from tools.recompiler.regions import partition

_RAGE_DECOMPILER_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE_ROOT = Path(
    os.environ.get(
        'SOR_RECOMPILATION_DIR',
        _RAGE_DECOMPILER_ROOT.parent / 'StreetsOfRageRecompilation',
    )
)


def _tp():
    return TempPool(0x1000)


def _run_recompiler(out_dir, *args):
    env = dict(os.environ)
    env['PYTHONPATH'] = f"{_RAGE_DECOMPILER_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, '-m', 'tools.recompiler',
         'rom/SOR.bin', '-o', str(out_dir), *args],
        cwd=_FIXTURE_ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _function_source(source, name):
    start = source.index(f'void Sor::{name}(')
    end = source.find('\n}\n\n', start)
    if end < 0:
        end = source.rindex('\n}', start)
    return source[start:end]


# --- effective-address codegen -------------------------------------------

def test_areg_a7_is_ssp():
    assert ea.areg(7) == 'cpu().ssp'
    assert ea.areg(3) == 'cpu().a[3]'


def test_read_data_reg_sizes():
    assert ea.read_ea(EA(EAMode.DATA_REG, reg=3), 'l', _tp())[1] == 'cpu().d[3]'
    assert ea.read_ea(EA(EAMode.DATA_REG, reg=0), 'b', _tp())[1] == \
        'BYTE(cpu().d[0] & 0xFFu)'


def test_read_postinc_has_side_effect_after_read():
    stmts, expr = ea.read_ea(EA(EAMode.ADDR_POSTINC, reg=0), 'w', _tp())
    joined = '\n'.join(stmts)
    assert 'readWord(cpu().a[0])' in joined
    assert 'cpu().a[0] += 2;' in joined
    # the read must precede the increment
    assert joined.index('readWord') < joined.index('+= 2')
    assert expr in joined  # value materialized into a temp


def test_a7_byte_postinc_and_predec_step_by_word():
    post, _ = ea.read_ea(EA(EAMode.ADDR_POSTINC, reg=7), 'b', _tp())
    pre, _ = ea.read_ea(EA(EAMode.ADDR_PREDEC, reg=7), 'b', _tp())

    assert 'cpu().ssp += 2;' in '\n'.join(post)
    assert 'cpu().ssp -= 2;' in '\n'.join(pre)


def test_non_a7_byte_postinc_still_steps_by_byte():
    stmts, _ = ea.read_ea(EA(EAMode.ADDR_POSTINC, reg=4), 'b', _tp())

    assert 'cpu().a[4] += 1;' in '\n'.join(stmts)


def test_read_predec_decrements_before_read():
    stmts, _ = ea.read_ea(EA(EAMode.ADDR_PREDEC, reg=1), 'l', _tp())
    joined = '\n'.join(stmts)
    assert joined.index('-= 4') < joined.index('readLong')


def test_write_subregister_uses_merge_helpers():
    assert ea.write_ea(EA(EAMode.DATA_REG, reg=2), 'b', 'v', _tp()) == \
        ['cpu().d[2] = LONG((cpu().d[2] & 0xFFFFFF00u) | LONG(BYTE(v)));']
    assert ea.write_ea(EA(EAMode.DATA_REG, reg=2), 'w', 'v', _tp()) == \
        ['cpu().d[2] = LONG((cpu().d[2] & 0xFFFF0000u) | LONG(WORD(v)));']


def test_write_addr_reg_word_sign_extends():
    out = ea.write_ea(EA(EAMode.ADDR_REG, reg=4), 'w', 'v', _tp())[0]
    assert 'static_cast<int32_t>' in out
    assert 'cpu().a[4]' in out


def test_address_of_disp_and_abs():
    assert ea.address_of(EA(EAMode.ADDR_DISP, reg=2, disp=4), _tp())[1] == \
        '(cpu().a[2] + 4)'
    assert ea.address_of(EA(EAMode.ABS_L, abs_value=0xFF0000), _tp())[1] == \
        '0x00FF0000u'


# --- opcode lowering -----------------------------------------------------

def _instr(mnem, size, eas, flow=FlowType.SEQUENTIAL):
    return Instruction(address=0x1000, mnemonic=mnem, size=size, operands=[],
                       byte_length=2, flow=flow, eas=eas)


def test_move_sets_logical_flags():
    out = '\n'.join(opcodes.emit_dataop(_instr(
        'move', 'l', [EA(EAMode.DATA_REG, reg=0), EA(EAMode.DATA_REG, reg=1)])))
    assert 'cpu().d[1]' in out
    assert 'cpu().setNZClearVC' in out


def test_movea_no_flags_sign_extends():
    out = '\n'.join(opcodes.emit_dataop(_instr(
        'move', 'w', [EA(EAMode.DATA_REG, reg=0), EA(EAMode.ADDR_REG, reg=1)])))
    assert 'setCCR' not in out and 'setFlag' not in out  # movea never touches CCR
    assert 'cpu().a[1]' in out                   # word source sign-extended
    assert 'static_cast<int16_t>' in out


def test_move_word_to_data_reg_preserves_high_word():
    """68000 MOVE.W to Dn merges the low word and preserves the high word."""
    out = '\n'.join(opcodes.emit_dataop(_instr(
        'move', 'w', [EA(EAMode.DATA_REG, reg=0), EA(EAMode.DATA_REG, reg=7)])))
    assert 'cpu().d[7]' in out and '0xFFFF0000u' in out
    assert 'cpu().setNZClearVC' in out


def test_move_word_to_memory_uses_ea_writer_not_data_reg_macro():
    out = '\n'.join(opcodes.emit_dataop(_instr(
        'move', 'w', [EA(EAMode.DATA_REG, reg=0), EA(EAMode.ADDR_IND, reg=1)])))
    assert '0xFFFF0000u' not in out
    assert 'memory().writeWord(cpu().a[1]' in out
    assert 'cpu().setNZClearVC' in out


def test_moveq_long_signext_and_flags():
    out = '\n'.join(opcodes.emit_dataop(_instr(
        'moveq', None, [EA(EAMode.IMMEDIATE, imm=-1), EA(EAMode.DATA_REG, reg=0)])))
    assert 'cpu().d[0]' in out and 'cpu().setNZClearVC' in out
    # Compact: sign-extended literal, no nested int8/int32 casts.
    assert '0xFFFFFFFFu' in out
    assert 'static_cast<int8_t>' not in out


def test_move_does_not_copy_already_materialized_temp():
    """move.b (a0)+, d0 must not emit a redundant t1 = t0."""
    out = '\n'.join(opcodes.emit_dataop(_instr(
        'move', 'b',
        [EA(EAMode.ADDR_POSTINC, reg=0), EA(EAMode.DATA_REG, reg=1)])))
    assert 'cpu().a[0] += 1;' in out
    assert out.count('m_byte t') == 1  # one temp from postinc, not a second copy


def test_disp_zero_omits_plus_zero():
    setup, addr = ea.address_of(EA(EAMode.ADDR_DISP, reg=1, disp=0), _tp())
    assert setup == []
    assert addr == 'cpu().a[1]'
    assert '+ 0' not in addr


def test_move_special_registers_use_cpu_sr_helpers():
    from_sr = '\n'.join(opcodes.emit_dataop(_instr(
        'move', 'w', [EA(EAMode.SPECIAL_REG, special='sr'), EA(EAMode.DATA_REG, reg=0)])))
    to_ccr = '\n'.join(opcodes.emit_dataop(_instr(
        'move', 'w', [EA(EAMode.IMMEDIATE, imm=0x1F), EA(EAMode.SPECIAL_REG, special='ccr')])))

    assert 'cpu().status()' in from_sr
    assert 'cpu().setCCR(WORD(' in to_ccr
    assert 'cpu().sr' not in from_sr
    assert 'cpu().sr' not in to_ccr


def test_add_uses_macro_and_writes_back():
    out = '\n'.join(opcodes.emit_dataop(_instr(
        'add', 'w', [EA(EAMode.DATA_REG, reg=0), EA(EAMode.DATA_REG, reg=1)])))
    assert '+ LONG(' in out
    assert 'cpu().d[1]' in out
    assert 'cpu().setNZVCX' in out


def test_cmp_sets_flags_without_writing():
    out = '\n'.join(opcodes.emit_dataop(_instr(
        'cmp', 'l', [EA(EAMode.DATA_REG, reg=0), EA(EAMode.DATA_REG, reg=1)])))
    assert '- static_cast<uint64_t>' in out
    assert 'cpu().d[1] =' not in out              # compare writes nothing


def test_adda_is_address_arith_no_flags():
    out = '\n'.join(opcodes.emit_dataop(_instr(
        'adda', 'l', [EA(EAMode.DATA_REG, reg=0), EA(EAMode.ADDR_REG, reg=1)])))
    assert 'cpu().a[1] = LONG(cpu().a[1] +' in out
    assert 'setCCR' not in out and 'setFlag' not in out


def test_shift_immediate_count():
    out = '\n'.join(opcodes.emit_dataop(_instr(
        'lsr', 'w', [EA(EAMode.IMMEDIATE, imm=3), EA(EAMode.DATA_REG, reg=2)])))
    assert 'static_cast<int>(3)' in out
    assert '>>= 1' in out
    assert 'cpu().setVCX(' in out                 # count is never 0, X = C


def test_shift_register_count_is_mod_64_and_zero_shifts_nothing():
    """Register count is Dn mod 64; a count of 0 must not become 8/16/32
    and must leave X unchanged (C and V are still cleared)."""
    out = '\n'.join(opcodes.emit_dataop(_instr(
        'lsl', 'w', [EA(EAMode.DATA_REG, reg=1), EA(EAMode.DATA_REG, reg=0)])))
    assert 'cpu().d[1] & 63' in out
    assert '!= 0 ?' not in out                    # no zero → 16 remap
    assert 'cpu().setVC(' in out
    assert '!= 0) cpu().setFlagX(' in out         # X only touched when count != 0


def test_movem_unsupported_via_generator():
    # movem is handled by the generator, not emit_dataop.
    assert opcodes.emit_dataop(_instr('movem', 'l', [], FlowType.SEQUENTIAL)) is None


def test_movem_reglist_crosses_data_to_addr():
    """d5-a4 spans D5..D7 then A0..A4 — common in SoR boot (movem.l (a5)+, d5-a4)."""
    regs = Generator._parse_reglist('d5-a4')
    assert regs == [(False, 5), (False, 6), (False, 7),
                    (True, 0), (True, 1), (True, 2), (True, 3), (True, 4)]


def test_movem_word_load_sign_extends_into_data_reg():
    """68000 MOVEM.W memory→register sign-extends each word — Dn included."""
    ins = {0x100: _instr('rts', None, [], FlowType.RETURN)}
    ins[0x100].address = 0x100
    gen = Generator(ins, {0x100})
    out = '\n'.join(gen._emit_movem(_instr(
        'movem', 'w',
        [EA(EAMode.ADDR_POSTINC, reg=0), EA(EAMode.DATA_REG, reg=2)])))
    assert 'static_cast<int16_t>' in out       # sign-extended…
    assert 'cpu().d[2] = LONG(' in out         # …into the full register
    assert '0xFFFF0000u' not in out            # no preserve-high merge


def test_scc_sets_byte_by_condition():
    out = '\n'.join(opcodes.emit_dataop(_instr(
        'sne', 'b', [EA(EAMode.DATA_REG, reg=6)])))
    assert 'BYTE(0xFF)' in out and 'BYTE(0)' in out
    assert 'cpu().condition(6)' in out
    assert 'cpu().d[6]' in out


def test_exg_swaps_registers():
    out = '\n'.join(opcodes.emit_dataop(_instr(
        'exg', 'l', [EA(EAMode.DATA_REG, reg=0), EA(EAMode.ADDR_REG, reg=1)])))
    assert 'cpu().d[0] = cpu().a[1];' in out
    assert 'cpu().a[1] =' in out


def test_abcd_uses_macro():
    out = '\n'.join(opcodes.emit_dataop(_instr(
        'abcd', 'b', [EA(EAMode.DATA_REG, reg=0), EA(EAMode.DATA_REG, reg=1)])))
    assert '& 0x0F' in out
    assert 'cpu().setFlag' in out


def test_flow_opcodes_return_none():
    assert opcodes.emit_dataop(_instr('bra', None, [], FlowType.BRANCH)) is None
    assert opcodes.emit_dataop(_instr('rts', None, [], FlowType.RETURN)) is None


def test_unsupported_opcode_raises():
    # tas is not reached by the SoR ROM and is intentionally not implemented;
    # the generator must reject it (hard error) rather than silently degrade.
    try:
        opcodes.emit_dataop(_instr('tas', 'b', [EA(EAMode.DATA_REG, reg=0)]))
    except Unsupported:
        return
    assert False, 'expected Unsupported for tas'


# --- region partitioning -------------------------------------------------

def test_irq_check_emitted_before_each_instruction():
    from tools.recompiler.generator import Generator
    ins = {0x100: _instr('nop', None, []),
           0x102: _instr('rts', None, [], FlowType.RETURN)}
    for a in ins:
        ins[a].address = a
    src = Generator(ins, {0x100}).emit_source()
    assert '#define BEFORE_INSTRUCTION if (irqLevel() > cpu().interruptMask()) serviceIRQ(); pace();' in src
    assert src.count('BEFORE_INSTRUCTION') == 3
    assert '(void)0;' in src
    assert 'cpu().ssp += 4;' in src
    assert 'void Sor::serviceIRQ()' in src
    assert '#define BYTE(v) static_cast<m_byte>(v)' in src
    assert '#define F_Z' not in src
    assert 'cpu().enterInterrupt(level);' in src
    assert '#include "M68KMacros.hpp"' not in src


# --- CCR liveness (omit dead flag updates) --------------------------------

def _fn_instr(addr, mnem, size, eas, flow=FlowType.SEQUENTIAL, bl=2, targets=None):
    ins = Instruction(address=addr, mnemonic=mnem, size=size, operands=[],
                      byte_length=bl, flow=flow, eas=eas or [],
                      targets=list(targets or []))
    return ins


def test_dead_move_flags_omitted_before_overwriting_move():
    """move; move; rts — only the last move's CCR update is observable."""
    ins = {
        0x100: _fn_instr(0x100, 'move', 'l',
                         [EA(EAMode.DATA_REG, reg=0), EA(EAMode.DATA_REG, reg=1)]),
        0x102: _fn_instr(0x102, 'move', 'l',
                         [EA(EAMode.DATA_REG, reg=2), EA(EAMode.DATA_REG, reg=3)]),
        0x104: _fn_instr(0x104, 'rts', None, [], FlowType.RETURN),
    }
    src = Generator(ins, {0x100}).emit_source()
    body = _function_source(src, 'sub_000100')
    # Data moves still happen…
    assert 'cpu().d[1] = LONG(' in body
    assert 'cpu().d[3] = LONG(' in body
    # …but only the second move updates flags (escape via rts).
    assert body.count('setNZClearVC') == 1
    first_move = body.split('// $000100')[1].split('// $000102')[0]
    second_move = body.split('// $000102')[1].split('// $000104')[0]
    assert 'setNZ' not in first_move
    assert 'setNZClearVC' in second_move


def test_live_flags_kept_before_conditional_branch():
    """move; beq; rts — the branch reads Z, so the move must set flags."""
    ins = {
        0x100: _fn_instr(0x100, 'move', 'l',
                         [EA(EAMode.DATA_REG, reg=0), EA(EAMode.DATA_REG, reg=1)]),
        0x102: _fn_instr(0x102, 'beq', None, [], FlowType.CONDITIONAL,
                         targets=[0x108]),
        0x104: _fn_instr(0x104, 'rts', None, [], FlowType.RETURN),
        0x108: _fn_instr(0x108, 'rts', None, [], FlowType.RETURN),
    }
    src = Generator(ins, {0x100}).emit_source()
    body = _function_source(src, 'sub_000100')
    assert 'setNZClearVC' in body
    assert 'condition(7)' in body


def test_dead_cmp_omitted_entirely():
    """cmp whose flags are immediately overwritten is a pure no-op."""
    ins = {
        0x100: _fn_instr(0x100, 'cmp', 'l',
                         [EA(EAMode.DATA_REG, reg=0), EA(EAMode.DATA_REG, reg=1)]),
        0x102: _fn_instr(0x102, 'move', 'l',
                         [EA(EAMode.DATA_REG, reg=2), EA(EAMode.DATA_REG, reg=3)]),
        0x104: _fn_instr(0x104, 'rts', None, [], FlowType.RETURN),
    }
    src = Generator(ins, {0x100}).emit_source()
    body = _function_source(src, 'sub_000100')
    # cmp block should have no arithmetic / flag work — only BEFORE scaffolding.
    cmp_region = body.split('// $000100')[1].split('// $000102')[0]
    assert 'setNZ' not in cmp_region
    assert ' - ' not in cmp_region
    assert 'cpu().d[3]' in body  # trailing move still present


def test_emit_dataop_live_flags_none_keeps_full_update():
    """Unit-level emit with live=None (default) still emits full CCR update."""
    out = '\n'.join(opcodes.emit_dataop(_instr(
        'move', 'l', [EA(EAMode.DATA_REG, reg=0), EA(EAMode.DATA_REG, reg=1)])))
    assert 'setNZClearVC' in out


def test_emit_dataop_empty_live_omits_flags():
    out = '\n'.join(opcodes.emit_dataop(
        _instr('move', 'l',
               [EA(EAMode.DATA_REG, reg=0), EA(EAMode.DATA_REG, reg=1)]),
        live_flags=frozenset()))
    assert 'cpu().d[1]' in out
    assert 'setNZ' not in out
    assert 'setFlag' not in out


def test_ccr_effects_move_vs_beq():
    from tools.recompiler import ccr_liveness as ccr
    move = _fn_instr(0x100, 'move', 'l',
                     [EA(EAMode.DATA_REG, reg=0), EA(EAMode.DATA_REG, reg=1)])
    beq = _fn_instr(0x102, 'beq', None, [], FlowType.CONDITIONAL, targets=[0x108])
    assert ccr.effects(move) == (frozenset(), ccr.NZVC)
    assert ccr.effects(beq) == (frozenset({ccr.Z}), frozenset())


def test_jsr_emits_nonlocal_return_guard():
    ins = {
        0x100: _instr('jsr', None, [], FlowType.CALL),
        0x106: _instr('rts', None, [], FlowType.RETURN),
        0x200: _instr('rts', None, [], FlowType.RETURN),
    }
    ins[0x100].byte_length = 6
    ins[0x100].targets = [0x200]
    for a in ins:
        ins[a].address = a

    src = Generator(ins, {0x100, 0x200}).emit_source()

    assert 'm_long sp_000100 = cpu().ssp;' in src
    assert 'memory().writeLong(cpu().ssp, LONG(0x0106u));' in src
    assert 'if ((cpu().ssp & 0x00FFFFFFu) > (sp_000100 & 0x00FFFFFFu)) return;' in src


def test_partition_assigns_to_nearest_entry():
    ins = {
        0x100: _instr('nop', None, []),
        0x102: _instr('rts', None, [], FlowType.RETURN),
        0x200: _instr('nop', None, []),
        0x202: _instr('rts', None, [], FlowType.RETURN),
    }
    for a in ins:
        ins[a].address = a
    part = partition(ins, {0x100, 0x200})
    assert part.entries == [0x100, 0x200]
    assert part.func_of(0x102) == 0x100
    assert part.func_of(0x202) == 0x200
    assert part.functions[0x100].addrs == [0x100, 0x102]


def test_load_aux_ignores_vector_table_and_odd_addresses(tmp_path):
    aux = tmp_path / 'aux.txt'
    aux.write_text('0000001e\n00000200\n00000201\n00000436 ; valid\n')

    assert _load_aux(aux) == [0x200, 0x436]


def test_load_aux_empty_path_disables_optional_inputs():
    assert _load_aux('') == []


def test_recompiler_default_emits_no_speculative_hooks(tmp_path):
    out = tmp_path / 'normal'

    _run_recompiler(out)

    source = (out / 'Sor.cpp').read_text()
    assert 'confirmSpeculative(' not in source


def test_recompiler_speculative_option_emits_speculative_hooks(tmp_path):
    out = tmp_path / 'discover'
    aux = tmp_path / 'aux_without_012b94.txt'
    aux.write_text('\n'.join(
        line for line in
        (_FIXTURE_ROOT / 'code-analysis/aux_addresses.txt').read_text().splitlines()
        if line.split(';')[0].split('#')[0].strip().upper()
        not in {'009214', '012B94'}
    ) + '\n')

    _run_recompiler(out, '--aux', str(aux),
                    '--speculative', 'code-analysis/speculative_addresses.txt')

    source = (out / 'Sor.cpp').read_text()
    raw_candidates = _load_aux(
        _FIXTURE_ROOT / 'code-analysis/speculative_addresses.txt')
    # Every instruction reached only by the speculative phase is exposed, so
    # there must be more exact-address hooks than scanner candidate starts.
    assert source.count('confirmSpeculative(') > len(raw_candidates)
    # Mid-instruction entries are real lightweight functions which forward to
    # their grouped owner with the precise 68000 address.
    assert 'void Sor::sub_0003be() {' in source
    assert 'sub_0003ba(0x03BEu);' in source
    assert ('case 0x00012B94u: confirmSpeculative(0x00012B94u); '
            'sub_012b94(); return;') in source
    assert 'void Sor::sub_012b94() {' in source
    # $009214 is already baseline code but not a baseline function entry. It
    # must still be exposed and confirmed during discovery so a mid-function
    # indirect call does not force exit 42 and a rebuild.
    assert ('case 0x9214u: confirmSpeculative(0x9214u); '
            'sub_009214(); return;') in source
    assert 'void Sor::sub_009214() {' in source
    goto_targets = set(re.findall(r'\bgoto ([A-Za-z_][A-Za-z0-9_]*);', source))
    defined_labels = set(re.findall(
        r'^\s*([A-Za-z_][A-Za-z0-9_]*):$', source, flags=re.MULTILINE))
    assert goto_targets <= defined_labels


def test_speculative_expansion_validates_overlapping_aligned_entries():
    # $200: move.l ($4E71).w,d0 ; $204: rts is the primary stream.
    # $202: nop ; $204: rts is a second valid aligned entry stream inside it.
    rom_data = bytearray(ROM.END + 1)
    rom_data[0x200:0x206] = b'\x20\x38\x4E\x71\x4E\x75'
    rom = ROM(bytes(rom_data))

    class EmptyBaseline:
        instructions = {}
        subroutines = set()
        labels = set()

    assert _expand_speculative_entries(
        rom, EmptyBaseline(), {0x200}) == {0x200, 0x202, 0x204}


def test_real_012b7a_candidate_expands_to_012b94():
    rom = ROM.from_file(str(_FIXTURE_ROOT / 'rom/SOR.bin'))
    seeds = set(_load_aux(_FIXTURE_ROOT / 'code-analysis/aux_addresses.txt'))
    seeds.discard(0x012B94)
    baseline, _ = recompiler_main._disassemble_to_fixpoint(rom, seeds)

    expanded = _expand_speculative_entries(rom, baseline, {0x012B7A})

    assert 0x012B94 in expanded


def test_disassemble_to_fixpoint_repeats_after_new_table_targets(monkeypatch):
    calls = []

    class FakeDisassembler:
        def __init__(self, rom, aux_addresses, verbose=False):
            self.rom = rom
            self.aux_addresses = set(aux_addresses)
            self.verbose = verbose
            self.subroutines = set()
            self.instructions = {}
            calls.append(self.aux_addresses)

        def disassemble(self):
            pass

    def fake_discover(disasm, rom):
        return {0x200} if 0x200 not in disasm.aux_addresses else set()

    monkeypatch.setattr(recompiler_main, 'Disassembler', FakeDisassembler)
    monkeypatch.setattr(recompiler_main, '_discover_table_targets', fake_discover)

    disasm, seeds = recompiler_main._disassemble_to_fixpoint('rom', {0x100})

    assert calls == [{0x100}, {0x100, 0x200}]
    assert disasm.aux_addresses == {0x100, 0x200}
    assert seeds == {0x100, 0x200}


def test_banked_word_dispatch_table_discovers_016d0a_without_runtime_aux():
    rom = ROM.from_file(str(_FIXTURE_ROOT / 'rom/SOR.bin'))
    seeds = set(_load_aux(str(_FIXTURE_ROOT / 'code-analysis/aux_addresses.txt')))
    seeds.discard(0x016D0A)

    _, fixed = recompiler_main._disassemble_to_fixpoint(rom, seeds)

    assert 0x016D0A in fixed


def test_shared_dispatcher_backward_table_discovers_00d62a_without_runtime_aux():
    rom = ROM.from_file(str(_FIXTURE_ROOT / 'rom/SOR.bin'))
    seeds = set(_load_aux(str(_FIXTURE_ROOT / 'code-analysis/aux_addresses.txt')))
    seeds.discard(0x00D62A)

    _, fixed = recompiler_main._disassemble_to_fixpoint(rom, seeds)

    assert 0x00D62A in fixed


def test_speculative_scope_does_not_confirm_derived_entries():
    ins = {
        0x100: _instr('rts', None, [], FlowType.RETURN),
        0x200: _instr('rts', None, [], FlowType.RETURN),
        0x300: _instr('rts', None, [], FlowType.RETURN),
    }
    for a in ins:
        ins[a].address = a

    src = Generator(ins, {0x100, 0x200, 0x300},
                    speculative_addrs={0x200},
                    speculative_scope={0x200, 0x300},
                    baseline_instrs={0x100}).emit_source()

    assert ('case 0x0200u: confirmSpeculative(0x0200u); '
            'sub_000200(); return;') in src
    assert 'confirmSpeculative(0x0200u);' not in _function_source(src, 'sub_000200')
    assert 'confirmSpeculative(0x0300u);' not in src


def test_every_speculative_instruction_has_an_exact_entry_function():
    ins = {
        0x100: _instr('rts', None, [], FlowType.RETURN),
        0x200: _instr('nop', None, []),
        0x202: _instr('nop', None, []),
        0x204: _instr('rts', None, [], FlowType.RETURN),
    }
    for address in ins:
        ins[address].address = address

    gen = Generator(ins, {0x100, 0x200},
                    speculative_addrs={0x200, 0x202, 0x204},
                    speculative_scope={0x200},
                    baseline_instrs={0x100})
    source = gen.emit_source()
    header = gen.emit_header()

    assert ('case 0x0200u: confirmSpeculative(0x0200u); '
            'sub_000200(); return;') in source
    assert ('case 0x0202u: confirmSpeculative(0x0202u); '
            'sub_000202(); return;') in source
    assert ('case 0x0204u: confirmSpeculative(0x0204u); '
            'sub_000204(); return;') in source
    assert 'void Sor::sub_000202() {\n    sub_000200(0x0202u);\n}' in source
    assert 'void Sor::sub_000204() {\n    sub_000200(0x0204u);\n}' in source
    assert 'case 0x0202u: goto L000202;' in _function_source(source, 'sub_000200')
    assert 'case 0x0204u: goto L000204;' in _function_source(source, 'sub_000200')
    assert 'void sub_000202();' in header
    assert 'void sub_000204();' in header


def test_baseline_mid_instruction_is_confirmable_in_discovery():
    ins = {
        0x100: _instr('nop', None, []),
        0x102: _instr('rts', None, [], FlowType.RETURN),
    }
    for address in ins:
        ins[address].address = address

    gen = Generator(ins, {0x100}, speculative_addrs=set(),
                    speculative_scope=set(), baseline_instrs=set(ins),
                    confirm_addrs={0x102})
    source = gen.emit_source()

    assert gen.part.func_of(0x102) == 0x100
    assert ('case 0x0102u: confirmSpeculative(0x0102u); '
            'sub_000102(); return;') in source
    assert 'void Sor::sub_000102() {\n    sub_000100(0x0102u);\n}' in source


def test_overlapping_speculative_flow_keeps_phase2_instruction_ownership():
    """A baseline entry inside an overlapping speculative byte stream must not
    steal and filter the later speculative instructions (the $014EAE case)."""
    ins = {
        0x100: _instr('nop', None, []),
        0x102: _instr('rts', None, [], FlowType.RETURN),
        0x104: _instr('nop', None, []),
        0x106: _instr('rts', None, [], FlowType.RETURN),
    }
    ins[0x100].byte_length = 4  # speculative stream jumps over baseline $102
    for address in ins:
        ins[address].address = address

    gen = Generator(ins, {0x100, 0x102},
                    speculative_addrs={0x100, 0x104, 0x106},
                    speculative_scope={0x100},
                    baseline_instrs={0x102})
    source = gen.emit_source()

    assert gen.part.func_of(0x104) == 0x100
    assert gen.part.func_of(0x106) == 0x100
    assert ('case 0x0104u: confirmSpeculative(0x0104u); '
            'sub_000104(); return;') in source
    assert ('case 0x0106u: confirmSpeculative(0x0106u); '
            'sub_000106(); return;') in source
    owner = _function_source(source, 'sub_000100')
    assert 'case 0x0104u: goto L000104;' in owner
    assert 'case 0x0106u: goto L000106;' in owner
    assert '// $000104 nop' in owner
    assert '// $000106 rts' in owner


def test_invalid_speculative_derived_entry_is_rejected_not_fatal():
    ins = {
        0x100: _instr('rts', None, [], FlowType.RETURN),
        0x200: _instr('rts', None, [], FlowType.RETURN),
        0x300: _instr('tas', 'b', [EA(EAMode.DATA_REG, reg=0)]),
    }
    for a in ins:
        ins[a].address = a

    src = Generator(ins, {0x100, 0x200, 0x300},
                    speculative_addrs={0x200},
                    speculative_scope={0x200, 0x300},
                    baseline_instrs={0x100}).emit_source()

    assert 'void Sor::sub_000300' not in src
    assert ('case 0x0200u: confirmSpeculative(0x0200u); '
            'sub_000200(); return;') in src
    assert 'confirmSpeculative(0x0200u);' not in _function_source(src, 'sub_000200')


def test_csv_names_applied_to_goto_labels():
    ins = {
        0x100: _instr('bra', None, [], FlowType.BRANCH),
        0x106: _instr('nop', None, []),
        0x108: _instr('rts', None, [], FlowType.RETURN),
    }
    ins[0x100].byte_length = 6
    ins[0x100].targets = [0x106]
    for a in ins:
        ins[a].address = a

    src = Generator(ins, {0x100}, names={0x106: 'my_loop'}).emit_source()

    assert 'my_loop:' in src
    assert 'goto my_loop;' in src
    assert 'L000106:' not in src


def test_manual_function_keeps_declaration_calls_and_dispatch_but_omits_body():
    ins = {
        0x100: _instr('jsr', None, [], FlowType.CALL),
        0x102: _instr('rts', None, [], FlowType.RETURN),
        0x200: _instr('rts', None, [], FlowType.RETURN),
    }
    ins[0x100].targets = [0x200]
    for address in ins:
        ins[address].address = address

    gen = Generator(ins, {0x100, 0x200}, names={0x200: 'manual_wait'},
                    manual_functions={0x200})
    source = gen.emit_source()
    header = gen.emit_header()

    assert 'void manual_wait(m_long entry_ = 0x0200u);' in header
    assert 'case 0x0200u: manual_wait(); return;' in source
    assert 'manual_wait();' in _function_source(source, 'sub_000100')
    assert 'void Sor::manual_wait(' not in source
