import contextlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
import pyvisa


SUPPORTED_2450_CURRENT_RANGES = (
    10e-9,
    100e-9,
    1e-6,
    10e-6,
    100e-6,
    1e-3,
    10e-3,
    100e-3,
    1.0,
)
NPLC_MIN = 0.01
NPLC_MAX = 10.0
MAX_2450_SOURCE_VOLTAGE = 210.0
SUPPORTED_2450_VOLTAGE_RANGES = (0.2, 2.0, 20.0, 200.0)


class MeasurementReadError(RuntimeError):
    """A required instrument reading failed; fabricated replacement data is forbidden."""


class InstrumentConfigurationError(RuntimeError):
    """The instrument rejected or did not apply the requested configuration."""


def validate_2450_idn(idn):
    text = str(idn).strip()
    upper = text.upper()
    if 'KEITHLEY' not in upper or '2450' not in upper:
        raise InstrumentConfigurationError(
            f'仅支持 Keithley 2450，当前仪器返回: {text or "<空>"}'
        )
    return text


def validate_distinct_addresses(bias_address, gate_address, gate_enabled=True):
    if gate_enabled and str(bias_address).strip().upper() == str(gate_address).strip().upper():
        raise ValueError('偏压表和栅压表不能使用同一个仪器地址')


def _match_supported_current_range(value):
    requested = float(value)
    for supported in SUPPORTED_2450_CURRENT_RANGES:
        if math.isclose(requested, supported, rel_tol=1e-9, abs_tol=supported * 1e-12):
            return supported
    supported_text = ', '.join(f'{item:g}' for item in SUPPORTED_2450_CURRENT_RANGES)
    raise ValueError(
        f'2450不支持固定电流量程 {requested:g} A；可用量程为 {supported_text} A 或 AUTO'
    )


def validate_current_range_limit(current_range, current_limit, label='电流'):
    limit = float(current_limit)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError(f'{label}限流必须为有限正数，当前值: {current_limit}')
    if str(current_range).strip().upper() == 'AUTO':
        return 'AUTO', limit
    matched_range = _match_supported_current_range(current_range)
    maximum = matched_range * 1.05
    if limit > maximum and not math.isclose(limit, maximum, rel_tol=1e-9):
        raise ValueError(
            f'{label}固定量程 {matched_range:g} A 最大允许限流为 {maximum:g} A，'
            f'当前限流 {limit:g} A；程序不会自动修改量程或限流'
        )
    return matched_range, limit


def validate_nplc(value, label='NPLC'):
    nplc = float(value)
    if not math.isfinite(nplc) or not (NPLC_MIN <= nplc <= NPLC_MAX):
        raise ValueError(
            f'{label}必须在 {NPLC_MIN:g} 到 {NPLC_MAX:g} 之间，当前值: {value}'
        )
    return nplc


def validate_source_voltage(value, label='源电压'):
    voltage = float(value)
    if (
        not math.isfinite(voltage)
        or abs(voltage) > MAX_2450_SOURCE_VOLTAGE
    ):
        raise ValueError(
            f'{label}必须在 ±{MAX_2450_SOURCE_VOLTAGE:g} V 内，当前值: {value}'
        )
    return voltage


def validate_positive_step(value, label='步长'):
    step = float(value)
    if not math.isfinite(step) or step <= 0:
        raise ValueError(f'{label}必须为有限正数，当前值: {value}')
    return step


def validate_step_divides_interval(start, stop, step, label='步进'):
    """Require an interval to contain an integer number of requested steps."""
    start_value = float(start)
    stop_value = float(stop)
    step_value = validate_positive_step(step, label)
    if not math.isfinite(start_value) or not math.isfinite(stop_value):
        raise ValueError(
            f'{label}的起点和终点必须为有限数，当前为 '
            f'{start} 到 {stop}'
        )
    distance = abs(stop_value - start_value)
    if math.isclose(distance, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return 0
    count = int(round(distance / step_value))
    if count <= 0 or not math.isclose(
        distance,
        count * step_value,
        rel_tol=1e-9,
        abs_tol=max(1e-12, distance * 1e-10),
    ):
        raise ValueError(
            f'{label}参数不能整除：从 {start_value:g} 到 {stop_value:g}，'
            f'步长为 {step_value:g}。请修改起点、终点或步长；'
            '程序不会自动改变步长。'
        )
    return count


def validate_program_step_plan(program, params):
    """Validate every normal voltage ramp/grid before an output can be enabled."""
    if program == 'iv':
        validate_step_divides_interval(
            0, params['v_start'], params['v_step'], 'IV到起始电压的步进'
        )
        validate_step_divides_interval(
            params['v_start'], params['v_end'], params['v_step'], 'IV扫描步进'
        )
    elif program == 'isd_vg':
        validate_step_divides_interval(
            0, params['Bias_target'], params['Bias_step'], 'Isd-Vg偏压爬坡'
        )
        validate_step_divides_interval(
            0, params['Vg_1st'], params['Vg_step'], 'Isd-Vg到第一栅压的步进'
        )
        validate_step_divides_interval(
            params['Vg_1st'], params['Vg_2nd'], params['Vg_step'],
            'Isd-Vg栅压扫描步进',
        )
    elif program == 'mapping':
        validate_step_divides_interval(
            params['vg_start'], params['vg_stop'], params['vg_step'],
            'Mapping栅压扫描步进',
        )
        validate_step_divides_interval(
            0, params['vg_start'], min(params['vg_step'], 0.1),
            'Mapping到初始栅压的步进',
        )
        validate_step_divides_interval(
            0, params['bias_max'], params['bias_step_up'],
            'Mapping偏压上升步进',
        )
        validate_step_divides_interval(
            params['bias_max'], params['bias_min'], params['bias_step_full'],
            'Mapping偏压全程扫描步进',
        )
        validate_step_divides_interval(
            params['bias_min'], 0, params['bias_step_up'],
            'Mapping偏压返回零点步进',
        )
    elif program == 'it':
        for prefix, name, enabled in (
            ('b', 'It偏压', True),
            ('g', 'It栅压', params.get('gate_enabled', False)),
        ):
            if not enabled:
                continue
            ramp_step = params[f'{prefix}_ramp_step']
            if params[f'{prefix}_mode'] == 'single':
                validate_step_divides_interval(
                    0, params[f'{prefix}_target'], ramp_step, f'{name}爬坡'
                )
            else:
                test_step = params[f'{prefix}_test_step']
                validate_step_divides_interval(
                    0, params[f'{prefix}_start'], ramp_step, f'{name}到起点的爬坡'
                )
                validate_step_divides_interval(
                    params[f'{prefix}_start'], params[f'{prefix}_end'],
                    test_step, f'{name}测量点步进',
                )
                validate_step_divides_interval(
                    0, test_step, ramp_step, f'{name}相邻测量点间的爬坡'
                )
    elif program == 'bias_switch':
        for key, label in (
            ('test_v', '偏压开关测试电压'),
            ('sw1_v', '偏压开关状态1'),
            ('sw2_v', '偏压开关状态2'),
        ):
            validate_step_divides_interval(
                0, params[key], params['b_ramp_step'], label
            )
        if params.get('gate_enabled', False):
            validate_step_divides_interval(
                0, params['g_target'], params['g_ramp_step'], '偏压开关栅压爬坡'
            )
    elif program == 'gate_switch':
        validate_step_divides_interval(
            0, params['b_target'], params['b_ramp_step'], '栅压开关偏压爬坡'
        )
        for key, label in (
            ('test_vg', '栅压开关测试电压'),
            ('sw1_vg', '栅压开关状态1'),
            ('sw2_vg', '栅压开关状态2'),
        ):
            validate_step_divides_interval(
                0, params[key], params['g_ramp_step'], label
            )
    elif program == 'arbitrary_bias':
        for index, (voltage, _duration) in enumerate(params['waveform'], 1):
            validate_step_divides_interval(
                0, voltage, params['b_ramp_step'], f'任意偏压第{index}段'
            )
        if params.get('gate_enabled', False):
            validate_step_divides_interval(
                0, params['g_target'], params['g_ramp_step'], '任意偏压栅压爬坡'
            )
    elif program == 'arbitrary_gate':
        validate_step_divides_interval(
            0, params['b_target'], params['b_ramp_step'], '任意栅压偏压爬坡'
        )
        for index, (voltage, _duration) in enumerate(params['waveform'], 1):
            validate_step_divides_interval(
                0, voltage, params['g_ramp_step'], f'任意栅压第{index}段'
            )
    elif program == 'break_junction':
        validate_step_divides_interval(
            0, params['V0'], params['ZERO_STEP_V'], '断裂结到起始电压的步进'
        )
        if 'step_mV' in params:
            validate_step_divides_interval(
                params['V0'], params['V_MAX'], params['step_mV'] / 1000.0,
                '断裂结初始扫描步进',
            )
    else:
        raise ValueError(f'未知的步进校验类型: {program}')


def validate_terminal(terminal, label='端子'):
    requested = str(terminal).strip().upper()
    if requested not in ('FRONT', 'FRON', 'REAR'):
        raise ValueError(f'{label}只能是 FRONT 或 REAR，当前值: {terminal}')
    return 'FRONT' if requested in ('FRONT', 'FRON') else 'REAR'


def validate_voltage_range(value, label='电压量程'):
    requested = float(value)
    for supported in SUPPORTED_2450_VOLTAGE_RANGES:
        if math.isclose(requested, supported, rel_tol=1e-9):
            return supported
    raise ValueError(
        f'{label}必须为 0.2、2、20 或 200 V，当前值: {value}'
    )


def validate_voltage_within_range(
    voltage,
    voltage_range,
    label='源电压',
):
    actual_voltage = validate_source_voltage(voltage, label)
    actual_range = validate_voltage_range(voltage_range)
    maximum = actual_range * 1.05
    if abs(actual_voltage) > maximum and not math.isclose(
        abs(actual_voltage), maximum, rel_tol=1e-9
    ):
        raise ValueError(
            f'{label} {actual_voltage:g} V 超出固定电压量程 '
            f'{actual_range:g} V 的允许范围 ±{maximum:g} V；'
            '程序不会自动改变量程'
        )
    return actual_voltage, actual_range


def clear_scpi_status(instrument):
    instrument.write('*CLS')


def drain_scpi_errors(instrument, max_errors=32):
    errors = []
    for _ in range(max_errors):
        response = instrument.query(':SYST:ERR?').strip()
        if response.startswith(('+0,', '0,')):
            return errors
        errors.append(response)
    errors.append(f'错误队列超过 {max_errors} 条，停止读取')
    return errors


def assert_no_scpi_errors(instrument, stage='仪器配置'):
    errors = drain_scpi_errors(instrument)
    if errors:
        raise InstrumentConfigurationError(
            f'{stage}被仪器拒绝: ' + ' | '.join(errors)
        )


def _query_float(instrument, command, label):
    try:
        value = float(instrument.query(command).strip())
    except Exception as exc:
        raise InstrumentConfigurationError(f'无法回读{label}: {exc}') from exc
    if not math.isfinite(value):
        raise InstrumentConfigurationError(f'{label}回读为非有限值: {value}')
    return value


def verify_current_configuration(
    instrument,
    *,
    nplc,
    current_range,
    current_limit,
    terminal=None,
    autozero_mode=None,
    label='仪器',
):
    actual_nplc = _query_float(instrument, ':SENS:CURR:NPLC?', f'{label} NPLC')
    if not math.isclose(actual_nplc, float(nplc), rel_tol=1e-6, abs_tol=1e-9):
        raise InstrumentConfigurationError(
            f'{label} NPLC未生效：请求 {float(nplc):g}，回读 {actual_nplc:g}'
        )

    auto = int(round(_query_float(
        instrument, ':SENS:CURR:RANG:AUTO?', f'{label}自动量程状态'
    )))
    if str(current_range).strip().upper() == 'AUTO':
        if auto != 1:
            raise InstrumentConfigurationError(f'{label}自动量程未生效')
        actual_range = 'AUTO'
    else:
        expected_range = _match_supported_current_range(current_range)
        if auto != 0:
            raise InstrumentConfigurationError(f'{label}固定量程未生效')
        actual_range = _query_float(
            instrument, ':SENS:CURR:RANG?', f'{label}电流量程'
        )
        if not math.isclose(actual_range, expected_range, rel_tol=1e-6, abs_tol=expected_range * 1e-9):
            raise InstrumentConfigurationError(
                f'{label}电流量程未生效：请求 {expected_range:g} A，回读 {actual_range:g} A'
            )

    actual_limit = _query_float(
        instrument, ':SOUR:VOLT:ILIM?', f'{label}电流限流'
    )
    if not math.isclose(actual_limit, float(current_limit), rel_tol=1e-6, abs_tol=1e-12):
        raise InstrumentConfigurationError(
            f'{label}限流未生效：请求 {float(current_limit):g} A，回读 {actual_limit:g} A'
        )

    actual_terminal = None
    if terminal:
        validate_terminal(terminal, f'{label}端子')
        actual_terminal = instrument.query(':ROUT:TERM?').strip().upper()
        requested_terminal = str(terminal).strip().upper()
        terminal_aliases = {
            'FRONT': 'FRON',
            'FRON': 'FRON',
            'REAR': 'REAR',
        }
        if (
            terminal_aliases.get(requested_terminal, requested_terminal)
            != terminal_aliases.get(actual_terminal, actual_terminal)
        ):
            raise InstrumentConfigurationError(
                f'{label}端子未生效：请求 {terminal}，回读 {actual_terminal}'
            )

    actual_autozero = int(round(_query_float(
        instrument, ':SENS:CURR:AZER?', f'{label}自动调零状态'
    )))
    if autozero_mode is not None:
        expected_autozero = 1 if autozero_mode == 'continuous' else 0
        if actual_autozero != expected_autozero:
            raise InstrumentConfigurationError(
                f'{label}自动调零状态未生效：请求 {autozero_mode}，'
                f'回读 {actual_autozero}'
            )

    assert_no_scpi_errors(instrument, f'{label}配置')
    return {
        'nplc': actual_nplc,
        'current_range_A': actual_range,
        'current_limit_A': actual_limit,
        'terminal': actual_terminal,
        'autozero': actual_autozero,
    }


def configure_current_autozero(instrument, mode):
    """Configure Model 2450 current autozero without relying on *RST defaults."""
    if mode == 'continuous':
        instrument.write(':SENS:CURR:AZER ON')
        return
    if mode == 'block_once':
        instrument.write(':SENS:CURR:AZER OFF')
        instrument.query(':SENS:AZER:ONCE;*OPC?')
        return
    if mode == 'off':
        instrument.write(':SENS:CURR:AZER OFF')
        return
    raise ValueError(f'未知自动调零模式: {mode}')


def required_float_query(instrument, command, label='读数'):
    try:
        value = float(instrument.query(command).strip())
    except Exception as exc:
        raise MeasurementReadError(f'{label}失败: {exc}') from exc
    if not math.isfinite(value):
        raise MeasurementReadError(f'{label}返回非有限值: {value}')
    return value


def reliable_output_off(instrument, label='仪器'):
    """Best-effort independent shutdown commands; never skip OFF after another failure."""
    failures = []
    if instrument is None:
        return False, [f'{label}未连接，无法确认输出状态']
    for command in (':ABOR', ':SOUR:VOLT 0', ':OUTP OFF'):
        try:
            instrument.write(command)
        except Exception as exc:
            failures.append(f'{command}: {exc}')
    confirmed = False
    try:
        confirmed = int(float(instrument.query(':OUTP?').strip())) == 0
        if not confirmed:
            failures.append(':OUTP? 回读仍为开启')
    except Exception as exc:
        failures.append(f'无法回读输出状态: {exc}')
    return confirmed, failures


def generate_exact_ramp_levels(current, target, max_step):
    """Return source levels that reach target without exceeding max_step."""
    current_value = float(current)
    target_value = float(target)
    step_value = validate_positive_step(max_step, '归零步长')
    if not math.isfinite(current_value) or not math.isfinite(target_value):
        raise ValueError('归零起点和终点必须为有限数')
    distance = abs(target_value - current_value)
    tolerance = max(1e-12, distance * 1e-12)
    if distance <= tolerance:
        return []
    direction = 1.0 if target_value > current_value else -1.0
    full_steps = int(math.floor((distance + tolerance) / step_value))
    levels = [
        current_value + direction * step_value * index
        for index in range(1, full_steps + 1)
    ]
    if levels and (
        (direction > 0 and levels[-1] > target_value)
        or (direction < 0 and levels[-1] < target_value)
    ):
        levels.pop()
    if not levels or not math.isclose(
        levels[-1], target_value, rel_tol=1e-10, abs_tol=tolerance
    ):
        levels.append(target_value)
    else:
        levels[-1] = target_value
    return levels


def _is_full_step(previous, current, max_step):
    return math.isclose(
        abs(float(current) - float(previous)),
        float(max_step),
        rel_tol=1e-9,
        abs_tol=max(1e-12, abs(float(max_step)) * 1e-9),
    )


def fast_shutdown_zero_2450(
    instrument,
    max_step,
    *,
    label='仪器',
    force_event=None,
    chunk_changes=250,
):
    """Use the 2450 trigger model for final zeroing, then confirm output OFF."""
    started = time.monotonic()
    report = {
        'status': 'complete',
        'start_voltage': None,
        'target_voltage': 0.0,
        'requested_step': float(max_step),
        'step_count': 0,
        'chunk_count': 0,
        'elapsed_s': 0.0,
        'zero_readback': None,
        'output_off_confirmed': False,
        'errors': [],
    }
    buffer_name = f'zero_{time.monotonic_ns() & 0xFFFFFFFFFFFF:x}'
    buffer_created = False
    original_timeout = None
    timeout_changed = False
    try:
        if instrument is None:
            raise InstrumentConfigurationError(f'{label}未连接')
        if force_event is not None and force_event.is_set():
            report['status'] = 'emergency_off'
            confirmed, failures = reliable_output_off(instrument, label)
            report['output_off_confirmed'] = confirmed
            report['errors'].extend(failures)
            try:
                report['zero_readback'] = required_float_query(
                    instrument, ':SOUR:VOLT?', f'{label}紧急归零回读'
                )
            except Exception as exc:
                report['errors'].append(str(exc))
            zero_confirmed = (
                report['zero_readback'] is not None
                and math.isclose(
                    report['zero_readback'], 0.0,
                    rel_tol=0.0, abs_tol=1e-9,
                )
            )
            if not confirmed or not zero_confirmed:
                report['status'] = 'unconfirmed'
            return report
        if int(chunk_changes) <= 0:
            raise ValueError('内部归零每批步数必须为正整数')
        step_value = validate_positive_step(max_step, f'{label}归零步长')
        current = required_float_query(
            instrument, ':SOUR:VOLT?', f'{label}归零起点回读'
        )
        report['start_voltage'] = current
        levels = generate_exact_ramp_levels(current, 0.0, step_value)
        report['step_count'] = len(levels)

        # Formal acquisition sessions commonly use a short VISA timeout
        # (5–10 s).  A 250-change internal zeroing batch can legitimately take
        # longer even at NPLC 0.01, so that measurement timeout must not abort
        # an otherwise healthy final sweep.  Restore it after cleanup.
        try:
            original_timeout = instrument.timeout
            if (
                original_timeout is not None
                and float(original_timeout) < 120000
            ):
                instrument.timeout = 120000
                timeout_changed = True
        except Exception:
            original_timeout = None
            timeout_changed = False

        instrument.write(':ABOR')
        instrument.write(':SENS:CURR:NPLC 0.01')
        instrument.write(':SENS:CURR:AZER OFF')
        instrument.write(':SOUR:VOLT:READ:BACK OFF')
        # The generated sweep measures current at every source level.  Keep
        # that internal settling action: with a 1 nA gate limit, removing it
        # can drive the source into limit before the voltage reaches zero.
        # Select the smallest standard measurement range compatible with the
        # existing limit so autorange cannot leave a bias meter on an
        # unnecessarily slow lower range.  Verify that the protection level
        # remains bit-for-bit effective before initiating the sweep.
        current_limit = required_float_query(
            instrument, ':SOUR:VOLT:ILIM?', f'{label}归零前限流回读'
        )
        if current_limit >= SUPPORTED_2450_CURRENT_RANGES[0]:
            zero_measure_range = next(
                (
                    range_value
                    for range_value in SUPPORTED_2450_CURRENT_RANGES
                    if current_limit <= range_value * 1.05 + 1e-15
                ),
                SUPPORTED_2450_CURRENT_RANGES[-1],
            )
            instrument.write(f':SENS:CURR:RANG {zero_measure_range:.12g}')
            applied_limit = required_float_query(
                instrument, ':SOUR:VOLT:ILIM?', f'{label}归零限流复核'
            )
            if not math.isclose(
                applied_limit, current_limit, rel_tol=1e-9, abs_tol=1e-15
            ):
                raise InstrumentConfigurationError(
                    f'{label}归零量程设置改变了限流：'
                    f'{current_limit:g} A → {applied_limit:g} A'
                )
        # Explicit zero source delay removes only the additional programmable
        # delay; the 2450's range-dependent physical settling remains active.
        instrument.write(':SOUR:VOLT:DEL 0')
        instrument.write(':SOUR:VOLT:DEL:AUTO OFF')
        instrument.write(
            f':TRAC:MAKE "{buffer_name}", {max(100, int(chunk_changes) + 2)}'
        )
        buffer_created = True

        previous = current
        full_levels = []
        residual_target = None
        for level in levels:
            if _is_full_step(previous, level, step_value):
                full_levels.append(level)
                previous = level
            else:
                residual_target = level

        # A two-point 2450 source sweep (current value plus one change) is not
        # reliable on all firmware revisions: INIT may finish while the source
        # remains at the start value.  A single exact change is therefore sent
        # directly; multi-step final zeroing still uses the internal sweep.
        if len(full_levels) == 1:
            endpoint = full_levels[0]
            instrument.write(f':SOUR:VOLT {endpoint:.12g}')
            instrument.query('*OPC?')
            actual = required_float_query(
                instrument, ':SOUR:VOLT?', f'{label}单步归零回读'
            )
            if not math.isclose(
                actual, endpoint, rel_tol=1e-8,
                abs_tol=max(1e-9, step_value * 1e-6),
            ):
                raise InstrumentConfigurationError(
                    f'{label}单步归零未到达终点：'
                    f'请求 {endpoint:g} V，回读 {actual:g} V'
                )
            full_levels = []

        previous = current
        for offset in range(0, len(full_levels), int(chunk_changes)):
            if force_event is not None and force_event.is_set():
                raise RuntimeError('强制停止')
            chunk = full_levels[offset:offset + int(chunk_changes)]
            endpoint = chunk[-1]
            point_count = len(chunk) + 1
            instrument.write(f':TRAC:CLEAR "{buffer_name}"')
            instrument.write(
                ':SOUR:SWE:VOLT:LIN '
                f'{previous:.12g},{endpoint:.12g},{point_count},'
                f'0,1,FIXED,OFF,OFF,"{buffer_name}"'
            )
            trigger_model = instrument.query(':TRIG:BLOC:LIST?')
            trigger_lines = {
                ' '.join(line.split())
                for line in str(trigger_model).splitlines()
            }
            if not any(
                line.startswith('6) MEASURE_DIGITIZE')
                for line in trigger_lines
            ) or not any(
                line.startswith('8) SOURCE_OUTPUT')
                for line in trigger_lines
            ):
                raise InstrumentConfigurationError(
                    f'{label}内部扫描触发模型与2450预期不一致'
                )
            # Keep output on between chunks; the common shutdown below turns
            # it off only after the final confirmed 0 V.
            instrument.write(':TRIG:BLOC:NOP 8')
            instrument.write(':INIT')
            instrument.write('*WAI')
            actual = required_float_query(
                instrument, ':SOUR:VOLT?', f'{label}内部归零批次回读'
            )
            if not math.isclose(
                actual, endpoint, rel_tol=1e-8, abs_tol=max(1e-9, step_value * 1e-6)
            ):
                raise InstrumentConfigurationError(
                    f'{label}内部归零未到达批次终点：'
                    f'请求 {endpoint:g} V，回读 {actual:g} V'
                )
            previous = endpoint
            report['chunk_count'] += 1

        if residual_target is not None:
            if force_event is not None and force_event.is_set():
                raise RuntimeError('强制停止')
            instrument.write(f':SOUR:VOLT {residual_target:.12g}')
            instrument.query('*OPC?')

        instrument.write(':SOUR:VOLT 0')
        instrument.query('*OPC?')
        zero_readback = required_float_query(
            instrument, ':SOUR:VOLT?', f'{label}零点回读'
        )
        report['zero_readback'] = zero_readback
        if not math.isclose(zero_readback, 0.0, rel_tol=0.0, abs_tol=1e-9):
            raise InstrumentConfigurationError(
                f'{label}归零回读不是0 V：{zero_readback:g} V'
            )
        instrument.write(':OUTP OFF')
        report['output_off_confirmed'] = (
            int(float(instrument.query(':OUTP?').strip())) == 0
        )
        if not report['output_off_confirmed']:
            raise InstrumentConfigurationError(f'{label}输出回读仍为开启')
        instrument.write(f':TRAC:DEL "{buffer_name}"')
        buffer_created = False
        errors = drain_scpi_errors(instrument)
        if errors:
            raise InstrumentConfigurationError(
                f'{label}归零错误队列: ' + ' | '.join(errors)
            )
    except Exception as exc:
        report['status'] = 'emergency_off'
        report['errors'].append(str(exc))
        confirmed, failures = reliable_output_off(instrument, label)
        report['output_off_confirmed'] = confirmed
        report['errors'].extend(failures)
        try:
            report['zero_readback'] = required_float_query(
                instrument, ':SOUR:VOLT?', f'{label}紧急归零回读'
            )
        except Exception as readback_exc:
            report['errors'].append(str(readback_exc))
        zero_confirmed = (
            report['zero_readback'] is not None
            and math.isclose(
                report['zero_readback'], 0.0,
                rel_tol=0.0, abs_tol=1e-9,
            )
        )
        if not confirmed or not zero_confirmed:
            report['status'] = 'unconfirmed'
    finally:
        if buffer_created:
            try:
                instrument.write(f':TRAC:DEL "{buffer_name}"')
            except Exception as exc:
                report['errors'].append(f'临时归零缓冲区清理失败: {exc}')
        if timeout_changed:
            try:
                instrument.timeout = original_timeout
            except Exception as exc:
                report['errors'].append(f'恢复VISA超时设置失败: {exc}')
        report['elapsed_s'] = time.monotonic() - started
    return report


def result_metadata_path(data_path):
    data_path = Path(data_path)
    return data_path.parent / 'metadata' / f'{data_path.stem}_meta.json'


def allocate_unique_path(folder, filename):
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    requested = folder_path / filename
    stem = requested.stem
    suffix = requested.suffix
    candidate = requested
    index = 1
    while True:
        reservation = candidate.with_name(f'.{candidate.name}.reserve')
        occupied = (
            candidate.exists()
            or reservation.exists()
            or result_metadata_path(candidate).exists()
            # Retain collision protection for metadata written by older
            # versions next to the data file.
            or candidate.with_name(candidate.stem + '_meta.json').exists()
            or candidate.with_name(
                candidate.stem + '_partial' + candidate.suffix
            ).exists()
        )
        if not occupied:
            try:
                descriptor = os.open(
                    reservation,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                occupied = True
            else:
                os.close(descriptor)
                return candidate
        candidate = folder_path / f'{stem}_{index:03d}{suffix}'
        index += 1


def release_path_reservation(path):
    path = Path(path)
    reservation = path.with_name(f'.{path.name}.reserve')
    try:
        reservation.unlink()
    except FileNotFoundError:
        pass


@contextlib.contextmanager
def atomic_text_writer(final_path, encoding='utf-8'):
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode='w',
        encoding=encoding,
        newline='',
        prefix=f'.{final_path.name}.',
        suffix='.tmp',
        dir=final_path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temp_path, final_path)
        reservation = final_path.with_name(f'.{final_path.name}.reserve')
        try:
            reservation.unlink()
        except FileNotFoundError:
            pass
    except Exception:
        try:
            handle.close()
        finally:
            raise


def write_result_metadata(
    data_path,
    *,
    status,
    point_count,
    error=None,
    extra=None,
    stopped_at_local=None,
):
    data_path = Path(data_path)
    if status not in ('complete', 'partial', 'error'):
        raise ValueError(f'未知测量结果状态: {status}')
    metadata_path = result_metadata_path(data_path)
    payload = {
        'status': status,
        'point_count': int(point_count),
        'stopped_at_local': (
            stopped_at_local
            or time.strftime('%Y-%m-%d %H:%M:%S')
        ),
        'error': None if error is None else str(error),
        'error_type': (
            None
            if error is None
            else type(error).__name__
            if isinstance(error, BaseException)
            else 'MeasurementError'
        ),
    }
    if extra:
        payload.update(extra)
    with atomic_text_writer(metadata_path) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return metadata_path


class BaseMeasurement:
    def __init__(self, stop_event, force_stop_event, update_queue=None, alarm_queue=None):
        self.stop_event = stop_event
        self.force_stop_event = force_stop_event
        self.update_queue = update_queue
        self.alarm_queue = alarm_queue

    def connect_instrument(self, address, timeout_ms=5000):
        try:
            rm = pyvisa.ResourceManager()
            inst = rm.open_resource(address)
            inst.timeout = timeout_ms
            idn = validate_2450_idn(inst.query('*IDN?'))
            if self.alarm_queue is not None:
                self.alarm_queue.put(f"仪器已连接: {idn}")
            return inst
        except Exception as exc:
            if self.alarm_queue is not None:
                self.alarm_queue.put(f"连接失败: {repr(exc)}")
            raise

    def _interruptible_sleep(self, duration_s, stage_msg=None, update_type='stage'):
        if duration_s <= 0:
            return True

        steps = int(duration_s / 0.1)
        for i in range(steps):
            if self.stop_event.is_set() or self.force_stop_event.is_set():
                return False
            time.sleep(0.1)
            if stage_msg and self.update_queue is not None and i % 10 == 0:
                remain = duration_s - (i * 0.1)
                self.update_queue.put((update_type, f"{stage_msg} ({remain:.1f}s)"))

        remain = duration_s - steps * 0.1
        if remain > 0:
            time.sleep(remain)
        return not (self.stop_event.is_set() or self.force_stop_event.is_set())

    def _ramp_voltage(self, inst, target_v, step_abs, step_delay_s, is_gate=False, is_zeroing=False,
                      read_query=':READ?', update_prefix=None):
        try:
            current_v = required_float_query(inst, ':SOUR:VOLT?', '源电压回读')
        except MeasurementReadError:
            raise

        if abs(target_v - current_v) < 1e-9:
            return True

        step_abs = abs(step_abs)
        if step_abs == 0:
            step_abs = 0.001

        direction = 1 if target_v > current_v else -1
        steps = int(round(abs(target_v - current_v) / step_abs))
        if steps == 0:
            steps = 1

        for i in range(1, steps + 1):
            if self.force_stop_event.is_set():
                return False
            if not is_zeroing and self.stop_event.is_set():
                return False

            v = current_v + direction * i * step_abs
            if direction == 1 and v > target_v:
                v = target_v
            elif direction == -1 and v < target_v:
                v = target_v

            inst.write(f':SOUR:VOLT {v}')
            time.sleep(step_delay_s)

            if self.update_queue is not None:
                try:
                    reading = required_float_query(inst, read_query, '爬坡电流读数')
                except MeasurementReadError:
                    raise

                if update_prefix:
                    self.update_queue.put((update_prefix, v, reading))
                elif is_gate:
                    self.update_queue.put(('ramp_g', v, reading))
                else:
                    self.update_queue.put(('ramp_b', v, reading))

            if v == target_v:
                break

        return True
