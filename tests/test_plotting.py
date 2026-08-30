import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.plotting import (
    PlotManager,
    _engineering_current,
    _engineering_voltage,
    default_plot_settings,
    merge_plot_settings,
    render_result,
    select_preview_paths,
)


class PlotConfigurationTests(unittest.TestCase):
    def test_new_result_clears_stale_plot_paths_when_auto_plot_is_off(self):
        settings = default_plot_settings()
        settings['auto_plot'] = False
        manager = PlotManager(settings)
        manager.latest_plot_paths = ['old_run.png']

        manager.handle_result({
            'run_id': 'new_run',
            'module_id': 'break_junction',
            'status': 'complete',
            'data_files': ['new_run.txt'],
        })

        self.assertEqual(manager.latest_result['run_id'], 'new_run')
        self.assertEqual(manager.latest_plot_paths, [])

    def test_small_voltages_use_explicit_millivolts(self):
        values, unit, factor = _engineering_voltage([-.01, 0, .01])
        self.assertEqual(unit, 'mV')
        self.assertEqual(factor, 1000.0)
        np.testing.assert_allclose(values, [-10, 0, 10])

    def test_mapping_preview_selects_only_differential_conductance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = []
            for name in (
                'raw_current_mapping.png',
                'smoothed_current_mapping.png',
                'differential_conductance_mapping.png',
                'sample_full.png',
            ):
                path = root / name
                path.write_bytes(b'png')
                paths.append(str(path))
            selected = select_preview_paths(paths, 'mapping_scan')
            self.assertEqual(
                [Path(path).name for path in selected],
                ['differential_conductance_mapping.png'],
            )

    def test_current_uses_explicit_engineering_units(self):
        values, unit, factor = _engineering_current(
            np.array([-2e-9, 0.0, 2e-9])
        )
        self.assertEqual(unit, 'nA')
        self.assertEqual(factor, 1e9)
        np.testing.assert_allclose(values, [-2.0, 0.0, 2.0])

    def test_partial_configuration_gets_complete_plot_defaults(self):
        settings = merge_plot_settings({'auto_plot': False})
        self.assertFalse(settings['auto_plot'])
        self.assertIn('mapping_scan', settings['modules'])
        self.assertTrue(settings['modules']['iv_curve']['enabled'])
        self.assertEqual(settings['plot_schema_version'], 2)
        self.assertEqual(
            settings['modules']['iv_curve']['tick_size'], 7
        )
        self.assertEqual(
            settings['modules']['iv_curve']['line_width'], 1.0
        )
        self.assertFalse(
            settings['modules']['mapping_scan']['mapping_full_iv']
        )


class PlotRenderingTests(unittest.TestCase):
    def test_break_junction_result_renders_into_plots_subfolder(self):
        with tempfile.TemporaryDirectory() as folder:
            data_path = Path(folder) / 'break.txt'
            np.savetxt(
                data_path,
                np.array([
                    [0.01, 1e-9, 0.001],
                    [0.02, 2e-9, 0.002],
                    [0.03, 3e-9, 0.003],
                ]),
                header='Voltage (V)\tCurrent (A)\tConductance (G0)',
            )
            result = {
                'run_id': 'testbreak',
                'module_id': 'break_junction',
                'status': 'complete',
                'data_files': [str(data_path)],
            }
            paths = render_result(result, default_plot_settings())
            self.assertEqual(len(paths), 3)
            self.assertTrue(all(Path(path).exists() for path in paths))
            self.assertEqual(
                {Path(path).suffix for path in paths},
                {'.svg', '.pdf', '.png'},
            )
            self.assertTrue(all(Path(path).parent.name == 'plots' for path in paths))

    def test_partial_result_is_marked_in_output_filename(self):
        with tempfile.TemporaryDirectory() as folder:
            data_path = Path(folder) / 'break_partial.txt'
            np.savetxt(
                data_path,
                np.array([[0.01, 1e-9, 0.001], [0.02, 2e-9, 0.002]]),
                header='Voltage\tCurrent\tConductance',
            )
            result = {
                'run_id': 'partialrun',
                'module_id': 'break_junction',
                'status': 'partial',
                'data_files': [str(data_path)],
            }
            paths = render_result(result, default_plot_settings())
            self.assertIn('partial', Path(paths[0]).stem)
            svg = next(Path(path) for path in paths if path.endswith('.svg'))
            self.assertIn('PARTIAL', svg.read_text(encoding='utf-8'))

    def test_it_result_renders_every_data_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data_paths = []
            for index, current in enumerate((1e-6, 10e-6), 1):
                path = root / f'It_Vg={index}V_Vb=0.1V.txt'
                np.savetxt(
                    path,
                    np.array([
                        [0.0, 0.1, current, current * 1.1],
                        [1.0, 0.1, current * 2, current * 2.1],
                    ]),
                    header='Time Bias Raw Filtered',
                )
                data_paths.append(str(path))
            settings = default_plot_settings()
            settings['modules']['it_step_setgate']['formats'] = ['svg']

            outputs = render_result({
                'run_id': 'itmultifile',
                'module_id': 'it_step_setgate',
                'status': 'complete',
                'data_files': data_paths,
            }, settings)

            self.assertEqual(len(outputs), 2)
            self.assertEqual(
                {Path(path).stem for path in outputs},
                {
                    'it_step_setgate_itmultifile_001',
                    'it_step_setgate_itmultifile_002',
                },
            )
            svg_text = [
                Path(path).read_text(encoding='utf-8') for path in outputs
            ]
            self.assertNotEqual(svg_text[0], svg_text[1])

    def test_mapping_uses_only_files_in_result_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            full = Path(folder) / 'full'
            full.mkdir()
            paths = []
            for gate in (0.0, 0.1):
                path = full / f'Mapping_Vg={gate:.3f}V_full.txt'
                values = np.column_stack([
                    np.full(5, gate),
                    np.linspace(-0.1, 0.1, 5),
                    np.linspace(-1e-9, 1e-9, 5) + gate * 1e-9,
                    np.zeros(5),
                    np.zeros(5),
                ])
                np.savetxt(
                    path,
                    values,
                    header='Vg\tBias_voltage\tBias_current\tGate_current\tAge',
                )
                paths.append(str(path))
            unrelated = full / 'Mapping_Vg=9.000V_full.txt'
            np.savetxt(
                unrelated,
                np.ones((5, 5)),
                header='Vg\tBias_voltage\tBias_current\tGate_current\tAge',
            )
            result = {
                'run_id': 'testmapping',
                'module_id': 'mapping_scan',
                'status': 'complete',
                'data_files': paths,
            }
            settings = default_plot_settings()
            settings['modules']['mapping_scan']['mapping_full_iv'] = True
            outputs = render_result(result, settings)
            self.assertEqual(len(outputs), 15)
            self.assertTrue(all(Path(path).exists() for path in outputs))
            full_iv = [
                Path(path) for path in outputs
                if Path(path).parent.name == 'full_iv'
            ]
            self.assertEqual(len(full_iv), 6)
            self.assertFalse(any('9.000' in path.name for path in full_iv))

    def test_mapping_does_not_treat_prefix_full_as_full_scan_suffix(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for gate in (0.0, 0.1):
                for suffix, points in (('full', 5), ('pos', 3)):
                    path = Path(folder) / (
                        f'run_full_range_Vg={gate:.3f}V_{suffix}.txt'
                    )
                    values = np.column_stack([
                        np.full(points, gate),
                        np.linspace(-0.1, 0.1, points),
                        np.linspace(-1e-9, 1e-9, points),
                        np.zeros(points),
                        np.zeros(points),
                    ])
                    np.savetxt(path, values, header='Vg Vsd Isd Ig Age')
                    paths.append(str(path))
            settings = default_plot_settings()
            module = settings['modules']['mapping_scan']
            module['formats'] = ['png']
            module['mapping_full_iv'] = True
            result = {
                'run_id': 'mappingprefix',
                'module_id': 'mapping_scan',
                'status': 'complete',
                'data_files': paths,
            }
            outputs = render_result(result, settings)
            self.assertEqual(len(outputs), 5)
            full_iv_names = {
                Path(path).name for path in outputs
                if Path(path).parent.name == 'full_iv'
            }
            self.assertEqual(len(full_iv_names), 2)
            self.assertTrue(all('_full' in name for name in full_iv_names))

    def test_break_uses_readable_engineering_current_unit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'break.txt'
            np.savetxt(
                path,
                np.array([[0.01, 10e-9, 0.1], [0.1, 100e-9, 0.2]]),
                header='Voltage Current Conductance',
            )
            settings = default_plot_settings()
            settings['modules']['break_junction']['formats'] = ['svg']
            outputs = render_result({
                'run_id': 'breakunit', 'module_id': 'break_junction',
                'status': 'complete', 'data_files': [str(path)],
            }, settings)
            svg = Path(outputs[0]).read_text(encoding='utf-8')
            self.assertIn('Current (nA)', svg)

    def test_arbitrary_gate_four_columns_render_two_panels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'arbitrary_gate.txt'
            np.savetxt(
                path,
                np.array([
                    [0.0, 0.1, 0.05, 1e-6],
                    [1.0, -0.1, 0.05, 2e-6],
                ]),
                header='Time GateVoltage BiasVoltage BiasCurrent',
            )
            settings = default_plot_settings()
            settings['modules']['arbitrary_gate']['formats'] = ['svg', 'pdf', 'png']
            outputs = render_result({
                'run_id': 'gatecurrents',
                'module_id': 'arbitrary_gate',
                'status': 'complete',
                'data_files': [str(path)],
            }, settings)
            self.assertEqual(len(outputs), 3)
            svg_path = next(Path(item) for item in outputs if str(item).endswith('.svg'))
            svg = svg_path.read_text(encoding='utf-8')
            self.assertIn('Gate voltage', svg)
            self.assertIn('I', svg)
            self.assertIn('sd', svg)

    def test_iv_legend_uses_all_semantic_segment_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for ordinal, start, stop in (
                ('1st', 0.0, -0.1), ('2nd', -0.1, 0.1),
                ('3rd', 0.1, -0.1), ('4th', -0.1, 0.0),
            ):
                path = Path(folder) / f'IV_hysteresis_{ordinal}_cyc1.txt'
                voltage = np.linspace(start, stop, 5)
                np.savetxt(
                    path, np.column_stack([voltage, voltage * 1e-6]),
                    header='Voltage Current',
                )
                paths.append(str(path))
            settings = default_plot_settings()
            settings['modules']['iv_curve']['formats'] = ['svg']
            outputs = render_result({
                'run_id': 'ivlegend', 'module_id': 'iv_curve',
                'status': 'complete', 'data_files': paths,
            }, settings)
            svg = Path(outputs[0]).read_text(encoding='utf-8')
            for segment in range(1, 5):
                self.assertIn(f'Segment {segment}:', svg)

    def test_iv_custom_sequence_renders_in_entered_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'IV_custom_6points_cyc1.txt'
            voltage = np.array([-1.0, 0.2, -0.4, 0.8, 0.8, 0.0])
            np.savetxt(
                path,
                np.column_stack([voltage, voltage * 1e-6]),
                header='Voltage Current',
            )
            settings = default_plot_settings()
            settings['modules']['iv_curve']['formats'] = ['svg']
            outputs = render_result({
                'run_id': 'ivcustom',
                'module_id': 'iv_curve',
                'status': 'complete',
                'data_files': [str(path)],
            }, settings)
            self.assertEqual(len(outputs), 1)
            self.assertTrue(Path(outputs[0]).exists())


if __name__ == '__main__':
    unittest.main()
