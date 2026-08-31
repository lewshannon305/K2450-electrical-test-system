import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import (
    QApplication, QGroupBox, QLabel, QLineEdit, QPushButton,
)
from matplotlib.figure import Figure

from core.plotting import (
    MODULE_NAMES,
    PlotManager,
    PlotPreviewDialog,
    PlotSettingsDialog,
    _apply_figure_style,
    _engineering_current,
    _engineering_voltage,
    _mapping_norm,
    default_plot_settings,
    merge_plot_settings,
    render_result,
    select_preview_paths,
)


class PlotDialogUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _show_dialog(self, width=1200, height=700):
        dialog = PlotSettingsDialog(default_plot_settings())
        dialog.resize(width, height)
        dialog.show()
        self.app.processEvents()
        self.addCleanup(dialog.close)
        return dialog

    def test_navigation_uses_ready_persistent_preview_pages(self):
        dialog = self._show_dialog()
        self.assertEqual(len(dialog._preview_pages), len(MODULE_NAMES) + 1)
        for row in range(dialog.nav.count()):
            dialog.nav.setCurrentRow(row)
            self.app.processEvents()
            key = None if row == 0 else list(MODULE_NAMES)[row - 1]
            self.assertFalse(dialog._preview_timer.isActive())
            self.assertEqual(
                dialog.preview_stack.currentIndex(),
                dialog._preview_page_indices[key],
            )
            self.assertGreater(
                len(dialog._preview_pages[key].current_canvas().ci.items), 0
            )

    def test_module_values_and_scroll_positions_do_not_leak(self):
        dialog = self._show_dialog()
        dialog.nav.setCurrentRow(2)
        self.app.processEvents()
        dialog.edit_title.setText('IV local title')
        iv_scroll = dialog.module_scroll.verticalScrollBar()
        iv_scroll.setValue(min(120, iv_scroll.maximum()))
        expected_scroll = iv_scroll.value()

        dialog.nav.setCurrentRow(4)
        self.app.processEvents()
        dialog.edit_title.setText('Mapping local title')
        dialog.nav.setCurrentRow(2)
        self.app.processEvents()

        self.assertEqual(dialog.edit_title.text(), 'IV local title')
        self.assertEqual(iv_scroll.value(), expected_scroll)

    def test_dialog_remains_usable_at_supported_sizes(self):
        dialog = self._show_dialog()
        for width, height in ((1200, 700), (1366, 768), (1536, 912)):
            dialog.resize(width, height)
            self.app.processEvents()
            self.assertGreaterEqual(dialog.settings_group.width(), 360)
            self.assertGreaterEqual(
                dialog.content_splitter.widget(1).width(), 480
            )
            for button in dialog.findChildren(QPushButton):
                if button in (
                    dialog.btn_apply, dialog.btn_save_profile,
                    dialog.btn_load_profile, dialog.btn_set_default,
                    dialog.btn_restore, dialog.btn_cancel, dialog.btn_ok,
                ):
                    self.assertLessEqual(
                        button.geometry().right(), dialog.width()
                    )

    def test_only_group_titles_are_bold(self):
        dialog = self._show_dialog()
        self.assertIn('QGroupBox::title', dialog.styleSheet())
        self.assertIn('font-weight: 600', dialog.styleSheet())
        for widget_type in (QLabel, QLineEdit, QPushButton):
            for widget in dialog.findChildren(widget_type):
                self.assertFalse(widget.font().bold(), widget.objectName())
        for group in dialog.findChildren(QGroupBox):
            self.assertFalse(group.font().bold(), group.title())

    def test_preview_canvas_tracks_export_aspect_ratio(self):
        dialog = self._show_dialog()
        dialog.nav.setCurrentRow(4)
        self.app.processEvents()
        for width, height in ((120.0, 95.0), (183.0, 35.0), (40.0, 170.0)):
            dialog.spin_width_mm.setValue(width)
            dialog.spin_height_mm.setValue(height)
            self.app.processEvents()
            ratio = dialog.preview_stack.width() / dialog.preview_stack.height()
            expected = width / height
            self.assertLessEqual(abs(ratio - expected) / expected, 0.02)
        wrapped = dialog._wrap_preview_text(
            'A very long preview title ' * 8, 10
        )
        self.assertIn('<br>', wrapped)

    def test_mapping_preview_contains_separate_colorbar_panel(self):
        dialog = self._show_dialog()
        dialog.nav.setCurrentRow(4)
        self.app.processEvents()
        canvas = dialog._preview_pages['mapping_scan'].current_canvas()
        plot_items = [
            item for item in canvas.ci.items
            if type(item).__name__ == 'PlotItem'
        ]
        self.assertEqual(len(plot_items), 2)
        self.assertTrue(all(
            any(type(item).__name__ == 'ImageItem' for item in plot.items)
            for plot in plot_items
        ))

    def test_png_preview_refits_wide_tall_and_square_images(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for name, size in (
                ('wide.png', (1200, 300)),
                ('tall.png', (300, 1200)),
                ('square.png', (600, 600)),
            ):
                path = Path(folder) / name
                image = QImage(*size, QImage.Format.Format_RGB32)
                image.fill(QColor('white'))
                self.assertTrue(image.save(str(path)))
                paths.append(str(path))
            invalid = Path(folder) / 'invalid.png'
            invalid.write_text('not an image', encoding='utf-8')
            paths.append(str(invalid))

            dialog = PlotPreviewDialog(paths)
            dialog.resize(900, 650)
            dialog.show()
            self.app.processEvents()
            self.addCleanup(dialog.close)
            self.assertEqual(dialog.tabs.count(), 3)
            for index in range(dialog.tabs.count()):
                dialog.tabs.setCurrentIndex(index)
                self.app.processEvents()
                label = dialog.tabs.currentWidget()
                self.assertIsInstance(label, QLabel)
                pixmap = label.pixmap()
                self.assertIsNotNone(pixmap)
                self.assertLessEqual(pixmap.width(), label.contentsRect().width())
                self.assertLessEqual(pixmap.height(), label.contentsRect().height())


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

    def test_mapping_norm_centers_signed_data_and_spans_positive_data(self):
        signed = _mapping_norm(np.array([-3.0, -1.0, 2.0]), signed=True)
        self.assertIsInstance(signed, type(_mapping_norm([-1.0, 1.0], True)))
        self.assertEqual(signed.vcenter, 0.0)
        positive = _mapping_norm(np.array([1.0, 2.0, 3.0]), signed=False)
        self.assertLess(positive.vmin, positive.vmax)
        self.assertGreaterEqual(positive.vmin, 1.0)

    def test_export_style_uses_equal_four_sided_frame(self):
        settings = default_plot_settings()
        module = settings['modules']['iv_curve']
        settings['_module_settings'] = module
        figure = Figure()
        axis = figure.subplots()
        axis.plot([0, 1], [0, 1], label='trace')
        axis.legend()
        _apply_figure_style(figure, settings)
        widths = []
        for spine in axis.spines.values():
            self.assertTrue(spine.get_visible())
            widths.append(spine.get_linewidth())
        self.assertEqual(widths, [module['tick_width']] * 4)
        self.assertFalse(axis.get_legend().get_frame_on())

    def test_partial_configuration_gets_complete_plot_defaults(self):
        settings = merge_plot_settings({'auto_plot': False})
        self.assertFalse(settings['auto_plot'])
        self.assertIn('mapping_scan', settings['modules'])
        self.assertTrue(settings['modules']['iv_curve']['enabled'])
        self.assertEqual(settings['plot_schema_version'], 3)
        self.assertEqual(
            settings['modules']['iv_curve']['tick_size'], 6
        )
        self.assertEqual(
            settings['modules']['iv_curve']['line_width'], 1.0
        )
        self.assertFalse(
            settings['modules']['mapping_scan']['mapping_full_iv']
        )
        self.assertEqual(
            settings['modules']['iv_curve']['legend_location'], 'upper left'
        )
        self.assertEqual(
            settings['modules']['isd_vg_setvsd']['isd_mode'], 'stacked'
        )
        self.assertTrue(all(
            module['dpi'] == 450 for module in settings['modules'].values()
        ))
        self.assertTrue(all(
            'top_spine' not in module and 'right_spine' not in module
            for module in settings['modules'].values()
        ))
        self.assertNotIn(
            'iv_file_mode', settings['modules']['break_junction']
        )
        self.assertNotIn(
            'mapping_full_iv', settings['modules']['iv_curve']
        )
        self.assertNotIn(
            'it_bin_method', settings['modules']['mapping_scan']
        )

    def test_legacy_plot_schema_is_not_migrated(self):
        settings = merge_plot_settings({
            'plot_schema_version': 2,
            'auto_plot': False,
            'modules': {
                'iv_curve': {
                    'legend_location': 'lower right',
                    'top_spine': False,
                },
            },
        })
        self.assertTrue(settings['auto_plot'])
        self.assertEqual(settings['plot_schema_version'], 3)
        self.assertEqual(
            settings['modules']['iv_curve']['legend_location'], 'upper left'
        )
        self.assertNotIn('top_spine', settings['modules']['iv_curve'])


class PlotRenderingTests(unittest.TestCase):
    def test_break_junction_result_mirrors_into_root_figures(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data_folder = root / 'Break'
            data_folder.mkdir()
            data_path = data_folder / 'break.txt'
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
                'data_root': str(root),
                'save_prefix': 'break',
            }
            paths = render_result(result, default_plot_settings())
            self.assertEqual(len(paths), 3)
            self.assertTrue(all(Path(path).exists() for path in paths))
            self.assertEqual(
                {Path(path).suffix for path in paths},
                {'.svg', '.pdf', '.png'},
            )
            self.assertTrue(all(
                os.path.samefile(
                    Path(path).parent,
                    root / 'figures' / 'Break',
                )
                for path in paths
            ))
            self.assertEqual({Path(path).stem for path in paths}, {'break'})

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

    def test_custom_figure_root_keeps_module_subfolder_and_redraw_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'data'
            custom = Path(folder) / 'custom-plots'
            data_folder = root / 'IV'
            data_folder.mkdir(parents=True)
            data_path = data_folder / 'IV.txt'
            np.savetxt(
                data_path,
                np.array([[-0.1, -1e-7], [0.1, 1e-7]]),
                header='Voltage Current',
            )
            settings = default_plot_settings()
            settings['output_mode'] = 'custom'
            settings['output_folder'] = str(custom)
            settings['modules']['iv_curve']['formats'] = ['svg']
            result = {
                'module_id': 'iv_curve',
                'status': 'complete',
                'data_files': [str(data_path)],
                'data_root': str(root),
                'save_prefix': 'IV',
            }
            first = render_result(result, settings)
            second = render_result(result, settings)
            expected = custom / 'IV' / 'IV.svg'
            self.assertEqual(len(first), 1)
            self.assertEqual(len(second), 1)
            self.assertTrue(os.path.samefile(first[0], expected))
            self.assertTrue(os.path.samefile(second[0], expected))
            self.assertTrue(expected.exists())
            self.assertFalse((custom / 'IV' / 'IV_backup001.svg').exists())

    def test_isdvg_aggregate_uses_save_prefix_and_backup_token(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data_folder = root / 'Isd_Vg'
            data_folder.mkdir()
            paths = []
            for segment in (1, 2):
                path = data_folder / (
                    f'isdvg_accept_seg{segment}_backup001.txt'
                )
                np.savetxt(
                    path,
                    np.array([
                        [0.05, 0.0, 5e-8, 1e-12],
                        [0.05, 0.05, 5e-8, 1e-12],
                    ]),
                    header='Vsd Vg Isd Ig',
                )
                paths.append(str(path))
            settings = default_plot_settings()
            settings['modules']['isd_vg_setvsd']['formats'] = ['svg']
            outputs = render_result({
                'module_id': 'isd_vg_setvsd',
                'status': 'complete',
                'data_files': paths,
                'data_root': str(root),
                'save_prefix': 'isdvg_accept',
            }, settings)
            expected = (root / 'figures' / 'Isd_Vg'
                        / 'isdvg_accept_backup001.svg')
            self.assertEqual(len(outputs), 1)
            self.assertTrue(os.path.samefile(outputs[0], expected))

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
                    'It_Vg=1V_Vb=0.1V',
                    'It_Vg=2V_Vb=0.1V',
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
                if Path(path).parent.name == 'full'
            ]
            self.assertEqual(len(full_iv), 6)
            self.assertFalse(any('9.000' in path.name for path in full_iv))

    def test_mapping_default_draws_only_three_maps_and_ignores_pos_files(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for gate in (0.0, 0.1):
                for phase in ('pos', 'full'):
                    path = Path(folder) / f'Mapping_Vg={gate:.3f}V_{phase}.txt'
                    values = np.column_stack([
                        np.full(5, gate),
                        np.linspace(-0.1, 0.1, 5),
                        np.linspace(-1e-9, 1e-9, 5),
                        np.zeros(5),
                        np.zeros(5),
                    ])
                    np.savetxt(path, values, header='Vg Vsd Isd Ig Age')
                    paths.append(str(path))
            settings = default_plot_settings()
            settings['modules']['mapping_scan']['formats'] = ['png']
            outputs = render_result({
                'module_id': 'mapping_scan',
                'status': 'complete',
                'data_files': paths,
                'save_prefix': 'Mapping',
            }, settings)
            self.assertEqual(len(outputs), 3)
            self.assertTrue(all('mapping' in Path(path).stem for path in outputs))
            self.assertFalse(any(
                Path(path).parent.name in {'pos', 'full'} for path in outputs
            ))

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
                if Path(path).parent.name == 'full'
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
