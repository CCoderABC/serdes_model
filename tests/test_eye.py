from __future__ import annotations

import unittest

import numpy as np

from serdes_model.eye import (
    eye_diagram_from_symbol_pulse,
    main_cursor_index,
    pam4_symbol_sequence,
)


class EyeDiagramTests(unittest.TestCase):
    def test_main_cursor_uses_center_of_flat_top(self) -> None:
        pulse = np.asarray([0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
        self.assertEqual(main_cursor_index(pulse), 2)

    def test_ideal_zoh_eye_has_exact_pam4_center_levels(self) -> None:
        samples_per_ui = 8
        unit_voltage_v = 0.15
        pulse = np.zeros(33)
        pulse[8:16] = unit_voltage_v
        symbols = pam4_symbol_sequence(256, seed=7)
        eye = eye_diagram_from_symbol_pulse(
            symbols,
            pulse,
            sampling_index=12,
            samples_per_ui=samples_per_ui,
            trace_count=80,
            guard_symbols=8,
        )
        np.testing.assert_allclose(
            eye.center_samples_v,
            eye.transmitted_symbols * unit_voltage_v,
            atol=1e-13,
        )
        self.assertEqual(eye.traces_v.shape, (80, 129))

    def test_same_sequence_produces_same_selected_symbol_indices(self) -> None:
        symbols = pam4_symbol_sequence(128, seed=11)
        pulse_a = np.asarray([0.0, 0.15, 0.15, 0.0])
        pulse_b = np.asarray([0.0, 0.03, 0.02, 0.01])
        eye_a = eye_diagram_from_symbol_pulse(
            symbols,
            pulse_a,
            sampling_index=1,
            samples_per_ui=2,
            trace_count=32,
            guard_symbols=4,
        )
        eye_b = eye_diagram_from_symbol_pulse(
            symbols,
            pulse_b,
            sampling_index=1,
            samples_per_ui=2,
            trace_count=32,
            guard_symbols=4,
        )
        np.testing.assert_array_equal(eye_a.symbol_indices, eye_b.symbol_indices)
        np.testing.assert_array_equal(
            eye_a.transmitted_symbols, eye_b.transmitted_symbols
        )


if __name__ == "__main__":
    unittest.main()
