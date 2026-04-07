import os
import csv
import tempfile
import unittest
from typing import Any

# Project imports
import cfg_csv_utils

class TestCFGCSVUtils(unittest.TestCase):
    def setUp(self):
        # Create a temporary file name for our CSV tests.
        temp = tempfile.NamedTemporaryFile(mode="w+", delete=False, newline="")
        self.filename = temp.name
        temp.close()

    def tearDown(self):
        # Clean up the temporary file after each test.
        os.remove(self.filename)

    def test_save_and_load(self):
        # Define a sample CFG rules dictionary.
        sample_cfg: dict[Any, Any] = {
            "S": [["A", "B"], ["B", "A"]],
            "A": [["a", "A"], ["a"]],
            "B": [["b", "B"], ["b"]]
        }
        # Save the CFG to CSV.
        cfg_csv_utils.save_cfg_to_csv(sample_cfg, self.filename)
        # Load the CFG from the CSV file.
        loaded_cfg = cfg_csv_utils.load_from_csv(self.filename)
        
        # Since keys are converted to strings during CSV saving, we compare stringified keys.
        expected_keys = set(str(k) for k in sample_cfg.keys())
        self.assertEqual(set(loaded_cfg.keys()), expected_keys)
        
        # For each key, compare the set of productions (order independent).
        for key in loaded_cfg:
            expected_prods = {tuple(prod) for prod in sample_cfg.get(key, [])}
            loaded_prods = {tuple(prod) for prod in loaded_cfg[key]}
            self.assertEqual(expected_prods, loaded_prods)

if __name__ == '__main__':
    unittest.main()
