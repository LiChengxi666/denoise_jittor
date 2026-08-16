import os
import sys
import unittest

import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class VMCheckpointCompatibilityIntegrationTest(unittest.TestCase):
    def test_explicit_checkpoint_loads_without_parameter_mismatch(self):
        checkpoint = os.environ.get("VM_COMPAT_CHECKPOINT")
        if not checkpoint:
            self.skipTest("set VM_COMPAT_CHECKPOINT to require checkpoint validation")
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(ROOT, checkpoint)
        self.assertTrue(
            os.path.isfile(checkpoint),
            f"required compatibility checkpoint does not exist: {checkpoint}",
        )

        from src.model.vm import VelocityModule

        with open(os.path.join(ROOT, "configs/model/vm_strong.yaml"), "r") as f:
            config = yaml.safe_load(f)
        with open(os.path.join(ROOT, "configs/transform/vm_strong.yaml"), "r") as f:
            transform = yaml.safe_load(f)
        model = VelocityModule(config, transform)
        model.load(checkpoint)


if __name__ == "__main__":
    unittest.main()
