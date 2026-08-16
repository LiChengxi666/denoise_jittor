import csv
import importlib.util
import os
import tempfile
import unittest

import numpy as np
import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_tool(name):
    path = os.path.join(ROOT, "tools", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


grid = load_tool("eval_predict_grid")
blend = load_tool("blend_predictions")


class SprintGridTests(unittest.TestCase):
    def test_sprint_grid_contains_anchor_and_challengers(self):
        settings = grid.build_sprint_settings([1.0, 4.0, 9.0])
        self.assertEqual(len(settings), 18)

        anchors = [
            setting for setting in settings
            if setting["predict_momentum"] == 0.0
        ]
        self.assertEqual(len(anchors), 3)
        self.assertEqual(
            {setting["predict_patch_weight_gamma"] for setting in anchors},
            {1.0, 4.0, 9.0},
        )
        for setting in anchors:
            self.assertEqual(setting["predict_step_size"], 0.8)
            self.assertEqual(setting["predict_num_steps"], 2)
            self.assertEqual(setting["denoise_inner_steps"], 4)
            self.assertEqual(setting["alpha_blend"], 1.0)
            self.assertEqual(setting["predict_step_decay"], "none")

        challengers = [setting for setting in settings if setting not in anchors]
        self.assertEqual(len(challengers), 15)
        self.assertTrue(all(s["predict_momentum"] == 0.6 for s in challengers))
        self.assertTrue(all(s["predict_step_decay"] == "linear" for s in challengers))

    def test_auto_grid_mode_uses_two_stage_policy(self):
        self.assertEqual(grid.resolve_grid_mode("auto", 1), "sprint")
        self.assertEqual(grid.resolve_grid_mode("auto", 2), "screen")
        self.assertEqual(grid.resolve_grid_mode("auto", 10), "screen")
        self.assertEqual(
            grid.resolve_grid_mode("auto", 2, from_screen_csv=True),
            "sprint",
        )
        self.assertEqual(grid.resolve_grid_mode("cartesian", 10), "cartesian")

        screen = grid.build_screen_settings()
        self.assertEqual(len(screen), 1)
        self.assertEqual(screen[0]["predict_momentum"], 0.0)
        self.assertEqual(screen[0]["predict_step_decay"], "none")
        self.assertEqual(screen[0]["predict_patch_weight_gamma"], 1.0)

    def test_score_selection_uses_official_score(self):
        metrics = {"score": 71.87, "cd_score": 60.36, "p2s_score": 83.37}
        self.assertEqual(
            grid.select_score(metrics),
            (71.87, "official_equal", None),
        )

    def test_score_selection_supports_cd_only(self):
        metrics = {"score": None, "cd_score": 60.36, "p2s_score": None}
        self.assertEqual(
            grid.select_score(metrics, allow_cd_only=True),
            (60.36, "cd_only", None),
        )
        self.assertEqual(grid.select_score(metrics)[2], "missing_p2s")

    def test_screen_csv_top_k_and_selected_artifact_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoints = []
            for epoch in range(3):
                path = os.path.join(temp_dir, f"checkpoint_{epoch}.pkl")
                with open(path, "wb") as f:
                    f.write(f"checkpoint-{epoch}".encode("ascii"))
                checkpoints.append(path)

            csv_path = os.path.join(temp_dir, "screen.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["checkpoint", "score", "score_mode", "status"],
                )
                writer.writeheader()
                writer.writerows([
                    {
                        "checkpoint": checkpoints[0],
                        "score": 70.0,
                        "score_mode": "official_equal",
                        "status": "ok",
                    },
                    {
                        "checkpoint": checkpoints[1],
                        "score": 72.0,
                        "score_mode": "official_equal",
                        "status": "ok",
                    },
                    {
                        "checkpoint": checkpoints[2],
                        "score": 71.0,
                        "score_mode": "official_equal",
                        "status": "ok",
                    },
                ])

            selected = grid.load_top_checkpoints(csv_path, top_k=2)
            self.assertEqual(selected, [checkpoints[1], checkpoints[2]])

            selected_checkpoint = os.path.join(temp_dir, "selected", "checkpoint.pkl")
            selected_model = os.path.join(temp_dir, "selected", "model.yaml")
            model_config = {
                "__target__": "VelocityModule",
                "predict_step_size": 0.9,
                "predict_patch_weight_gamma": 9.0,
                "predict_momentum": 0.6,
                "predict_step_decay": "linear",
            }
            grid.export_selected_artifacts(
                best_row={
                    "checkpoint": checkpoints[1],
                    "score_mode": "official_equal",
                },
                model_config=model_config,
                model_path=selected_model,
                checkpoint_path=selected_checkpoint,
            )
            with open(selected_checkpoint, "rb") as f:
                self.assertEqual(f.read(), b"checkpoint-1")
            with open(selected_model, "r") as f:
                self.assertEqual(yaml.safe_load(f), model_config)

    def test_cd_only_result_is_not_exported_by_default(self):
        with self.assertRaisesRegex(ValueError, "CD-only"):
            grid.export_selected_artifacts(
                best_row={"checkpoint": "unused.pkl", "score_mode": "cd_only"},
                model_config={},
                model_path="selected.yaml",
                checkpoint_path="selected.pkl",
            )

    def test_prediction_tasks_use_selected_model_configs(self):
        cases = {
            "predict_vm_sprint_cd.yaml": "vm_sprint_cd_selected",
            "predict_vm_sprint_info_cd.yaml": "vm_sprint_info_cd_selected",
        }
        for task_name, expected_model in cases.items():
            with open(os.path.join(ROOT, "configs", "task", task_name), "r") as f:
                task = yaml.safe_load(f)
            self.assertEqual(task["components"]["model"], expected_model)
            self.assertTrue(task["load_ckpt"].endswith("checkpoint_selected.pkl"))

            model_path = os.path.join(
                ROOT, "configs", "model", f"{expected_model}.yaml"
            )
            with open(model_path, "r") as f:
                model = yaml.safe_load(f)
            for field in (
                "predict_step_size",
                "predict_num_steps",
                "denoise_inner_steps",
                "predict_patch_size",
                "predict_patch_weight_gamma",
                "alpha_blend",
                "predict_momentum",
                "predict_step_decay",
            ):
                self.assertIn(field, model)


class BlendPredictionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.anchor_dir = os.path.join(self.temp_dir.name, "anchor")
        self.challenger_dir = os.path.join(self.temp_dir.name, "challenger")
        self.rel_path = os.path.join("shapenet", "001", "model", "denoised.npy")
        self.anchor = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
        self.challenger = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.float32)
        for base_dir, value in (
            (self.anchor_dir, self.anchor),
            (self.challenger_dir, self.challenger),
        ):
            path = os.path.join(base_dir, self.rel_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, value)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_blend_weights_are_float32_and_finite(self):
        for weight in (0.25, 0.5, 0.75):
            output_dir = os.path.join(self.temp_dir.name, f"blend_{weight}")
            count = blend.blend_predictions(
                self.anchor_dir,
                self.challenger_dir,
                output_dir,
                weight,
            )
            self.assertEqual(count, 1)
            output = np.load(os.path.join(output_dir, self.rel_path))
            expected = (
                (1.0 - weight) * self.anchor + weight * self.challenger
            ).astype(np.float32)
            np.testing.assert_array_equal(output, expected)
            self.assertEqual(output.dtype, np.float32)
            self.assertTrue(np.isfinite(output).all())

    def test_nonempty_output_dir_is_rejected(self):
        output_dir = os.path.join(self.temp_dir.name, "old_blend")
        os.makedirs(output_dir)
        np.save(os.path.join(output_dir, "denoised.npy"), self.anchor)
        with self.assertRaisesRegex(ValueError, "output_dir must be empty"):
            blend.blend_predictions(
                self.anchor_dir,
                self.challenger_dir,
                output_dir,
                0.5,
            )

    def test_input_output_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "output_dir must not equal"):
            blend.blend_predictions(
                self.anchor_dir,
                self.challenger_dir,
                self.anchor_dir,
                0.5,
            )


if __name__ == "__main__":
    unittest.main()
