import asyncio
import base64
import unittest

import cv2
import numpy as np
from fastapi import HTTPException

import remotesam


def encoded_image(width=13, height=7):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, : width // 2] = (20, 80, 140)
    ok, payload = cv2.imencode(".png", image)
    assert ok
    return base64.b64encode(payload).decode("ascii")


class FakeRemoteSAM:
    def referring_seg(self, *, image, sentence, return_prob=False):
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[:, width // 2 :] = 1
        probability = np.full((height, width), 0.15, dtype=np.float32)
        probability[mask > 0] = 0.85
        return (mask, probability) if return_prob else mask

    def semantic_seg(self, *, image, classnames, return_prob=False):
        height, width = image.shape[:2]
        masks = {}
        probabilities = {}
        for index, class_name in enumerate(classnames):
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[index : index + 2, :] = 1
            masks[class_name] = mask
            probabilities[class_name] = mask.astype(np.float32) * 0.8
        return (masks, probabilities) if return_prob else masks


class WrongShapeRemoteSAM(FakeRemoteSAM):
    def referring_seg(self, *, image, sentence, return_prob=False):
        return np.ones((2, 2), dtype=np.uint8)


class RemoteSAMServiceTests(unittest.TestCase):
    def setUp(self):
        self.old_model = remotesam.sam_model
        self.old_state = dict(remotesam.MODEL_STATE)
        self.old_device = remotesam.DEVICE
        remotesam.sam_model = FakeRemoteSAM()
        remotesam.DEVICE = "cpu"
        remotesam.MODEL_STATE.update(ready=True, error=None, checkpoint_sha256="test-checkpoint")

    def tearDown(self):
        remotesam.sam_model = self.old_model
        remotesam.DEVICE = self.old_device
        remotesam.MODEL_STATE.clear()
        remotesam.MODEL_STATE.update(self.old_state)

    def run_predict(self, **payload):
        request = remotesam.PredictRequest(image=encoded_image(), **payload)
        return asyncio.run(remotesam.predict(request))

    def test_referring_segmentation_returns_aligned_auditable_result(self):
        result = self.run_predict(
            task="referring_seg",
            text="the buildings on the right",
            return_probability=True,
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["original_size"], [13, 7])
        self.assertEqual(result["output_size"], [13, 7])
        self.assertIn("probability_map", result)
        self.assertGreater(result["quality"]["foreground_pixels"], 0)
        self.assertEqual(result["model"]["checkpoint_sha256"], "test-checkpoint")
        self.assertEqual(result["quality_mode"], "fast")
        self.assertEqual(result["mode_parameters"]["strategy"], "whole_image_argmax")
        self.assertGreater(result["candidate_stats"]["candidate_count"], 0)

    def test_legacy_request_defaults_to_fast_mode(self):
        result = self.run_predict(task="referring_seg", text="all planes")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["quality_mode"], "fast")
        self.assertNotIn("probability_map", result)

    def test_accurate_mode_returns_fixed_parameters_and_aligned_mask(self):
        result = self.run_predict(
            task="referring_seg",
            text="all planes",
            quality_mode="accurate",
            candidate_min_area_pixels=5,
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["quality_mode"], "accurate")
        self.assertEqual(result["original_size"], [13, 7])
        self.assertEqual(result["output_size"], [13, 7])
        self.assertEqual(result["mode_parameters"]["tile_size"], 1024)
        self.assertEqual(result["mode_parameters"]["overlap"], 256)
        self.assertEqual(result["mode_parameters"]["probability_threshold"], 0.88)
        self.assertNotIn("probability_map", result)

    def test_auto_mode_resolution_is_deterministic(self):
        self.assertEqual(remotesam.resolve_quality_mode("auto", (800, 800)), "fast")
        self.assertEqual(remotesam.resolve_quality_mode("auto", (2048, 2048)), "accurate")

    def test_invalid_mode_returns_error_without_mask(self):
        result = self.run_predict(
            task="referring_seg", text="all planes", quality_mode="turbo"
        )
        self.assertEqual(result["status"], "error")
        self.assertIn("Unsupported quality_mode", result["message"])
        self.assertNotIn("mask", result)

    def test_candidate_statistics_filter_noise_and_handle_empty_mask(self):
        mask = np.zeros((10, 12), dtype=np.uint8)
        mask[1:4, 2:6] = 1
        mask[8, 10] = 1
        result = remotesam.candidate_summary(mask, min_area_pixels=2)
        self.assertEqual(result["raw_candidate_count"], 2)
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["filtered_out_count"], 1)
        self.assertEqual(result["total_area_pixels"], 12)
        self.assertEqual(
            result["candidates"][0]["bbox"],
            {"x": 2, "y": 1, "width": 4, "height": 3},
        )

        empty = remotesam.candidate_summary(np.zeros((3, 4), dtype=np.uint8))
        self.assertEqual(empty["raw_candidate_count"], 0)
        self.assertEqual(empty["candidate_count"], 0)
        self.assertEqual(empty["total_area_pixels"], 0)

    def test_semantic_segmentation_returns_per_class_masks(self):
        result = self.run_predict(
            task="semantic_seg",
            classes=["building", "road"],
            return_probability=True,
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(set(result["masks"]), {"building", "road"})
        self.assertIn("probability_map", result)

    def test_wrong_shape_is_rejected_instead_of_resized(self):
        remotesam.sam_model = WrongShapeRemoteSAM()
        result = self.run_predict(task="referring_seg", text="building")
        self.assertEqual(result["status"], "error")
        self.assertIn("does not match input image", result["message"])

    def test_invalid_image_returns_error_without_mask(self):
        request = remotesam.PredictRequest(task="referring_seg", image="not-base64", text="building")
        result = asyncio.run(remotesam.predict(request))
        self.assertEqual(result["status"], "error")
        self.assertNotIn("mask", result)

    def test_mask_size_is_explicitly_rejected(self):
        request = remotesam.PredictRequest(
            task="referring_seg", image=encoded_image(), text="building", mask_size=256
        )
        with self.assertRaises(HTTPException) as error:
            asyncio.run(remotesam.predict(request))
        self.assertEqual(error.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
