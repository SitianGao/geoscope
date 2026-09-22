import base64
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import agent_bridge
import server


def decision(status="tool_call", text="", tool_name="referring_expression_segmentation", arguments=None):
    call = None if tool_name is None else SimpleNamespace(
        name=tool_name,
        arguments=arguments or {"image_path": "ignored.png", "prompt": "all buildings"},
    )
    return SimpleNamespace(status=status, text=text, tool_call=call, history=[{"round": 1}])


class AgentBridgeTests(unittest.TestCase):
    def test_semantic_building_decision_is_bounded_to_one_supported_class(self):
        plan = agent_bridge.decision_to_plan(
            "Extract all buildings using RemoteSAM", {},
            decision(tool_name="semantic_segmentation", arguments={"image_path": "ignored.png", "classes": ["building"]}),
        )
        self.assertEqual(plan["tool"], "semantic_segmentation")
        self.assertEqual(plan["classes"], ["building"])
        self.assertEqual(plan["side"], "all")
        with self.assertRaises(ValueError):
            agent_bridge.decision_to_plan(
                "Extract buildings", {},
                decision(tool_name="semantic_segmentation", arguments={"classes": ["building", "road"]}),
            )
        with self.assertRaises(ValueError):
            agent_bridge.decision_to_plan(
                "Extract buildings", {},
                decision(tool_name="semantic_segmentation", arguments={"classes": ["unknown"]}),
            )

    def test_scope_uses_agent_prompt_and_user_geometry(self):
        plan = agent_bridge.resolve_scope(
            "提取右侧目标并计算占比",
            {},
            {"prompt": "all buildings"},
        )
        self.assertEqual(plan["target"], "building")
        self.assertEqual(plan["side"], "right")
        self.assertEqual(plan["prompt"], "all buildings")

    def test_rejected_tool_fails_closed(self):
        plan = agent_bridge.decision_to_plan(
            "检测车辆", {}, decision(status="rejected_tool", tool_name="skysense_detection")
        )
        self.assertEqual(plan["action"], "clarify")
        self.assertEqual(plan["agent_status"], "rejected_tool")


class OfficialAgentHarnessFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data_patch = patch.object(server, "DATA", Path(self.temp.name))
        self.data_patch.start()
        buffer = io.BytesIO()
        Image.new("RGB", (800, 600), "white").save(buffer, format="PNG")
        uploaded = base64.b64encode(buffer.getvalue()).decode()
        self.session = server.SESSIONS[server.new_session(uploaded=uploaded, name="test.png")["id"]]

    def tearDown(self):
        server.SESSIONS.clear()
        self.data_patch.stop()
        self.temp.cleanup()

    def test_plan_tool_execute_measure_and_continue(self):
        first = decision()
        final = decision(status="completed", text="已根据工具结果完成。", tool_name=None)
        fake_agent = object()
        mask = Image.new("L", (800, 600), 255)
        remote_response = {
            "status": "success",
            "mask": server.image_b64(mask),
            "original_size": [800, 600],
            "output_size": [800, 600],
            "timing_ms": 12.5,
            "peak_vram_mb": 1024,
            "checkpoint_sha256": "abc123",
        }
        with (
            patch.object(server, "live_plan", return_value=(
                {"action": "segment", "target": "building", "side": "right",
                 "tool": "referring_expression_segmentation", "prompt": "all buildings"},
                fake_agent,
                first,
            )),
            patch.object(server, "post_json", return_value=remote_response) as post,
            patch.object(server, "finish_with_agent", return_value=final) as finish,
            patch.dict(server.os.environ, {"GEO_REMOTESAM_URL": "http://example.invalid/predict"}),
        ):
            result = server.run_task(
                self.session,
                {"query": "提取右侧建筑并计算面积占比", "mode": "live"},
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["metrics"]["area_ratio"], 0.5)
        self.assertEqual(result["agent"]["decision"]["status"], "tool_call")
        self.assertEqual(result["agent"]["completion"]["status"], "completed")
        self.assertNotIn("mask", finish.call_args.args[-1]["data"])
        self.assertIn("measurements", finish.call_args.args[-1])
        self.assertEqual(post.call_args.args[1]["text"], "all buildings")
        self.assertIn("RemoteAgent", result["message"])

    def test_semantic_segmentation_uses_model_selected_class(self):
        first = decision(tool_name="semantic_segmentation", arguments={"classes": ["building"]})
        final = decision(status="completed", text="Measured 240000 pixels.", tool_name=None)
        fake_agent = object()
        mask = Image.new("L", (800, 600), 0)
        mask.paste(255, (0, 0, 400, 600))
        response = {"status": "success", "task": "semantic_seg", "mask": server.image_b64(mask)}
        with (
            patch.object(server, "live_plan", return_value=(
                agent_bridge.decision_to_plan("Extract all buildings", {}, first), fake_agent, first,
            )),
            patch.object(server, "post_json", return_value=response) as post,
            patch.object(server, "finish_with_agent", return_value=final) as finish,
            patch.dict(server.os.environ, {"GEO_REMOTESAM_URL": "http://example.invalid/predict"}),
        ):
            result = server.run_task(self.session, {"query": "Extract all buildings", "mode": "live"})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["metrics"]["pixel_area"], 240000)
        self.assertEqual(post.call_args.args[1]["task"], "semantic_seg")
        self.assertEqual(post.call_args.args[1]["classes"], ["building"])
        self.assertEqual(finish.call_args.args[-1]["request"], {"task": "semantic_seg", "classes": ["building"]})

    def test_direct_agent_answer_does_not_call_remotesam(self):
        direct = decision(status="completed", text="这是城市遥感影像。", tool_name=None)
        with (
            patch.object(server, "live_plan", return_value=(
                {"action": "scene", "answer": direct.text, "agent_status": "completed"},
                object(),
                direct,
            )),
            patch.object(server, "post_json") as post,
        ):
            result = server.run_task(self.session, {"query": "介绍影像", "mode": "live"})
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["message"], direct.text)
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
