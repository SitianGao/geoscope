from __future__ import annotations

import base64
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REMOTEAGENT_ROOT = Path(__file__).resolve().parents[1] / "services" / "remoteagent" / "upstream"
sys.path.insert(0, str(REMOTEAGENT_ROOT))

from remoteagent import RemoteAgent
from remoteagent.services import ServiceExecutor


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class FakeRemoteAgent(RemoteAgent):
    def __init__(self, responses, **kwargs):
        super().__init__(system_prompt="test prompt", **kwargs)
        self.responses = list(responses)
        self.seen_messages = []

    def _run_llm(self, messages):
        self.seen_messages.append(messages)
        return self.responses.pop(0)


class RemoteAgentWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.image_path = Path(self.tempdir.name) / "sample.png"
        self.image_path.write_bytes(PNG_1X1)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_plan_sends_real_image_and_keeps_direct_answer(self):
        agent = FakeRemoteAgent(["<answer>urban residential area</answer>"])

        result = agent.plan(
            "Describe the image.",
            self.image_path,
            context={"side": "right"},
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.text, "urban residential area")
        content = agent.seen_messages[0][-1]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertIn('"side": "right"', content[0]["text"])
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_plan_binds_tool_to_current_image(self):
        agent = FakeRemoteAgent(
            [
                'T_call(referring_expression_segmentation, "C:/untrusted.png", "all buildings")'
            ],
            allowed_tools={"referring_expression_segmentation"},
        )

        result = agent.plan("Extract buildings.", self.image_path)

        self.assertEqual(result.status, "tool_call")
        self.assertEqual(result.tool_call.name, "referring_expression_segmentation")
        self.assertEqual(result.tool_call.arguments["image_path"], str(self.image_path.resolve()))

    def test_plan_accepts_tool_call_wrapped_in_answer_tag(self):
        agent = FakeRemoteAgent(
            [
                '<think>Use RemoteSAM.</think><answer>'
                'T_call(semantic_segmentation, "C:/untrusted.png", ["building"])'
                '</answer>'
            ],
            allowed_tools={"semantic_segmentation"},
        )

        result = agent.plan("Extract all buildings.", self.image_path)

        self.assertEqual(result.status, "tool_call")
        self.assertEqual(result.tool_call.name, "semantic_segmentation")
        self.assertEqual(result.tool_call.arguments["classes"], ["building"])
        self.assertEqual(result.tool_call.arguments["image_path"], str(self.image_path.resolve()))

    def test_plan_accepts_single_xml_style_tool_call_from_checkpoint(self):
        agent = FakeRemoteAgent(
            [
                '<think>Use one segmentation tool.</think>\n'
                '<T_call semantic_segmentation("/C:/untrusted.png", ["aircraft"])></T_call>'
            ],
            allowed_tools={"semantic_segmentation"},
        )

        result = agent.plan("改成左侧的飞机", self.image_path)

        self.assertEqual(result.status, "tool_call")
        self.assertEqual(result.tool_call.name, "semantic_segmentation")
        self.assertEqual(result.tool_call.arguments["classes"], ["aircraft"])
        self.assertEqual(result.tool_call.arguments["image_path"], str(self.image_path.resolve()))

    def test_plan_rejects_angle_wrapped_unavailable_tool(self):
        agent = FakeRemoteAgent(
            [
                '<T_call(oriented_object_detection, "C:/untrusted.png", '
                '["aircraft"], modality="rgb")>'
            ],
            allowed_tools={"semantic_segmentation"},
        )

        result = agent.plan("提取右侧飞机", self.image_path)

        self.assertEqual(result.status, "rejected_tool")
        self.assertEqual(result.tool_call.name, "sm3det_oriented_detection")

    def test_plan_does_not_salvage_multiple_xml_tool_calls(self):
        agent = FakeRemoteAgent(
            [
                '<T_call>(semantic_segmentation, ["building"]), '
                'T_call(region_contour_extraction, [420, 0, 812, 588])</T_call>'
            ],
            allowed_tools={"semantic_segmentation"},
        )

        result = agent.plan("提取右侧建筑", self.image_path)

        self.assertEqual(result.status, "unparsed")
        self.assertIsNone(result.tool_call)

    def test_plan_rejects_tool_that_is_not_deployed(self):
        agent = FakeRemoteAgent(
            ['T_call(object_detection, "C:/untrusted.png", ["ship"])'],
            allowed_tools={"referring_expression_segmentation"},
        )

        result = agent.plan("Find ships.", self.image_path)

        self.assertEqual(result.status, "rejected_tool")
        self.assertIn("skysense_detection", result.text)

    def test_self_executing_loop_preserves_full_tool_response(self):
        agent = FakeRemoteAgent(
            [
                'T_call(referring_expression_segmentation, "ignored.png", "all buildings")',
                "<answer>The mask was produced successfully.</answer>",
            ],
            allowed_tools={"referring_expression_segmentation"},
            api_urls={"remotesam": "http://127.0.0.1:6657"},
        )
        full_result = {
            "status": "success",
            "tool_name": "referring_expression_segmentation",
            "service": "remotesam",
            "message": "Success",
            "data": {"status": "success", "mask": "very-large-mask-data"},
        }
        agent._executor.execute_structured = lambda name, args: full_result

        result = agent.run("Extract buildings.", self.image_path)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tool_results[0]["data"]["mask"], "very-large-mask-data")
        feedback = agent.seen_messages[1][-1]["content"]
        self.assertNotIn("very-large-mask-data", feedback)
        self.assertIn("artifact omitted", feedback)

    @patch("remoteagent.services.executor.http_post_json")
    def test_remote_sam_structured_execution_retains_mask(self, post_json):
        post_json.return_value = {
            "status": "success",
            "message": "segmented",
            "mask": "base64-mask",
            "mask_width": 1,
            "mask_height": 1,
        }
        executor = ServiceExecutor({"remotesam": "http://127.0.0.1:6657"})

        result = executor.execute_structured(
            "referring_expression_segmentation",
            {"image_path": str(self.image_path), "prompt": "all buildings"},
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"]["mask"], "base64-mask")
        sent_payload = post_json.call_args.args[1]
        self.assertEqual(sent_payload["task"], "referring_seg")
        self.assertNotIn("image", result["request"])


if __name__ == "__main__":
    unittest.main()
