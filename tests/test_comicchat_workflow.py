from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes" / "comicchat-workflow" / "nodes.py"
SPEC = importlib.util.spec_from_file_location("comicchat_workflow_nodes", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ComicChatWorkflowNodesTest(unittest.TestCase):
    def setUp(self):
        self.schema = json.dumps(
            {
                "additionalProperties": False,
                "parameters": {
                    "prompt": {"type": "string"},
                    "steps": {"type": "integer", "minimum": 1, "maximum": 30},
                    "cfg": {"type": "number"},
                    "enabled": {"type": "boolean"},
                    "sampler": {"type": "string", "enum": ["euler", "dpmpp_2m"]},
                    "loras": {"type": "array"},
                },
            }
        )
        self.payload = json.dumps(
            {
                "prompt": "portrait",
                "steps": 12,
                "cfg": 1.25,
                "enabled": True,
                "sampler": "euler",
                "loras": [{"name": "style.safetensors", "strength": -0.5}],
            }
        )

    def test_gateway_and_typed_getters(self):
        config = MODULE.ComicChatWorkflowInput().build_config("krea", 2, self.schema, self.payload)[0]
        self.assertEqual(MODULE.ComicChatGetString.get_value(config, "prompt", ""), ("portrait",))
        self.assertEqual(MODULE.ComicChatGetInteger.get_integer(config, "steps", 1), (12,))
        self.assertEqual(MODULE.ComicChatGetFloat.get_float(config, "cfg", 0.0), (1.25,))
        self.assertEqual(MODULE.ComicChatGetBoolean.get_boolean(config, "enabled", False), (True,))
        self.assertEqual(MODULE.ComicChatGetCombo.get_combo(config, "sampler", "beta"), ("euler",))
        self.assertEqual(MODULE.ComicChatGetString.get_value(config, "missing", "fallback"), ("fallback",))

    def test_validation_rejects_unknown_and_out_of_range_values(self):
        invalid = json.dumps({"steps": 31})
        result = MODULE.ComicChatWorkflowInput.VALIDATE_INPUTS("krea", 2, self.schema, invalid)
        self.assertIn("at most 30", result)
        unknown = json.dumps({"other": 1})
        result = MODULE.ComicChatWorkflowInput.VALIDATE_INPUTS("krea", 2, self.schema, unknown)
        self.assertIn("Unknown ComicChat parameters", result)

    def test_negative_lora_strength_is_preserved(self):
        config = MODULE.ComicChatWorkflowInput().build_config("krea", 2, self.schema, self.payload)[0]
        self.assertEqual(config["values"]["loras"][0]["strength"], -0.5)


if __name__ == "__main__":
    unittest.main()
