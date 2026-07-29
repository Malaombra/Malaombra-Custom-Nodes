from __future__ import annotations

import json
from typing import Any


CONFIG_TYPE = "COMICCHAT_CONFIG"
MAX_INT = 0x7FFFFFFFFFFFFFFF


def _json_object(value: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _parameter_schema(schema: dict[str, Any]) -> dict[str, Any]:
    parameters = schema.get("parameters", schema)
    return parameters if isinstance(parameters, dict) else {}


def _validate_parameter(name: str, value: Any, specification: dict[str, Any]) -> None:
    expected = specification.get("type")
    valid = {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }
    if expected in valid and not valid[expected]:
        raise ValueError(f"Parameter '{name}' must be {expected}")
    choices = specification.get("enum")
    if isinstance(choices, list) and value not in choices:
        raise ValueError(f"Parameter '{name}' must be one of: {', '.join(map(str, choices))}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = specification.get("minimum")
        maximum = specification.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            raise ValueError(f"Parameter '{name}' must be at least {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            raise ValueError(f"Parameter '{name}' must be at most {maximum}")


def _validated_config(
    workflow_id: str,
    workflow_version: int,
    schema_json: str,
    payload_json: str,
) -> dict[str, Any]:
    schema = _json_object(schema_json, "ComicChat schema")
    values = _json_object(payload_json, "ComicChat payload")
    parameters = _parameter_schema(schema)
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(values) - set(parameters))
        if unknown:
            raise ValueError(f"Unknown ComicChat parameters: {', '.join(unknown)}")
    for name, value in values.items():
        specification = parameters.get(name)
        if isinstance(specification, dict):
            _validate_parameter(name, value, specification)
    return {
        "workflow_id": str(workflow_id).strip(),
        "workflow_version": int(workflow_version),
        "schema": schema,
        "values": values,
    }


def _value(config: dict[str, Any], key: str, default: Any) -> Any:
    current: Any = config.get("values", {})
    for segment in str(key).split("."):
        if not isinstance(current, dict) or segment not in current:
            return default
        current = current[segment]
    return current


class ComicChatWorkflowInput:
    RETURN_TYPES = (CONFIG_TYPE,)
    RETURN_NAMES = ("config",)
    FUNCTION = "build_config"
    CATEGORY = "Malaombra-Custom-Nodes/ComicChat"
    DESCRIPTION = "Single, typed configuration gateway between ComicChat and a ComfyUI workflow."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_id": ("STRING", {"default": "workflow"}),
                "workflow_version": ("INT", {"default": 1, "min": 1, "max": 100000}),
                "schema_json": ("STRING", {"default": "{\"parameters\":{}}", "multiline": True}),
                "payload_json": ("STRING", {"default": "{}", "multiline": True}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, workflow_id, workflow_version, schema_json, payload_json):
        try:
            config = _validated_config(workflow_id, workflow_version, schema_json, payload_json)
        except ValueError as error:
            return str(error)
        if not config["workflow_id"]:
            return "ComicChat workflow_id cannot be empty"
        return True

    def build_config(self, workflow_id, workflow_version, schema_json, payload_json):
        return (_validated_config(workflow_id, workflow_version, schema_json, payload_json),)


class _ComicChatGetter:
    CATEGORY = "Malaombra-Custom-Nodes/ComicChat/Get Parameter"

    @staticmethod
    def get_value(config, key, default):
        return (_value(config, key, default),)


class ComicChatGetString(_ComicChatGetter):
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_value"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"config": (CONFIG_TYPE,), "key": ("STRING", {"default": "prompt"}), "default": ("STRING", {"default": "", "multiline": True})}}


class ComicChatGetInteger(_ComicChatGetter):
    RETURN_TYPES = ("INT",)
    FUNCTION = "get_integer"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"config": (CONFIG_TYPE,), "key": ("STRING", {"default": "seed"}), "default": ("INT", {"default": 0, "min": -MAX_INT, "max": MAX_INT})}}

    @staticmethod
    def get_integer(config, key, default):
        return (int(_value(config, key, default)),)


class ComicChatGetFloat(_ComicChatGetter):
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "get_float"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"config": (CONFIG_TYPE,), "key": ("STRING", {"default": "cfg"}), "default": ("FLOAT", {"default": 0.0, "min": -1000000.0, "max": 1000000.0, "step": 0.01})}}

    @staticmethod
    def get_float(config, key, default):
        return (float(_value(config, key, default)),)


class ComicChatGetBoolean(_ComicChatGetter):
    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "get_boolean"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"config": (CONFIG_TYPE,), "key": ("STRING", {"default": "enabled"}), "default": ("BOOLEAN", {"default": False})}}

    @staticmethod
    def get_boolean(config, key, default):
        value = _value(config, key, default)
        if isinstance(value, str):
            value = value.strip().lower() in {"1", "true", "yes", "on"}
        return (bool(value),)


class ComicChatGetCombo(_ComicChatGetter):
    RETURN_TYPES = ("*",)
    FUNCTION = "get_combo"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"config": (CONFIG_TYPE,), "key": ("STRING", {"default": "choice"}), "default": ("STRING", {"default": ""})}}

    @staticmethod
    def get_combo(config, key, default):
        return (str(_value(config, key, default)),)


class ComicChatLoraStack:
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply_loras"
    CATEGORY = "Malaombra-Custom-Nodes/ComicChat"
    DESCRIPTION = "Applies the LoRA list declared in the ComicChat workflow payload."

    def __init__(self):
        self._loaders: dict[str, Any] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "config": (CONFIG_TYPE,),
                "key": ("STRING", {"default": "loras"}),
            }
        }

    def apply_loras(self, model, clip, config, key):
        from nodes import LoraLoader

        selections = _value(config, key, [])
        if not isinstance(selections, list):
            raise ValueError(f"ComicChat parameter '{key}' must be an array")
        active_names: set[str] = set()
        for selection in selections:
            if not isinstance(selection, dict) or selection.get("enabled") is False:
                continue
            name = str(selection.get("name") or "").strip()
            if not name:
                continue
            active_names.add(name)
            strength = float(selection.get("strength", 1.0))
            strength_model = float(selection.get("strengthModel", strength))
            strength_clip = float(selection.get("strengthClip", strength))
            loader = self._loaders.setdefault(name, LoraLoader())
            model, clip = loader.load_lora(model, clip, name, strength_model, strength_clip)
        for stale_name in set(self._loaders) - active_names:
            self._loaders.pop(stale_name, None)
        return (model, clip)


NODE_CLASS_MAPPINGS = {
    "Malaombra ComicChat Workflow Input": ComicChatWorkflowInput,
    "Malaombra ComicChat Get String": ComicChatGetString,
    "Malaombra ComicChat Get Integer": ComicChatGetInteger,
    "Malaombra ComicChat Get Float": ComicChatGetFloat,
    "Malaombra ComicChat Get Boolean": ComicChatGetBoolean,
    "Malaombra ComicChat Get Combo": ComicChatGetCombo,
    "Malaombra ComicChat LoRA Stack": ComicChatLoraStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Malaombra ComicChat Workflow Input": "ComicChat · Workflow Input",
    "Malaombra ComicChat Get String": "ComicChat · Get String",
    "Malaombra ComicChat Get Integer": "ComicChat · Get Integer",
    "Malaombra ComicChat Get Float": "ComicChat · Get Float",
    "Malaombra ComicChat Get Boolean": "ComicChat · Get Boolean",
    "Malaombra ComicChat Get Combo": "ComicChat · Get Choice",
    "Malaombra ComicChat LoRA Stack": "ComicChat · LoRA Stack",
}
