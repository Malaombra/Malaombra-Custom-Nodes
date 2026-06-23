from __future__ import annotations

import os
import importlib
import sys
import types
from pathlib import Path

from comfy_api.latest._input_impl.video_types import VideoFromFile

_COMFYUI_DIR = Path(__file__).resolve().parents[4]
_CUSTOM_NODES_DIR = _COMFYUI_DIR / "custom_nodes"
_VHS_PACKAGE_DIR = _CUSTOM_NODES_DIR / "comfyui-videohelpersuite" / "videohelpersuite"
if _COMFYUI_DIR.exists():
    sys.path.insert(0, str(_COMFYUI_DIR))
    importlib.import_module("utils")
if "videohelpersuite" not in sys.modules and _VHS_PACKAGE_DIR.exists():
    vhs_package = types.ModuleType("videohelpersuite")
    vhs_package.__path__ = [str(_VHS_PACKAGE_DIR)]
    sys.modules["videohelpersuite"] = vhs_package

from videohelpersuite.nodes import VideoCombine as VHSVideoCombine


class MalaombraVideoCombine(VHSVideoCombine):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"] = dict(inputs["required"])
        inputs["required"].pop("pingpong", None)
        inputs["required"].pop("save_output", None)
        inputs["required"]["crf"] = ("INT", {"default": 19, "min": 0, "max": 100, "step": 1})
        inputs["required"]["show_preview"] = ("BOOLEAN", {"default": True})
        inputs["required"]["save_output"] = ("BOOLEAN", {"default": False})
        return inputs

    RETURN_TYPES = ("VHS_FILENAMES", "VIDEO")
    RETURN_NAMES = ("Filenames", "video")
    CATEGORY = "Malaombra/video"
    FUNCTION = "combine_video"
    OUTPUT_NODE = True

    def combine_video(self, show_preview=True, save_output=False, **kwargs):
        result = super().combine_video(
            save_output=save_output,
            pingpong=False,
            pix_fmt="yuv420p",
            trim_to_audio=False,
            **kwargs,
        )

        if not isinstance(result, dict):
            return result

        filenames = result.get("result", ((save_output, []),))[0]
        output_files = filenames[1] if len(filenames) > 1 else []
        video_path = output_files[-1] if output_files else None
        video = VideoFromFile(video_path) if video_path and os.path.isfile(video_path) else None

        if not (show_preview or save_output):
            result = {key: value for key, value in result.items() if key != "ui"}
        elif "ui" in result and "gifs" in result["ui"]:
            result["ui"] = dict(result["ui"])
            result["ui"]["images"] = result["ui"]["gifs"]
            result["ui"]["animated"] = (True,)

        result["result"] = (filenames, video)
        return result

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


NODE_CLASS_MAPPINGS = {
    "Malaombra_VideoCombine": MalaombraVideoCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Malaombra_VideoCombine": "Video Combine",
}
