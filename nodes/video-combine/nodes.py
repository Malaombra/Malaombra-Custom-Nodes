from __future__ import annotations

import json
import os
from fractions import Fraction

import av
import folder_paths
import torch
from comfy.cli_args import args
from comfy_api.latest import InputImpl


class MalaombraVideoCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Malaombra/video"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 240.0, "step": 0.01}),
                "format": (["mp4", "webm", "mkv"], {"default": "mp4"}),
                "codec": (["h264", "h265", "vp9", "av1"], {"default": "h264"}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 63, "step": 1}),
                "show_preview": ("BOOLEAN", {"default": True}),
                "save_output": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", "VIDEO")
    RETURN_NAMES = ("filename", "video")
    CATEGORY = "Malaombra/video"
    FUNCTION = "combine_video"
    OUTPUT_NODE = True

    _CODEC_MAP = {
        "h264": "libx264",
        "h265": "libx265",
        "vp9": "libvpx-vp9",
        "av1": "libsvtav1",
    }

    _FORMAT_DEFAULT_CODEC = {
        "mp4": {"vp9": "h264"},
        "webm": {"h264": "vp9", "h265": "vp9"},
        "mkv": {},
    }

    def _select_codec(self, fmt: str, codec: str) -> str:
        return self._FORMAT_DEFAULT_CODEC.get(fmt, {}).get(codec, codec)

    def _frame_to_ndarray(self, frame):
        return torch.clamp(frame[..., :3] * 255, min=0, max=255).to(
            device=torch.device("cpu"),
            dtype=torch.uint8,
        ).numpy()

    def _metadata(self, prompt=None, extra_pnginfo=None):
        if args.disable_metadata:
            return {}

        metadata = {}
        if prompt is not None:
            metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for key, value in extra_pnginfo.items():
                metadata[str(key)] = json.dumps(value)
        return metadata

    def combine_video(
        self,
        images,
        filename_prefix,
        fps,
        format,
        codec,
        crf,
        show_preview=True,
        save_output=False,
        prompt=None,
        extra_pnginfo=None,
    ):
        if images is None or len(images) == 0:
            raise ValueError("Malaombra Video Combine requires at least one image frame.")

        height = int(images[0].shape[0])
        width = int(images[0].shape[1])
        output_root = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            output_root,
            width,
            height,
        )

        selected_codec = self._select_codec(format, codec)
        file = f"{filename}_{counter:05}_.{format}"
        output_path = os.path.join(full_output_folder, file)

        container = av.open(output_path, mode="w")
        container.metadata.update(self._metadata(prompt, extra_pnginfo))

        stream = container.add_stream(self._CODEC_MAP[selected_codec], rate=Fraction(round(float(fps) * 1000), 1000))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p10le" if selected_codec == "av1" else "yuv420p"
        stream.options = {"crf": str(int(crf))}
        if selected_codec == "av1":
            stream.options["preset"] = "6"

        for frame in images:
            video_frame = av.VideoFrame.from_ndarray(self._frame_to_ndarray(frame), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
        container.close()

        folder_type = "output" if save_output else "temp"
        result = {
            "result": (output_path, InputImpl.VideoFromFile(output_path)),
        }
        if show_preview or save_output:
            result["ui"] = {
                "gifs": [
                    {
                        "filename": file,
                        "subfolder": subfolder,
                        "type": folder_type,
                        "format": f"video/{format}",
                    }
                ],
                "animated": (True,),
            }
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
