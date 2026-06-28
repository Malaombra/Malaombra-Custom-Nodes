from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


class MalaombraCaptionSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"forceInput": True}),
                "image_path": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "custom_output_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Custom output directory path. If empty, will use the directory of image_path",
                    },
                ),
                "custom_file_name": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Custom filename (without extension). Leave empty to use original image names.",
                    },
                ),
                "overwrite": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If true, overwrite existing files. If false, add a number to make filenames unique.",
                    },
                ),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_caption"
    CATEGORY = "Malaombra/Text"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True

    def _as_list(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _first_widget_value(self, value, default=""):
        if isinstance(value, list):
            return value[0] if value else default
        return value

    def _get_unique_filename(self, base_path: Path) -> Path:
        if not base_path.exists():
            return base_path

        counter = 1
        while True:
            new_path = base_path.parent / f"{base_path.stem}_{counter:02d}{base_path.suffix}"
            if not new_path.exists():
                return new_path
            counter += 1

    def _tensor_to_image(self, image):
        if not isinstance(image, torch.Tensor):
            return image

        while len(image.shape) > 3:
            image = image.squeeze(0)

        image = (image.cpu().numpy() * 255).astype(np.uint8)
        if image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))
        return Image.fromarray(image)

    def save_caption(self, string, image_path, image=None, custom_output_path="", custom_file_name="", overwrite=True):
        strings = self._as_list(string)
        image_paths = self._as_list(image_path)
        images = self._as_list(image)
        custom_output_path = str(self._first_widget_value(custom_output_path, "") or "")
        custom_file_name = str(self._first_widget_value(custom_file_name, "") or "")
        overwrite = bool(self._first_widget_value(overwrite, True))

        item_count = max(len(strings), len(image_paths), len(images), 1)

        for index in range(item_count):
            current_string = strings[index] if index < len(strings) else strings[-1]
            current_image_path = Path(image_paths[index] if index < len(image_paths) else image_paths[-1])
            current_image = images[index] if index < len(images) else None

            save_dir = Path(custom_output_path.strip()) if custom_output_path.strip() else current_image_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)

            if custom_file_name.strip():
                base_name = custom_file_name.strip() if item_count == 1 else f"{custom_file_name.strip()}_{index + 1:04d}"
            else:
                base_name = current_image_path.stem

            txt_path = save_dir / f"{base_name}.txt"
            if not overwrite and txt_path.exists():
                txt_path = self._get_unique_filename(txt_path)
                base_name = txt_path.stem

            txt_path.write_text(str(current_string), encoding="utf-8")
            print(f"[Malaombra Caption Saver][{txt_path.name}]: {current_string}")

            if current_image is not None and custom_output_path.strip():
                try:
                    current_image = self._tensor_to_image(current_image)
                    current_image.save(save_dir / f"{base_name}{current_image_path.suffix}")
                except Exception as exc:
                    print(f"[Malaombra Caption Saver] Failed to copy image: {exc}")

        return ()


NODE_CLASS_MAPPINGS = {
    "MalaombraCaptionSaver": MalaombraCaptionSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MalaombraCaptionSaver": "Malaombra Caption Saver",
}
