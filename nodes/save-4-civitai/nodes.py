import hashlib
import json
import os
import re
from datetime import datetime

import comfy.samplers
import folder_paths
import numpy as np
import piexif
import piexif.helper
from nodes import MAX_RESOLUTION
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def parse_name(ckpt_name):
    path = ckpt_name
    filename = path.split("/")[-1]
    filename = filename.split(".")[:-1]
    filename = ".".join(filename)
    return filename


def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as file_obj:
        for byte_block in iter(lambda: file_obj.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def handle_whitespace(string: str):
    return string.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")


def get_timestamp(time_format):
    now = datetime.now()
    try:
        timestamp = now.strftime(time_format)
    except Exception:
        timestamp = now.strftime("%Y-%m-%d-%H%M%S")
    return timestamp


def make_pathname(filename, seed, modelname, counter, time_format):
    filename = filename.replace("%date", get_timestamp("%Y-%m-%d"))
    filename = filename.replace("%time", get_timestamp(time_format))
    filename = filename.replace("%model", modelname)
    filename = filename.replace("%seed", str(seed))
    filename = filename.replace("%counter", str(counter))
    return filename


def make_filename(filename, seed, modelname, counter, time_format):
    filename = make_pathname(filename, seed, modelname, counter, time_format)
    return get_timestamp(time_format) if filename == "" else filename


def _iter_model_types():
    models_root = os.path.abspath(folder_paths.models_dir)
    for model_type, (paths, _) in folder_paths.folder_names_and_paths.items():
        for path in paths:
            try:
                if os.path.commonpath([models_root, os.path.abspath(path)]) == models_root:
                    yield model_type
                    break
            except ValueError:
                continue


def _safe_model_label(model_ref):
    ref = (model_ref or "").replace("\\", "/").strip("/")
    base = os.path.basename(ref)
    parsed = parse_name(base)
    return parsed if parsed else base


def resolve_model_file(model_ref):
    if not isinstance(model_ref, str):
        return "", None, None

    ref = model_ref.replace("\\", "/").strip("/")
    if not ref:
        return "", None, None

    if "/" in ref:
        maybe_type, rel_name = ref.split("/", 1)
        maybe_type = folder_paths.map_legacy(maybe_type)
        if maybe_type in folder_paths.folder_names_and_paths:
            full = folder_paths.get_full_path(maybe_type, rel_name)
            if full is not None:
                return rel_name, full, maybe_type

    full = folder_paths.get_full_path("checkpoints", ref)
    if full is not None:
        return ref, full, "checkpoints"

    direct_matches = []
    for model_type in _iter_model_types():
        full = folder_paths.get_full_path(model_type, ref)
        if full is not None:
            direct_matches.append((ref, full, model_type))

    if len(direct_matches) == 1:
        return direct_matches[0]
    if len(direct_matches) > 1:
        for preferred in ("checkpoints", "diffusion_models"):
            for match in direct_matches:
                if match[2] == preferred:
                    return match
        return direct_matches[0]

    base = os.path.basename(ref).lower()
    basename_matches = []
    for model_type in _iter_model_types():
        try:
            names = folder_paths.get_filename_list(model_type)
        except Exception:
            continue
        for name in names:
            if os.path.basename(name.replace("\\", "/")).lower() == base:
                full = folder_paths.get_full_path(model_type, name)
                if full is not None:
                    basename_matches.append((name, full, model_type))

    if len(basename_matches) == 1:
        return basename_matches[0]
    if len(basename_matches) > 1:
        for preferred in ("checkpoints", "diffusion_models"):
            for match in basename_matches:
                if match[2] == preferred:
                    return match
        return basename_matches[0]

    return ref, None, None


def find_next_filename_counter(output_path, filename_prefix, extension):
    pattern = re.compile(
        rf"^{re.escape(filename_prefix)}_(\d+)(?:_\d+)?\.{re.escape(extension)}$",
        re.IGNORECASE,
    )
    existing_counters = []
    for entry in os.listdir(output_path):
        match = pattern.match(entry)
        if match is not None:
            existing_counters.append(int(match.group(1)))
    if existing_counters:
        return max(existing_counters) + 1
    return 1


def build_output_filename(filename_prefix, extension, counter_value, filename_number_padding, image_index, total_images):
    counter_stem = f"{filename_prefix}_{counter_value:0{filename_number_padding}d}"
    if total_images > 1:
        counter_stem = f"{counter_stem}_{image_index:02d}"
    return f"{counter_stem}.{extension}"


class Save4CivitAI:
    CATEGORY = "Malaombra-Custom-Nodes/Image"
    RETURN_TYPES = ()
    FUNCTION = "save_files"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename": ("STRING", {"default": "%time_%seed", "multiline": False}),
                "path": ("STRING", {"default": "", "multiline": False}),
                "extension": (["png", "jpeg", "webp"],),
                "filename_number_padding": ("INT", {"default": 4, "min": 1, "max": 9, "step": 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "modelname": (folder_paths.get_filename_list("checkpoints"),),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            },
            "optional": {
                "positive": ("STRING", {"default": "unknown", "multiline": True}),
                "negative": ("STRING", {"default": "unknown", "multiline": True}),
                "seed_value": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
                "lossless_webp": ("BOOLEAN", {"default": True}),
                "quality_jpeg_or_webp": ("INT", {"default": 100, "min": 1, "max": 100}),
                "counter": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "time_format": ("STRING", {"default": "%Y-%m-%d-%H%M%S", "multiline": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def save_files(
        self,
        images,
        seed_value,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        modelname,
        quality_jpeg_or_webp,
        lossless_webp,
        width,
        height,
        counter,
        filename,
        path,
        extension,
        filename_number_padding,
        time_format,
        prompt=None,
        extra_pnginfo=None,
    ):
        resolved_name, model_path, model_type = resolve_model_file(modelname)
        model_label = _safe_model_label(resolved_name if resolved_name else modelname)

        filename = make_filename(filename, seed_value, model_label, counter, time_format)
        path = make_pathname(path, seed_value, model_label, counter, time_format)

        if model_path is not None:
            modelhash = calculate_sha256(model_path)[:10]
        else:
            modelhash = "unknown"
            print(f"[Save 4 CivitAI] Could not resolve model path for '{modelname}'.")

        model_desc = model_label
        if model_type and model_type != "checkpoints":
            model_desc = f"{model_label} ({model_type})"

        comment = (
            f"{handle_whitespace(positive)}\n"
            f"Negative prompt: {handle_whitespace(negative)}\n"
            f"Steps: {steps}, Sampler: {sampler_name}{f'_{scheduler}' if scheduler != 'normal' else ''}, "
            f"CFG Scale: {cfg}, Seed: {seed_value}, Size: {width}x{height}, "
            f"Model hash: {modelhash}, Model: {model_desc}, Version: ComfyUI"
        )

        output_path = os.path.join(self.output_dir, path)
        os.makedirs(output_path, exist_ok=True)

        counter_start = find_next_filename_counter(output_path, filename, extension)
        filenames = self.save_images(
            images,
            output_path,
            filename,
            comment,
            extension,
            quality_jpeg_or_webp,
            lossless_webp,
            filename_number_padding,
            counter_start,
            prompt,
            extra_pnginfo,
        )

        subfolder = os.path.normpath(path)
        ui_images = [
            {
                "filename": saved_name,
                "subfolder": subfolder if subfolder != "." else "",
                "type": "output",
            }
            for saved_name in filenames
        ]
        return {"ui": {"images": ui_images}}

    def save_images(
        self,
        images,
        output_path,
        filename_prefix,
        comment,
        extension,
        quality_jpeg_or_webp,
        lossless_webp,
        filename_number_padding,
        counter_start,
        prompt=None,
        extra_pnginfo=None,
    ):
        total_images = int(images.size()[0])
        counter_value = counter_start

        while True:
            collision = False
            for image_index in range(1, total_images + 1):
                candidate = build_output_filename(
                    filename_prefix,
                    extension,
                    counter_value,
                    filename_number_padding,
                    image_index,
                    total_images,
                )
                if os.path.exists(os.path.join(output_path, candidate)):
                    collision = True
                    break
            if not collision:
                break
            counter_value += 1

        paths = []
        img_count = 1
        for image in images:
            array = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            out_name = build_output_filename(
                filename_prefix,
                extension,
                counter_value,
                filename_number_padding,
                img_count,
                total_images,
            )
            out_file = os.path.join(output_path, out_name)

            if extension == "png":
                metadata = PngInfo()
                metadata.add_text("parameters", comment)

                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for key in extra_pnginfo:
                        metadata.add_text(key, json.dumps(extra_pnginfo[key]))

                img.save(out_file, pnginfo=metadata, optimize=True)
            else:
                save_kwargs = {"optimize": True, "quality": quality_jpeg_or_webp}
                if extension == "webp":
                    save_kwargs["lossless"] = lossless_webp

                img.save(out_file, **save_kwargs)
                exif_bytes = piexif.dump(
                    {
                        "Exif": {
                            piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(
                                comment,
                                encoding="unicode",
                            )
                        }
                    }
                )
                piexif.insert(exif_bytes, out_file)

            paths.append(out_name)
            img_count += 1

        return paths


NODE_CLASS_MAPPINGS = {
    "Save 4 CivitAI": Save4CivitAI,
}
