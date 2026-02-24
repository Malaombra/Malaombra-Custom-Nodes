import os

import folder_paths


class GetModelNameFromModel:
    CATEGORY = "Malaombra-Custom-Nodes/utils"
    FUNCTION = "get_model_name"
    RETURN_TYPES = (folder_paths.get_filename_list("checkpoints"),)
    RETURN_NAMES = ("modelname",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("*", {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    @staticmethod
    def _model_categories():
        categories = []
        models_root = os.path.abspath(folder_paths.models_dir)
        for category, (paths, _) in folder_paths.folder_names_and_paths.items():
            for path in paths:
                try:
                    abs_path = os.path.abspath(path)
                    if os.path.commonpath([models_root, abs_path]) == models_root:
                        categories.append(category)
                        break
                except ValueError:
                    continue

        preferred = ["checkpoints", "diffusion_models"]
        ordered = []
        for name in preferred:
            if name in categories:
                ordered.append(name)
        for name in categories:
            if name not in ordered:
                ordered.append(name)
        return ordered

    @staticmethod
    def _build_model_index():
        index = []
        for category in GetModelNameFromModel._model_categories():
            try:
                names = folder_paths.get_filename_list(category)
            except Exception:
                continue
            for name in names:
                index.append((category, name))
        return index

    @staticmethod
    def _format_output_name(category, model_name):
        if category == "checkpoints":
            return model_name
        return f"{category}/{model_name}"

    @staticmethod
    def _match_model_name(raw_value, model_index):
        if not isinstance(raw_value, str):
            return None

        value = raw_value.strip()
        if not value:
            return None

        normalized = value.replace("\\", "/").strip("/")
        normalized_lower = normalized.lower()

        if "/" in normalized:
            maybe_category, rest = normalized.split("/", 1)
            maybe_category = maybe_category.strip().lower()
            for category, name in model_index:
                if category.lower() == maybe_category and name.replace("\\", "/").lower() == rest.lower():
                    return category, name

        exact_matches = []
        for category, name in model_index:
            if name.replace("\\", "/").lower() == normalized_lower:
                exact_matches.append((category, name))

        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            for preferred in ("checkpoints", "diffusion_models"):
                for match in exact_matches:
                    if match[0] == preferred:
                        return match
            return exact_matches[0]

        basename = os.path.basename(normalized_lower)
        basename_matches = []
        for category, name in model_index:
            if os.path.basename(name.replace("\\", "/")).lower() == basename:
                basename_matches.append((category, name))

        if len(basename_matches) == 1:
            return basename_matches[0]
        if len(basename_matches) > 1:
            for preferred in ("checkpoints", "diffusion_models"):
                for match in basename_matches:
                    if match[0] == preferred:
                        return match
            return basename_matches[0]

        return None

    @staticmethod
    def _resolve_prompt_key(prompt, node_id, current_key=None):
        if node_id is None:
            return None

        direct_candidates = [node_id, str(node_id)]
        for candidate in direct_candidates:
            if candidate in prompt:
                return candidate

        node_id_str = str(node_id)
        if current_key is not None:
            current_key_str = str(current_key)
            if ":" in current_key_str and ":" not in node_id_str:
                prefix = current_key_str.rsplit(":", 1)[0]
                prefixed = f"{prefix}:{node_id_str}"
                if prefixed in prompt:
                    return prefixed

        tail = node_id_str.split(":")[-1]
        matches = [key for key in prompt.keys() if str(key).split(":")[-1] == tail]
        if len(matches) == 1:
            return matches[0]

        if matches and current_key is not None:
            current_key_str = str(current_key)
            if ":" in current_key_str:
                prefix = current_key_str.rsplit(":", 1)[0]
                prefixed_matches = [key for key in matches if str(key).startswith(f"{prefix}:")]
                if len(prefixed_matches) == 1:
                    return prefixed_matches[0]

        return None

    def _find_upstream_model_name(self, prompt, start_node_key, model_index):
        model_input_keys = (
            "ckpt_name",
            "checkpoint",
            "checkpoint_name",
            "model_name",
            "modelname",
            "diffusion_model",
            "diffusion_model_name",
            "unet_name",
            "model_path",
            "model_file",
            "filename",
            "weight_name",
        )
        priority_link_inputs = {
            "source",
            "model",
            "base_model",
            "refiner_model",
            "model1",
            "model2",
        }

        visited = set()
        stack = [start_node_key]

        while stack:
            node_key = stack.pop()
            if node_key in visited:
                continue
            visited.add(node_key)

            node_data = prompt.get(node_key)
            if not isinstance(node_data, dict):
                continue

            inputs = node_data.get("inputs", {})
            if not isinstance(inputs, dict):
                continue

            for key in model_input_keys:
                match = self._match_model_name(inputs.get(key), model_index)
                if match is not None:
                    return self._format_output_name(match[0], match[1])

            priority_links = []
            other_links = []

            for input_name, input_value in inputs.items():
                if not isinstance(input_value, (list, tuple)) or len(input_value) < 1:
                    continue

                upstream_key = self._resolve_prompt_key(prompt, input_value[0], current_key=node_key)
                if upstream_key is None:
                    continue

                if input_name in priority_link_inputs or input_name.lower().startswith("model"):
                    priority_links.append(upstream_key)
                else:
                    other_links.append(upstream_key)

            stack.extend(reversed(other_links))
            stack.extend(reversed(priority_links))

        return None

    @staticmethod
    def _first_fallback(model_index):
        for category, name in model_index:
            if category == "checkpoints":
                return name
        if model_index:
            category, name = model_index[0]
            return GetModelNameFromModel._format_output_name(category, name)
        return ""

    @staticmethod
    def _get_upstream_from_current_node(prompt, current_key):
        current_node = prompt.get(current_key, {})
        if not isinstance(current_node, dict):
            return None
        inputs = current_node.get("inputs", {})
        if not isinstance(inputs, dict):
            return None

        for key in ("source", "model", "input", "any_input"):
            linked = inputs.get(key)
            if isinstance(linked, (list, tuple)) and len(linked) >= 1:
                return linked[0]

        for _, linked in inputs.items():
            if isinstance(linked, (list, tuple)) and len(linked) >= 1:
                return linked[0]

        return None

    def get_model_name(self, source, prompt=None, unique_id=None):
        model_index = self._build_model_index()
        fallback = self._first_fallback(model_index)

        checkpoints = folder_paths.get_filename_list("checkpoints")

        if not isinstance(prompt, dict):
            direct_match = self._match_model_name(source, model_index)
            if direct_match is not None:
                resolved = self._format_output_name(direct_match[0], direct_match[1])
                return (resolved,)
            print("[get-model-name-from-model] Prompt metadata unavailable, using fallback model.")
            return (fallback,)

        current_key = self._resolve_prompt_key(prompt, unique_id)
        if current_key is None:
            direct_match = self._match_model_name(source, model_index)
            if direct_match is not None:
                resolved = self._format_output_name(direct_match[0], direct_match[1])
                return (resolved,)
            print("[get-model-name-from-model] Could not resolve current node id in prompt, using fallback model.")
            return (fallback,)

        upstream_id = self._get_upstream_from_current_node(prompt, current_key)
        upstream_key = self._resolve_prompt_key(prompt, upstream_id, current_key=current_key)

        search_start = upstream_key if upstream_key is not None else current_key
        resolved = self._find_upstream_model_name(prompt, search_start, model_index)

        if resolved is None:
            direct_match = self._match_model_name(source, model_index)
            if direct_match is not None:
                resolved = self._format_output_name(direct_match[0], direct_match[1])
            else:
                print("[get-model-name-from-model] Could not find upstream model name, using fallback model.")
                resolved = fallback

        if resolved and checkpoints and resolved not in checkpoints and "/" in resolved:
            category = resolved.split("/", 1)[0]
            if category == "checkpoints":
                resolved = resolved.split("/", 1)[1]

        return (resolved,)


NODE_CLASS_MAPPINGS = {
    "Get Model Name from MODEL": GetModelNameFromModel,
}

