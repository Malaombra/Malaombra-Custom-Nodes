from nodes import EmptyLatentImage


class ImageRatioLatent:
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "Malaombra-Custom-Nodes/Latent"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "long_side": (["", "512", "1024", "2048", "4096"], {"default": "1024"}),
                "scale": (["x2", "x4", "x8"], {"default": "x2"}),
            }
        }

    @staticmethod
    def _round_to_vae_pixels(value: float) -> int:
        return max(8, int(round(value / 8.0) * 8))

    @staticmethod
    def _parse_scale(scale: str) -> int:
        return int(str(scale).lower().replace("x", ""))

    def generate(self, image, long_side, scale):
        batch_size, height, width = image.shape[:3]

        if long_side:
            long_side = int(long_side)
            if width >= height:
                latent_width = long_side
                latent_height = self._round_to_vae_pixels(long_side * height / width)
            else:
                latent_height = long_side
                latent_width = self._round_to_vae_pixels(long_side * width / height)
        else:
            scale_amount = self._parse_scale(scale)
            latent_width = self._round_to_vae_pixels(width * scale_amount)
            latent_height = self._round_to_vae_pixels(height * scale_amount)

        latent = EmptyLatentImage().generate(
            latent_width,
            latent_height,
            int(batch_size),
        )[0]
        return (latent,)


NODE_CLASS_MAPPINGS = {
    "Malaombra Image Ratio Latent": ImageRatioLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Malaombra Image Ratio Latent": "Image Ratio Latent",
}
