MAX_UINT64 = 0xFFFFFFFFFFFFFFFF


class SeedGenerator:
    RETURN_TYPES = ("INT",)
    FUNCTION = "get_seed"
    CATEGORY = "ImageSaverTools/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_UINT64}),
            }
        }

    def get_seed(self, seed: int):
        return (int(seed),)


NODE_CLASS_MAPPINGS = {
    "Seed Generator": SeedGenerator,
}
