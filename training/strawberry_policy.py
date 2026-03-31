import dataclasses
import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def make_strawberry_example() -> dict:
    return {
        "observation/state": np.random.rand(8).astype(np.float32),
        "observation/cam1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/cam2": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/cam3": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick the ripe strawberry and place it in the box",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class StrawberryInputs(transforms.DataTransformFn):
    """3-camera version: cam1 (right global) + cam2 (wrist) + cam3 (left global)."""
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        cam1_image = _parse_image(data["observation/cam1"])
        cam2_image = _parse_image(data["observation/cam2"])
        cam3_image = _parse_image(data["observation/cam3"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": cam1_image,
                "left_wrist_0_rgb": cam2_image,
                "right_wrist_0_rgb": cam3_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class StrawberryInputs2Cam(transforms.DataTransformFn):
    """2-camera version: cam1 (right global) + cam2 (wrist) only. cam3 slot masked out."""
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        cam1_image = _parse_image(data["observation/cam1"])
        cam2_image = _parse_image(data["observation/cam2"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": cam1_image,
                "left_wrist_0_rgb": cam2_image,
                "right_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class StrawberryOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}
