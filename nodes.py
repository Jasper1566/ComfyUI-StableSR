import os

import torch
from torch import Tensor

import comfy.model_management
import comfy.sample
import folder_paths

from .modules.colorfix import adain_color_fix, wavelet_color_fix
from .modules.spade import SPADELayers
from .modules.struct_cond import EncoderUNetModelWT, build_unetwt
from .modules.util import pil2tensor, tensor2pil

model_path = folder_paths.models_dir
folder_name = "stablesr"
folder_path = os.path.join(
    model_path, "stablesr"
)  # set a default path for the common comfyui model path
if folder_name in folder_paths.folder_names_and_paths:
    folder_path = folder_paths.folder_names_and_paths[folder_name][0][
        0
    ]  # if a custom path was set in extra_model_paths.yaml then use it
folder_paths.folder_names_and_paths["stablesr"] = (
    [folder_path],
    folder_paths.supported_pt_extensions,
)


class StableSRColorFix:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_map_image": ("IMAGE",),
                "color_fix": (
                    [
                        "Wavelet",
                        "AdaIN",
                    ],
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fix_color"
    CATEGORY = "image"

    def fix_color(self, image, color_map_image, color_fix):
        print(f"[StableSR] fix_color")
        try:
            color_fix_func = (
                wavelet_color_fix if color_fix == "Wavelet" else adain_color_fix
            )
            result_image = color_fix_func(
                tensor2pil(image), tensor2pil(color_map_image)
            )
            refined_image = pil2tensor(result_image)
            return (refined_image,)
        except Exception as e:
            print(f"[StableSR] Error fix_color: {e}")


original_sample = comfy.sample.sample
SAMPLE_X = None


def hook_sample(*args, **kwargs):
    global SAMPLE_X
    if len(args) >= 9:
        SAMPLE_X = args[8]
    elif "latent_image" in kwargs:
        SAMPLE_X = kwargs["latent_image"]
    return original_sample(*args, **kwargs)


comfy.sample.sample = hook_sample


class StableSR:
    """
    Initializes a StableSR model.

    Args:
        path: The path to the StableSR checkpoint file.
        dtype: The data type of the model. If not specified, the default data type will be used.
        device: The device to run the model on. If not specified, the default device will be used.
    """

    def __init__(self, stable_sr_model_path, dtype, device):
        print(f"[StbaleSR] in StableSR init - dtype: {dtype}, device: {device}")
        state_dict = comfy.utils.load_torch_file(stable_sr_model_path)

        self.struct_cond_model: EncoderUNetModelWT = build_unetwt(
            use_fp16=dtype == torch.float16
        )
        self.spade_layers: SPADELayers = SPADELayers()
        self.struct_cond_model.load_from_dict(state_dict)
        self.spade_layers.load_from_dict(state_dict)
        del state_dict

        self.dtype = dtype
        self.struct_cond_model.apply(lambda x: x.to(dtype=dtype, device=device))
        self.spade_layers.apply(lambda x: x.to(dtype=dtype, device=device))
        self.latent_image: Tensor = None
        self.set_image_hooks = {}
        self.struct_cond: Tensor = None

        self.auto_set_latent = False
        self.last_t = 0.0

    def set_latent_image(self, latent_image):
        self.latent_image = latent_image

    def set_auto_set_latent(self, auto_set_latent):
        self.auto_set_latent = auto_set_latent

    def __call__(self, model_function, params):
        # explode packed args
        input_x = params.get("input")
        timestep = params.get("timestep")
        c = params.get("c")

        t = model_function.__self__.model_sampling.timestep(timestep)

        if self.auto_set_latent:
            tt = float(t[0])
            if self.last_t <= tt:
                latent_image = model_function.__self__.process_latent_in(SAMPLE_X)
                self.set_latent_image(latent_image)
            self.last_t = tt

        # set latent image to device
        device = input_x.device
        latent_image = self.latent_image.to(dtype=self.dtype, device=device)

        # Ensure the device of all modules layers is the same as the unet
        # This will fix the issue when user use --medvram or --lowvram
        self.spade_layers.to(device)
        self.struct_cond_model.to(device)

        self.struct_cond = None  # mitigate vram peak
        self.struct_cond = self.struct_cond_model(
            latent_image, t[: latent_image.shape[0]]
        )

        self.spade_layers.hook(
            model_function.__self__.diffusion_model, lambda: self.struct_cond
        )

        # Call the model_function with the provided arguments
        result = model_function(input_x, timestep, **c)

        self.spade_layers.unhook()

        # Return the result
        return result

    def to(self, device):
        if type(device) == torch.device:
            self.struct_cond_model.apply(lambda x: x.to(device=device))
            self.spade_layers.apply(lambda x: x.to(device=device))
        return self


class ApplyStableSRUpscaler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "stablesr_model": (folder_paths.get_filename_list("stablesr"),),
            },
            "optional": {
                "latent_image": ("LATENT",),
            },
        }

    RETURN_TYPES = ("MODEL",)

    FUNCTION = "apply_stable_sr_upscaler"
    CATEGORY = "image/upscaling"

    def apply_stable_sr_upscaler(self, model, stablesr_model, latent_image=None):
        stablesr_model_path = folder_paths.get_full_path("stablesr", stablesr_model)
        if not os.path.isfile(stablesr_model_path):
            raise Exception(f"[StableSR] Invalid StableSR model reference")

        upscaler = StableSR(
            stablesr_model_path, dtype=comfy.model_management.unet_dtype(), device="cpu"
        )
        if latent_image != None:
            latent_image = model.model.process_latent_in(latent_image["samples"])
            upscaler.set_latent_image(latent_image)
        else:
            upscaler.set_auto_set_latent(True)

        model_sr = model.clone()
        model_sr.set_model_unet_function_wrapper(upscaler)
        return (model_sr,)


NODE_CLASS_MAPPINGS = {
    "StableSRColorFix": StableSRColorFix,
    "ApplyStableSRUpscaler": ApplyStableSRUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StableSRColorFix": "StableSRColorFix",
    "ApplyStableSRUpscaler": "ApplyStableSRUpscaler",
}
