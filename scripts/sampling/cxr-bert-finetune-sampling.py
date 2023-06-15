import os
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from transformers import AutoTokenizer, AutoModel
from utils import generate_validation_image, generate_validation_image_with_medclip


if __name__ == "__main__":
    base_model_id = "CompVis/stable-diffusion-v1-4"
    pretrained_model = "cxr-bert-sd-finetune"
    results_folder = "results"
    device = "cuda:1"
    # components reload
    tokenizer = AutoTokenizer.from_pretrained(
        f"pretrained_models/{pretrained_model}/tokenizer",
        trust_remote_code=True,
    )

    text_encoder = AutoModel.from_pretrained(
        f"pretrained_models/{pretrained_model}/text_encoder",
        trust_remote_code=True,
    )

    vae = AutoencoderKL.from_pretrained(
        f"pretrained_models/{pretrained_model}/vae",
        subfolder="vae",
    )

    unet = UNet2DConditionModel.from_pretrained(
        f"pretrained_models/{pretrained_model}/unet",
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    save_path = f"{results_folder}/{pretrained_model}"

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # generate_validation_image(pipe, save_path)
    generate_validation_image_with_medclip(
        pipe,
        save_path,
        device,
        each_samples_per_impression=4,
        length_per_disease=50,
    )
