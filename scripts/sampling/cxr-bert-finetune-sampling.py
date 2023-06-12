import os
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from transformers import AutoTokenizer, AutoModel



if __name__ == "__main__":
    base_model_id = "CompVis/stable-diffusion-v1-4"
    pretrained_model = "cxr-bert-sd-finetune"
    results_folder = "results"
    device = "cuda"
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

    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(device)

    prompt = "Focal consolidation at the left lung base, possibly representing aspiration or pneumonia.  Central vascular engorgement."
    # prompt = "Severe cardiomegaly is unchanged."

    if not os.exists(f"{results_folder}/{pretrained_model}"):
        os.mkdir(f"{results_folder}/{pretrained_model}")



    for i in range(10):
        image = pipeline(prompt=prompt, height=512, width=512).images[0]
        image.save(f"{results_folder}/cxr-bert-sd-finetune/demo_{'%02d'%i}.png")
    # sample_size = model.config.sample_size
