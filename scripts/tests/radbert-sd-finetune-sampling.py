import torch
from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import (
    BertModel,
    AutoTokenizer,
    PreTrainedTokenizer
)


# model_id = "CompVis/stable-diffusion-v1-4"
results_folder = "results"
model_folder = "pretrained_models/radbert-sd-finetune"
device = "cuda"

# text_encoder = BertModel.from_pretrained("pretrained_models/RadBERT").cuda()
# vae = AutoencoderKL.from_pretrained(
#     model_folder,
#     subfolder="vae",
# ).to()
# unet = UNet2DConditionModel.from_pretrained(
#     model_folder,
#     subfolder="unet",
# ).cuda()
tokenizer = AutoTokenizer.from_pretrained("pretrained_models/RadBERT")
pipe = StableDiffusionPipeline.from_pretrained(
    model_folder,
    # tokenizer=tokenizer
    # text_encoder=text_encoder,
    # vae=vae,
    # unet=unet,
    # safety_checker=None,
    # requires_safety_checker=False,
)
pipe = pipe.to('cpu')

prompt = "Focal consolidation at the left lung base, possibly representing aspiration or pneumonia.  Central vascular engorgement."
# prompt = "Severe cardiomegaly is unchanged."


for i in range(10):
    image = pipe(prompt=prompt, height=512, width=512).images[0]
    image.save(f"{results_folder}/radbert-sd-finetune/demo_{'%02d'%i}.png")
# sample_size = model.config.sample_size
