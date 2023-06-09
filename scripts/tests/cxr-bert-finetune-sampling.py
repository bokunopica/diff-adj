from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from transformers import AutoTokenizer, AutoModel

model_id = "CompVis/stable-diffusion-v1-4"
results_folder = "results"
device = "cuda"


# components reload
text_encoder = AutoModel.from_pretrained(
    "pretrained_models/cxr-bert-sd-finetune/text_encoder", trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "pretrained_models/cxr-bert-sd-finetune/tokenizer", trust_remote_code=True
)

vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")

unet = UNet2DConditionModel.from_pretrained(
    "pretrained_models/cxr-bert-sd-finetune/unet",
    subfolder="unet",
)

pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
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


for i in range(10):
    image = pipeline(prompt=prompt, height=512, width=512).images[0]
    image.save(f"{results_folder}/cxr-bert-sd-finetune/demo_{'%02d'%i}.png")
# sample_size = model.config.sample_size
