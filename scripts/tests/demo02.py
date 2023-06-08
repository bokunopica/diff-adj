import torch
from diffusers import StableDiffusionPipeline

# model_id = "CompVis/stable-diffusion-v1-4"
results_folder = "results"
model_folder = "pretrained_models/sd-finetune-test"
device = "cuda"



pipe = StableDiffusionPipeline.from_pretrained(model_folder)
pipe = pipe.to(device)

# prompt = "Focal consolidation at the left lung base, possibly representing aspiration or pneumonia.  Central vascular engorgement."
prompt = "Severe cardiomegaly is unchanged."


for i in range(10):
    image = pipe(prompt=prompt, height=256, width=256).images[0] 
    image.save(f"{results_folder}/demo_cardiomegaly{'%02d'%i}.png")
# sample_size = model.config.sample_size