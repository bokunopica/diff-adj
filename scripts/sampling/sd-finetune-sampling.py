import os
from diffusers import StableDiffusionPipeline


if __name__ == "__main__":
    results_folder = "results"
    pretrained_model = "sd-finetune"
    device = "cuda"



    pipe = StableDiffusionPipeline.from_pretrained(
        f"pretrained_models/{pretrained_model}"
    )
    pipe = pipe.to(device)

    prompt = "Focal consolidation at the left lung base, possibly representing aspiration or pneumonia.  Central vascular engorgement."
    # prompt = "Severe cardiomegaly is unchanged."

    if not os.path.exists(f"{results_folder}/{pretrained_model}"):
        os.mkdir(f"{results_folder}/{pretrained_model}")

    for i in range(10):
        image = pipe(prompt=prompt, height=512, width=512).images[0] 
        image.save(f"{results_folder}/sd-finetune/demo_{'%02d'%i}.png")
    # sample_size = model.config.sample_size