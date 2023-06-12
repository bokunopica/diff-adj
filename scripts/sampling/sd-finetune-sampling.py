import os
from utils import generate_validation_image
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

    save_path = f"{results_folder}/{pretrained_model}"

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    generate_validation_image(pipe, save_path)