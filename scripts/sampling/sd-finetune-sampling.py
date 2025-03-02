import os
from utils import (
    generate_validation_image,
    generate_four_validation_image,
    generate_validation_image_with_medclip,
    generate_images
)
from diffusers import StableDiffusionPipeline

if __name__ == "__main__":
    results_folder = "results"
    pretrained_model = "60k/sd-finetune"
    device = "cuda:0"

    pipe = StableDiffusionPipeline.from_pretrained(
        f"pretrained_models/{pretrained_model}"
    )
    pipe = pipe.to(device)

    save_path = f"{results_folder}/{pretrained_model}"

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # generate_four_validation_image(pipe, save_path)
    generate_validation_image_with_medclip(
        pipe,
        save_path,
        device,
        each_samples_per_impression=4,
        length_per_disease=50,
    )
    # generate_images(
    #     pipe,
    #     save_path,
    #     "Small right-sided plerual effusion",
    #     20
    # )

    # generate_images(
    #     pipe,
    #     save_path,
    #     "cardiomegaly in this photo",
    #     1
    # )
