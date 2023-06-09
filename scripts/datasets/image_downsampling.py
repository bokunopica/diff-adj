import os
from PIL import Image
from tqdm import trange

BASE_DIR = "/run/media"
SAVE_DIR = "/run/media/mimic-pa-512"
IMAGE_FOLDER = "mimic-pa"
RESAMPLE_FOLDER_ARRAY = ["valid", "train"]
# RESAMPLE_FOLDER_ARRAY = ["valid"]

RESAMPLE_SIZE = 512
SAVE_FOLDER = f"/run/media"


def resample_img(source_dir, save_dir, img_name):
    img_path = f"{source_dir}/{img_name}"
    save_path = f"{save_dir}/{img_name}"

    if os.path.exists(save_dir):
        if os.path.exists(save_path):
            os.remove(save_path)
    else:
        os.makedirs(save_dir)

    image = Image.open(img_path)

    if image.height >= image.width:
        ratio = image.width/512
        resize_height = int(round(image.height/ratio, 0))
        resize_width = 512
    else:
        ratio = image.height/512
        resize_width = int(round(image.width/ratio, 0))
        resize_height = 512

    image = image.resize(
        size=(resize_width, resize_height), # (width, height)
        # resample=Image.Resampling.NEAREST
    )
    image.save(save_path)


def main():
    for resample_folder in RESAMPLE_FOLDER_ARRAY:
        src_dir = f"{BASE_DIR}/{IMAGE_FOLDER}/{resample_folder}"
        dst_dir = f"{SAVE_DIR}/{resample_folder}"
        file_name_list = os.listdir(src_dir)
        for i in trange(len(file_name_list)):
            file_name = file_name_list[i]
            if file_name.endswith('jpg'):
                resample_img(src_dir, dst_dir, file_name)


if __name__ == "__main__":
    main()
