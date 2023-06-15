import os
import pandas as pd
import json
from medclip_adjustment import MedCLIPProcessorV2, MedCLIPModelV2
from medclip import (
    MedCLIPModel,
    MedCLIPVisionModelViT,
)
import torch
from torch import softmax
import random


DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


def read_jsonl(file_path):
    result_list = []
    with open(file_path, "r") as f:
        f.seek(0, 2)
        eof = f.tell()
        f.seek(0, 0)
        while f.tell() < eof:
            result_list.append(json.loads(f.readline()))
    return result_list


def generate_validation_image(pipe, save_path):
    validation_path = "/run/media/mimic-pa-512/valid"
    metadata_path = f"{validation_path}/metadata.jsonl"
    metadata_list = read_jsonl(metadata_path)

    length = len(metadata_list)
    i = 1
    for metadata in metadata_list:
        impression = metadata.get("impression", "")
        file_name = metadata.get("file_name", "")
        full_file_path = f"{validation_path}/{file_name}"
        if not os.path.exists(full_file_path):
            continue
        image = pipe(prompt=impression, height=512, width=512).images[0]
        image.save(f"{save_path}/{file_name}")
        print(f"{i}/{length} image saved...")
        i += 1


def generate_four_validation_image(pipe, save_path):
    validation_path = "/run/media/mimic-pa-512/valid"
    metadata_path = f"{validation_path}/metadata.jsonl"
    metadata_list = read_jsonl(metadata_path)

    length = len(metadata_list)
    i = 1
    for metadata in metadata_list:
        impression = metadata.get("impression", "")
        file_name = metadata.get("file_name", "")
        full_file_path = f"{validation_path}/{file_name}"
        if not os.path.exists(full_file_path):
            continue
        for j in range(4):
            image = pipe(prompt=impression, height=512, width=512).images[0]
            split_file_name = file_name.split(".")
            image.save(
                f"{save_path}/{split_file_name[0]}_{'%2d'%j}.{split_file_name[1]}"
            )
        print(f"{i}/{length} image saved...")
        i += 1


def generate_validation_image_with_medclip(
    pipe,
    save_path,
    device,
    each_samples_per_impression=4,
    length=None,
    start=0,
    length_per_disease=None,
):
    validation_path = "/run/media/mimic-pa-512/valid"
    metadata_path = f"{validation_path}/metadata.jsonl"
    full_metadata_list = read_jsonl(metadata_path)
    random.seed(111)
    random.shuffle(full_metadata_list)

    if length_per_disease:
        # 每个疾病挑选50种
        metadata_list = []
        for metadata in full_metadata_list:
            df_label = pd.read_csv("/run/media/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-negbio.csv")
            df_dicom = pd.read_csv("/run/media/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv")
            
            pass
        length = len(metadata_list)
    else:
        metadata_list = full_metadata_list
        length = length if length else len(metadata_list)

    # medclip processor and model
    processor = MedCLIPProcessorV2(image_size=512)
    model = MedCLIPModelV2(
        device=device,
        vision_cls=MedCLIPVisionModelViT,
        checkpoint="pretrained_models/medclip/medclip-vit-pretrained",
    )
    model.vision_model.to(device)
    model.to(device)
    for i in range(start, length):
        metadata = metadata_list[i]
        impression = metadata.get("impression", "")
        file_name = metadata.get("file_name", "")
        full_file_path = f"{validation_path}/{file_name}"
        if not os.path.exists(full_file_path):
            continue
        gen_image_list = []
        for j in range(each_samples_per_impression):
            # PIL.Image RGB mode
            image = pipe(prompt=impression, height=512, width=512).images[0]
            gen_image_list.append(image)
            split_file_name = file_name.split(".")
            image.save(
                f"{save_path}/{split_file_name[0]}_{'%d'%j}.{split_file_name[1]}"
            )

        inputs = processor(
            text=[f"A photo of a chest xray"],
            images=gen_image_list,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_text = outputs["logits_per_text"]
            results = softmax(logits_per_text, dim=1).tolist()

        _idx = results[0].index(max(results[0]))
        save_image = gen_image_list[_idx]
        save_image.save(f"{save_path}/{file_name}")

        print(f"{i+1}/{length} image saved...")
