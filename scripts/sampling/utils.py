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
    validation_path = "/home/qianq/data/mimic-pa-512/mimic-pa-512/valid"
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
    validation_path = "/home/qianq/data/mimic-pa-512/mimic-pa-512/valid"
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
    """
    params:
        pipe: model
        save_path: 保存路径
        device: gpu
        each_samples_per_impression: 每个impression对应生成多少张图像供medclip做滤波
        start: 数据偏移量
        length: 数据个数
        length_per_disease: 可选 每种疾病的个数
    """
    validation_path = "/home/qianq/data/mimic-pa-512/mimic-pa-512/valid"
    metadata_path = f"{validation_path}/metadata.jsonl"
    full_metadata_list = read_jsonl(metadata_path)
    random.seed(222)
    random.shuffle(full_metadata_list)
    random.shuffle(DISEASES)
    df_meta = pd.read_csv(metadata_path.replace('jsonl', 'csv'))


    if length_per_disease:
        # 每个疾病挑选50种
        metadata_list = []
        df_meta.head()
        existed_file_name_list = []
        for disease in DISEASES:
            single_disease_file_name_list = df_meta[df_meta[disease]==True]['file_name'].to_list()
            random.shuffle(single_disease_file_name_list)
            i = 0
            for file_name in single_disease_file_name_list:
                if file_name not in existed_file_name_list:
                    existed_file_name_list.append(file_name)
                    row = df_meta[df_meta['file_name']==file_name]
                    metadata_list.append({
                        "file_name": file_name,
                        "impression": row['impression'].to_list()[0],
                        "disease": disease
                    })
                    i+=1
                if i>=length_per_disease:
                    break
        length = len(metadata_list)
    else:
        metadata_list = full_metadata_list
        length = length if length else len(metadata_list)

    df_meta_filtered = pd.DataFrame(metadata_list)
    df_meta_filtered.to_csv(f"{save_path}/metadata.csv")

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
        disease = metadata.get("disease")
        disease_file_name = f"{disease}_{file_name}"
        full_file_path = f"{validation_path}/{file_name}"
        if not os.path.exists(full_file_path):
            continue
        gen_image_list = []
        for j in range(each_samples_per_impression):
            # PIL.Image RGB mode
            image = pipe(prompt=impression, height=512, width=512).images[0]
            gen_image_list.append(image)
            split_file_name = disease_file_name.split(".")
            image.save(
                f"{save_path}/{split_file_name[0]}_{'%d'%j}.{split_file_name[1]}"
            )

        inputs = processor(
            text=[
                f"A photo of a chest xray",
                f"{impression}"
            ],
            images=gen_image_list,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_text = outputs["logits_per_text"].tolist()
            result_static = logits_per_text[0]
            result_impression = logits_per_text[1]

        # TODO 这里做medclip logits的阈值处理
        for balabal in result_static:
            if balabal > 0.5:
                # 处理
                pass
        _idx_static = result_static.index(max(result_static))
        _idx_impression = result_impression.index(max(result_impression))
        static_prompt_image = gen_image_list[_idx_static]
        impression_prompt_image = gen_image_list[_idx_impression]

        split_file_name = disease_file_name.split(".")
        static_prompt_image.save(f"{save_path}/{split_file_name[0]}_static.{split_file_name[1]}")
        impression_prompt_image.save(f"{save_path}/{split_file_name[0]}_impression.{split_file_name[1]}")

        print(f"{i+1}/{length} image saved...")



def generate_images(pipe, save_path, impression, length):
    for i in range(length):
        file_name = "example_%s_%03d.png" % (impression, i)
        image = pipe(prompt=impression, height=512, width=512).images[0]
        image.save(f"{save_path}/{file_name}")
        print(f"{i+1}/{length} image saved...")