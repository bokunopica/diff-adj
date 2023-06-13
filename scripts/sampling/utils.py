import os
import json

def read_jsonl(file_path):
    result_list = []
    with open(file_path, 'r') as f:
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
        impression = metadata.get('impression', '')
        file_name = metadata.get('file_name', '')
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
        impression = metadata.get('impression', '')
        file_name = metadata.get('file_name', '')
        full_file_path = f"{validation_path}/{file_name}"
        if not os.path.exists(full_file_path):
            continue
        for j in range(4):
            image = pipe(prompt=impression, height=512, width=512).images[0]
            split_file_name = file_name.split('.')
            image.save(f"{save_path}/{split_file_name[0]}_{'%2d'%j}.{split_file_name[1]}")
        print(f"{i}/{length} image saved...")
        i += 1