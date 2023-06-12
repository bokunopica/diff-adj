import os
import json

def generate_validation_image(pipe, save_path):
    validation_path = "/run/media/mimic-pa-512/valid"
    metadata_path = f"{validation_path}/metadata.jsonl"
    metadata_list = []
    with open(metadata_path, 'r') as f:
        f.seek(0, 2)
        eof = f.tell()
        f.seek(0, 0)
        while f.tell() < eof:
            metadata_list.append(json.loads(f.readline()))

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