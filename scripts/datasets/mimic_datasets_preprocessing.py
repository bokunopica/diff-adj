import os
import pandas as pd
import json
from transformers import CLIPTokenizer

def get_impression(impression_file):
    result = ""
    with open(impression_file, 'r') as f:
        try:
            result = f.read()
            result = result.split('IMPRESSION:')[1].strip()
            result = result.replace('\n', '')
        except Exception as e:
            result = ""
    return result

def remove_metadata(output_dir):
    file_path = f"{output_dir}metadata.jsonl"
    if os.path.exists(file_path):
        os.remove(file_path)

def write_metadata(file_name, output_dir, impression):
    output_meta_dir = f"{output_dir}metadata.jsonl"
    metadata = {
        "file_name": file_name,
        "impression": impression
    }
    with open(output_meta_dir, 'a') as f:
        f.write(json.dumps(metadata))
        f.write('\n')

def write_to_output(base_dir, file_name, output_dir):
    with open(f'{base_dir}/{file_name}', 'rb') as f:
        img = f.read()
    with open(f'{output_dir}{file_name}', 'wb') as f:
        f.write(img)


def main():
    output_dir_train = "/run/media/mimic-pa/train/"
    output_dir_valid = "/run/media/mimic-pa/valid/"
    raw_input_dir = "/run/media/mimic-cxr-jpg/2.0.0/files/"
    meta_csv_location = "/home/qianq/data/mimic-cxr-2.0.0-metadata.csv"

    if not os.path.exists(output_dir_train):
        os.mkdir(output_dir_train)
    if not os.path.exists(output_dir_valid):
        os.mkdir(output_dir_valid)

    remove_metadata(output_dir_train)
    remove_metadata(output_dir_valid)

    tokenizer = CLIPTokenizer.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        subfolder="tokenizer",
    )

    # pa meta csv
    metadata_df = pd.read_csv(meta_csv_location)
    metadata_df = metadata_df[metadata_df['ViewPosition'] == 'PA']
    dicom_id_list = metadata_df['dicom_id'].to_list()

    for base_dir, dir_name_list, file_name_list in os.walk(raw_input_dir):
        if base_dir.find("/p19/") == -1:
            output_dir = output_dir_train
        else:
            output_dir = output_dir_valid
        
        for file_name in file_name_list:
            print(base_dir, file_name, end='\r')
            if file_name.endswith('jpg'):
                dicom_id = file_name.split('.')[0]
                if dicom_id not in dicom_id_list:
                    continue
                meta_base_dir = base_dir.replace('/mimic-cxr-jpg/2.0.0/', '/mimic-cxr-reports/')
                report_txt_path = meta_base_dir + ".txt"
                impression = get_impression(report_txt_path)
                outputs = tokenizer(impression)
                len_token =  len(outputs['input_ids'])
                if len_token < 7 or len_token > 77:
                    continue
                write_metadata(file_name, output_dir, impression)
                write_to_output(base_dir, file_name, output_dir)


if __name__ == "__main__":
    main()