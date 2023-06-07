import json
import os
import pandas as pd
from tqdm import trange


def get_impression(impression_file):
    result = ""
    with open(impression_file) as f:
        try:
            result = f.read()
            result = result.split('IMPRESSION:')[1].strip()
            result = result.replace('\n', '')
        except Exception as e:
            result = ""
    return result

def get_metadata_json(report_dir, img_dir, meta_csv_location):
    """
    {"file_name": "0001.png", "additional_feature": "This is a first value of a text feature you added to your images"}
    """
    meta_csv = pd.read_csv(meta_csv_location)
    meta_csv = meta_csv[meta_csv['ViewPosition'] == 'AP']
    dicom_id_list = meta_csv['dicom_id'].to_list()
    metadata = []
    for subject_id in os.listdir(img_dir):
        subject_dir = f'{img_dir}{subject_id}'
        if not os.path.isdir(subject_dir):
            continue
        for study_id in os.listdir(subject_dir):
            if(study_id.endswith('html')):
                continue
            study_dir = f'{subject_dir}/{study_id}'
            for dicom_file_name in os.listdir(study_dir):
                if dicom_file_name.endswith('html'):
                    continue
                file_name = f"{study_dir}/{dicom_file_name}"
                file_name = file_name.replace(img_dir, '')
                if dicom_file_name.split('.')[0] in dicom_id_list:
                    impression_file = study_dir.replace(img_dir, report_dir) + '.txt'
                    impression = get_impression(impression_file)
                    metadata.append({
                        'file_name': file_name,
                        "impression": impression,
                        "position": "AP"
                    })
                else:
                    metadata.append({
                        'file_name': file_name,
                        "impression": "",
                        "position": "other"
                    })
    return metadata


if __name__ == "__main__":
    output_file = "/home/qianq/data/mimic_p10_img/metadata.jsonl"
    report_dir = "/home/qianq/data/mimic_p10_report/"
    img_dir = "/home/qianq/data/mimic_p10_img/"
    meta_csv_location = "/home/qianq/data/mimic-cxr-2.0.0-metadata.csv"
    metadata = get_metadata_json(report_dir, img_dir, meta_csv_location)
    with open(output_file, 'w') as f:
        for i in trange(len(metadata)):
            data = metadata[i]
            f.write(json.dumps(data))
            f.write('\n')
    

    # print(os.listdir(img_dir)[:10])
    # os.path.isdir