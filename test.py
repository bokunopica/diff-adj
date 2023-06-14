from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from PIL import Image
# from transformers.image_transforms import convert_to_rgb

# prepare for the demo image and texts
processor = MedCLIPProcessor()
# processor.feature_extractor.convert_rgb = convert_to_rgb
image = Image.open('/home/qianq/mycodes/diff-adj/results/random/cxr-bert-sd-finetune/2c6767f3-38084bdb-4c09377c-51ca8ed8-6c932992.jpg')
inputs = processor(
    text=["lungs remain severely hyperinflated with upper lobe emphysema", 
        "opacity left costophrenic angle is new since prior exam ___ represent some loculated fluid cavitation unlikely"], 
    images=image, 
    return_tensors="pt", 
    padding=True
)

# pass to MedCLIP model
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()
model.cuda()
outputs = model(**inputs)
print(outputs.keys())