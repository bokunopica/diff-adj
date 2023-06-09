from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from transformers import BertTokenizer, AutoModel, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

model_id = "CompVis/stable-diffusion-v1-4"
results_folder = "results"
device = "cuda"


# components reload
# text_encoder = AutoModel.from_pretrained(
#     "pretrained_models/RadBERT", trust_remote_code=True
# )

# tokenizer = AutoTokenizer.from_pretrained("StanfordAIMI/RadBERT")

tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

text_encoder = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")

unet = UNet2DConditionModel.from_pretrained(
    "pretrained_models/radbert-sd-finetune/checkpoint-12500/unet",
)

pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    safety_checker=None,
    requires_safety_checker=False,
)
pipeline = pipeline.to(device)

prompt = "Focal consolidation at the left lung base, possibly representing aspiration or pneumonia.  Central vascular engorgement."
# prompt = "Severe cardiomegaly is unchanged."


for i in range(10):
    image = pipeline(prompt=prompt, height=512, width=512).images[0]
    image.save(f"{results_folder}/radbert-sd-finetune/demo_{'%02d'%i}.png")
