import os
import torch
from torch import nn
from medclip import (
    constants,
    MedCLIPTextModel,
    MedCLIPModel,
    MedCLIPVisionModel,
    MedCLIPVisionModelViT,
)
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from transformers import AutoModel, BertTokenizer

base_model_id = "CompVis/stable-diffusion-v1-4"
pretrained_model = "medclip-finetune"
results_folder = "results"
device = "cuda:1"


class MedCLIPTextModelV2(MedCLIPTextModel):
    def __init__(
        self, bert_type=constants.BERT_TYPE, proj_dim=512, proj_bias=False
    ) -> None:
        super().__init__()
        self.bert_type = bert_type
        self.last_n_layer = 4
        self.model = AutoModel.from_pretrained(
            self.bert_type, output_hidden_states=True
        )
        # this tokenizer is actually not used
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_type)
        self.projection_head = nn.Linear(768, proj_dim, bias=proj_bias)


class MedCLIPModelV2(MedCLIPModel):
    def __init__(
        self,
        vision_cls=MedCLIPVisionModel,
        checkpoint=None,
        vision_checkpoint=None,
        logit_scale_init_value=0.07,
    ) -> None:
        super().__init__()
        text_proj_bias = False
        assert vision_cls in [
            MedCLIPVisionModel,
            MedCLIPVisionModelViT,
        ], "vision_cls should be one of [MedCLIPVisionModel, MedCLIPVisionModelViT]"

        self.vision_model = vision_cls(checkpoint=vision_checkpoint)
        self.text_model = MedCLIPTextModelV2(proj_bias=False)

        # learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(1 / logit_scale_init_value))
        )

        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            self.load_state_dict(state_dict)
            print("load model weight from:", checkpoint)


medclip_model = MedCLIPModelV2(checkpoint="pretrained_models/medclip")
text_model = medclip_model.text_model

# components reload
tokenizer = text_model.tokenizer
tokenizer.model_max_length = 256

text_encoder = text_model.model

vae = AutoencoderKL.from_pretrained(
    f"pretrained_models/{pretrained_model}/vae",
)

unet = UNet2DConditionModel.from_pretrained(
    f"pretrained_models/{pretrained_model}/unet",
)

pipeline = StableDiffusionPipeline.from_pretrained(
    base_model_id,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    vae=vae,
    # unet=unet,
    safety_checker=None,
    requires_safety_checker=False,
)
pipeline = pipeline.to(device)

prompt = "Focal consolidation at the left lung base, possibly representing aspiration or pneumonia.  Central vascular engorgement."
# prompt = "Severe cardiomegaly is unchanged."

if not os.path.exists(f"{results_folder}/{pretrained_model}"):
    os.mkdir(f"{results_folder}/{pretrained_model}")

for i in range(10):
    image = pipeline(prompt=prompt, height=512, width=512).images[0]
    image.save(f"{results_folder}/cxr-bert-sd-finetune/demo_{'%02d'%i}.png")
# sample_size = model.config.sample_size
