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
from utils import generate_validation_image, generate_validation_image_with_medclip



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




if __name__ == "__main__":
    base_model_id = "CompVis/stable-diffusion-v1-4"
    pretrained_model = "medclip-finetune"
    results_folder = "results"
    device = "cuda:1"

    # components reload
    medclip_model = MedCLIPModelV2(checkpoint="pretrained_models/medclip/medclip-pretrained")
    text_model = medclip_model.text_model
    tokenizer = text_model.tokenizer
    tokenizer.model_max_length = 256

    text_encoder = text_model.model

    vae = AutoencoderKL.from_pretrained(
        f"pretrained_models/{pretrained_model}/vae",
    )

    unet = UNet2DConditionModel.from_pretrained(
        f"pretrained_models/{pretrained_model}/unet",
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    save_path = f"{results_folder}/{pretrained_model}"

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    # generate_validation_image(pipe, save_path)
    generate_validation_image_with_medclip(
        pipe,
        save_path,
        device,
        each_samples_per_impression=4,
        length=30,
    )
