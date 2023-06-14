import os
from transformers import AutoTokenizer
from transformers import CLIPProcessor, CLIPImageProcessor
import torch
from torch import nn
from transformers import AutoModel, BertTokenizer
from medclip import (
    constants,
    MedCLIPTextModel,
    MedCLIPModel,
    MedCLIPVisionModel,
    MedCLIPVisionModelViT,
)


class MedCLIPProcessorV2(CLIPProcessor):
    """
    A processor that takes input images and texts and provides inputs for
    `MedCLIPModel`.
    """

    feature_extractor_class = "CLIPFeatureExtractor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

    def __init__(self, image_size):
        feature_extractor = CLIPImageProcessor(size=image_size)
        tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        tokenizer.model_max_length = 77
        super().__init__(feature_extractor, tokenizer)


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
        device,
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

        self.device = device

    def encode_text(self, input_ids=None, attention_mask=None):
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        text_embeds = self.text_model(input_ids, attention_mask)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        return_loss=None,
        **kwargs,
    ):
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        pixel_values = pixel_values.to(self.device)

        img_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)

        logits_per_image = self.compute_logits(img_embeds, text_embeds)
        logits_per_text = logits_per_image.t()

        if return_loss:
            loss = self.clip_loss(logits_per_text)
        else:
            loss = None

        return {
            "img_embeds": img_embeds,
            "text_embeds": text_embeds,
            "logits": logits_per_image,
            "loss_value": loss,
            "logits_per_text": logits_per_text,
        }
