# coding: utf-8

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.functional import cosine_similarity
from transformers import ResNetConfig, ResNetModel, ResNetPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndNoAttention
from transformers.utils import ModelOutput


@dataclass
class ImageOrientationAngleAndFlipOutputWithNoAttention(ModelOutput):
    """
    Base class for outputs of image classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits_angle (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        logits_flip (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
            called feature maps) of the model at the output of each stage.
    """

    loss: Optional[torch.FloatTensor] = None
    logits_angle: Optional[torch.FloatTensor] = None
    logits_flip: Optional[torch.FloatTensor] = None
    hidden_states_1: Optional[tuple[torch.FloatTensor, ...]] = None
    hidden_states_2: Optional[tuple[torch.FloatTensor, ...]] = None


class OAaFDNet(ResNetPreTrainedModel):
    def __init__(
        self,
        config: ResNetConfig,
        num_labels_angle: int = 4,
        num_labels_flip: int = 2,
    ):
        super().__init__(config)
        self.num_labels_angle = getattr(config, "num_labels_angle", num_labels_angle)
        self.num_labels_flip = getattr(config, "num_labels_flip", num_labels_flip)
        self.resnet = ResNetModel(config)

        # classification head
        self.angle_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.hidden_sizes[-1], config.hidden_sizes[-1], bias=False),
            nn.ReLU(),
            nn.Linear(config.hidden_sizes[-1], num_labels_angle, bias=False),
        )
        self.flip_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.hidden_sizes[-1], config.hidden_sizes[-1], bias=False),
            nn.ReLU(),
            nn.Linear(config.hidden_sizes[-1], num_labels_flip, bias=False),
        )

        # update config
        config.update(
            dict(
                num_labels_angle=num_labels_angle,
                num_labels_flip=num_labels_flip,
            )
        )

        # initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values_1: Optional[torch.FloatTensor] = None,
        pixel_values_2: Optional[torch.FloatTensor] = None,
        pixel_values_3: Optional[torch.FloatTensor] = None,
        pixel_values_4: Optional[torch.FloatTensor] = None,
        labels: Optional[Sequence[torch.LongTensor]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ImageOrientationAngleAndFlipOutputWithNoAttention:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs_1_1: BaseModelOutputWithPoolingAndNoAttention = self.resnet(
            pixel_values_1,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        outputs_1_2: BaseModelOutputWithPoolingAndNoAttention = self.resnet(
            pixel_values_2,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output_1_1 = outputs_1_1.pooler_output if return_dict else outputs_1_1[1]
        pooled_output_1_2 = outputs_1_2.pooler_output if return_dict else outputs_1_2[1]
        feature_1 = pooled_output_1_1 + pooled_output_1_2
        logits_angle_1: torch.Tensor = self.angle_classifier(feature_1)
        logits_flip_1: torch.Tensor = self.flip_classifier(feature_1)

        loss = None
        if labels is not None:
            if len(labels) != 4:
                raise ValueError(f"labels format error, should dim-4, but got dim-{len(labels)}")

            outputs_2_1: BaseModelOutputWithPoolingAndNoAttention = self.resnet(
                pixel_values_3,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            outputs_2_2: BaseModelOutputWithPoolingAndNoAttention = self.resnet(
                pixel_values_4,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            pooled_output_2_1 = outputs_2_1.pooler_output if return_dict else outputs_2_1[1]
            pooled_output_2_2 = outputs_2_2.pooler_output if return_dict else outputs_2_2[1]
            feature_2 = pooled_output_2_1 + pooled_output_2_2
            logits_angle_2: torch.Tensor = self.angle_classifier(feature_2)
            logits_flip_2: torch.Tensor = self.flip_classifier(feature_2)

            angle_loss_fct = CrossEntropyLoss()
            loss_angle_1 = angle_loss_fct(logits_angle_1.view(-1, self.num_labels_angle), labels[0].view(-1))
            loss_angle_2 = angle_loss_fct(logits_angle_2.view(-1, self.num_labels_angle), labels[2].view(-1))

            if self.num_labels_flip == 1:
                flip_loss_fct = MSELoss()
                if self.num_labels_flip == 1:
                    loss_flip_1 = flip_loss_fct(logits_flip_1.squeeze(), labels[1].squeeze())
                    loss_flip_2 = flip_loss_fct(logits_flip_2.squeeze(), labels[3].squeeze())
                else:
                    loss_flip_1 = flip_loss_fct(logits_flip_1, labels[1])
                    loss_flip_2 = flip_loss_fct(logits_flip_2, labels[3])
            elif self.num_labels_flip > 1 and (labels[1].dtype == torch.long or labels[1].dtype == torch.int):
                flip_loss_fct = CrossEntropyLoss()
                loss_flip_1 = flip_loss_fct(logits_flip_1.view(-1, self.num_labels_flip), labels[1].view(-1))
                loss_flip_2 = flip_loss_fct(logits_flip_2.view(-1, self.num_labels_flip), labels[3].view(-1))

            # Contrastive learning loss
            representations_angle = torch.cat([logits_angle_1, logits_angle_2], dim=0)
            representations_flip = torch.cat([logits_flip_1, logits_flip_2], dim=0)
            sim_angle = cosine_similarity(representations_angle.unsqueeze(1), representations_angle.unsqueeze(0), dim=2)
            sim_flip = cosine_similarity(representations_flip.unsqueeze(1), representations_flip.unsqueeze(0), dim=2)
            sim_labels_angle = torch.LongTensor([[1] if l1 == l2 else [0] for l1, l2 in torch.cat(labels[0], labels[2])]).to(
                pixel_values_1.device
            )
            sim_labels_flip = torch.LongTensor([[1] if l1 == l2 else [0] for l1, l2 in torch.cat(labels[1], labels[3])]).to(
                pixel_values_1.device
            )

            sim_loss_fct = CrossEntropyLoss()
            sim_loss = sim_loss_fct(sim_angle, sim_labels_angle) + sim_loss_fct(sim_flip, sim_labels_flip)

            loss = loss_angle_1 + loss_angle_2 + loss_flip_1 + loss_flip_2 + sim_loss

        if not return_dict:
            output = (
                (
                    logits_angle_1,
                    logits_flip_1,
                )
                + outputs_1_1[2:]
                + outputs_1_2[2:]
            )
            return (loss,) + output if loss is not None else output

        return ImageOrientationAngleAndFlipOutputWithNoAttention(
            loss=loss,
            logits_angle=logits_angle_1,
            logits_flip=logits_flip_1,
            hidden_states_1=outputs_1_1.hidden_states,
            hidden_states_2=outputs_1_2.hidden_states,
        )
