# coding: utf-8

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
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
            nn.Linear(config.hidden_sizes[-1] * 2, num_labels_angle, bias=False),
        )
        self.flip_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.hidden_sizes[-1] * 2, num_labels_flip, bias=False),
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

        outputs_1: BaseModelOutputWithPoolingAndNoAttention = self.resnet(
            pixel_values_1,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        outputs_2: BaseModelOutputWithPoolingAndNoAttention = self.resnet(
            pixel_values_2,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output_1 = outputs_1.pooler_output if return_dict else outputs_1[1]
        pooled_output_2 = outputs_2.pooler_output if return_dict else outputs_2[1]
        feature = torch.cat([pooled_output_1, pooled_output_2], dim=1)
        logits_angle: torch.Tensor = self.angle_classifier(feature)
        logits_flip: torch.Tensor = self.flip_classifier(feature)

        loss = None
        if labels is not None and len(labels) == 2:
            angle_loss_fct = CrossEntropyLoss()
            loss_angle = angle_loss_fct(logits_angle.view(-1, self.num_labels_angle), labels[0].view(-1))

            if self.num_labels_flip == 1:
                flip_loss_fct = MSELoss()
                if self.num_labels_flip == 1:
                    loss_flip = flip_loss_fct(logits_flip.squeeze(), labels[1].squeeze())
                else:
                    loss_flip = flip_loss_fct(logits_flip, labels[1])
            elif self.num_labels_flip > 1 and (labels[1].dtype == torch.long or labels[1].dtype == torch.int):
                flip_loss_fct = CrossEntropyLoss()
                loss_flip = flip_loss_fct(logits_flip.view(-1, self.num_labels_flip), labels[1].view(-1))

            loss = 0.5 * loss_angle + 0.5 * loss_flip

        if not return_dict:
            output = (
                (
                    logits_angle,
                    logits_flip,
                )
                + outputs_1[2:]
                + outputs_2[2:]
            )
            return (loss,) + output if loss is not None else output

        return ImageOrientationAngleAndFlipOutputWithNoAttention(
            loss=loss,
            logits_angle=logits_angle,
            logits_flip=logits_flip,
            hidden_states_1=outputs_1.hidden_states,
            hidden_states_2=outputs_2.hidden_states,
        )
