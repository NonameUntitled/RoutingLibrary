from utils import MetaParent

import torch
import torch.nn as nn


class BaseLoss(metaclass=MetaParent):
    pass


class TorchLoss(BaseLoss, nn.Module):
    pass


class IdentityLoss(BaseLoss, config_name='identity'):

    def __call__(self, inputs):
        return inputs


class CompositeLoss(TorchLoss, config_name='composite'):

    def __init__(self, losses, weights=None, output_prefix=None):
        super().__init__()
        self._losses = losses
        self._weights = weights or [1.0] * len(losses)
        self._output_prefix = output_prefix

    @classmethod
    def create_from_config(cls, config, **kwargs):
        losses = []
        weights = []

        for loss_cfg in config['losses']:
            weight = loss_cfg.pop('weight') if 'weight' in loss_cfg else 1.0
            loss_function = BaseLoss.create_from_config(loss_cfg)

            weights.append(weight)
            losses.append(loss_function)

        return cls(losses=losses, weights=weights, output_prefix=config.get('output_prefix'))

    def forward(self, inputs):
        total_loss = 0.0
        for loss, weight in zip(self._losses, self._weights):
            total_loss += weight * loss(inputs)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = total_loss.cpu().item()

        return total_loss


class CrossEntropyLoss(TorchLoss, config_name='cross_entropy'):

    def __init__(self, predictions_prefix, labels_prefix, output_prefix=None):
        super().__init__()
        self._pred_prefix = predictions_prefix
        self._labels_prefix = labels_prefix
        self._output_prefix = output_prefix

        self._loss = nn.CrossEntropyLoss()

    def forward(self, inputs):
        all_logits = inputs[self._pred_prefix]  # (all_items, num_classes)
        all_labels = inputs[self._labels_prefix]  # (all_items)
        assert all_logits.shape[0] == all_labels.shape[0]
        loss = self._loss(all_logits.float(), all_labels)  # (1)
        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class BinaryCrossEntropyLoss(TorchLoss, config_name='binary_cross_entropy'):

    def __init__(
            self,
            predictions_prefix,
            labels_prefix,
            with_logits=True,
            output_prefix=None
    ):
        super().__init__()
        self._pred_prefix = predictions_prefix
        self._labels_prefix = labels_prefix
        self._output_prefix = output_prefix

        if with_logits:
            self._loss = nn.BCEWithLogitsLoss()
        else:
            self._loss = nn.BCELoss()

    def forward(self, inputs):
        all_logits = inputs[self._pred_prefix].float()  # (all_items)
        all_labels = inputs[self._labels_prefix].float()  # (all_items)
        assert all_logits.shape[0] == all_labels.shape[0]

        loss = self._loss(all_logits, all_labels)  # (1)
        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class MSELoss(TorchLoss, config_name='mse'):

    def __init__(
            self,
            predictions_prefix,
            ground_truth_prefix,
            output_prefix=None
    ):
        super().__init__()
        self._predictions_prefix = predictions_prefix
        self._ground_truth_prefix = ground_truth_prefix
        self._output_prefix = output_prefix

    def forward(self, inputs):
        predictions = inputs[self._predictions_prefix]
        ground_truth = inputs[self._ground_truth_prefix]

        loss = torch.nn.functional.mse_loss(predictions, ground_truth)  # (1)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss
