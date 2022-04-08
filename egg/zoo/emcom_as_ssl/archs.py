# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn
import torchvision

from egg.core.continous_communication import SenderReceiverContinuousCommunication
from egg.core.gs_wrappers import gumbel_softmax_sample


def get_vision_module(name: str = "resnet50", pretrained: bool = False):
    modules = {
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "resnet101": torchvision.models.resnet101(pretrained=pretrained),
        "resnet152": torchvision.models.resnet152(pretrained=pretrained),
    }
    if name not in modules:
        raise KeyError(f"{name} is not currently supported.")

    model = modules[name]

    n_features = model.fc.in_features
    model.fc = nn.Identity()

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
        model = model.eval()

    return model, n_features


def get_vision_modules(
    encoder_arch: str, shared: bool = False, pretrain_vision: bool = False
):
    if pretrain_vision:
        assert (
            shared
        ), "A pretrained not shared vision_module is a waste of memory. Please run with --shared set"

    encoder, features_dim = get_vision_module(encoder_arch, pretrain_vision)
    encoder_recv = None
    if not shared:
        encoder_recv, _ = get_vision_module(encoder_arch)

    return encoder, encoder_recv, features_dim


class VisionModule(nn.Module):
    def __init__(
        self,
        sender_vision_module: nn.Module,
        receiver_vision_module: Optional[nn.Module] = None,
        use_hooks: bool = True,
    ):
        super(VisionModule, self).__init__()

        self.encoder = sender_vision_module

        self.shared = receiver_vision_module is None
        if not self.shared:
            self.encoder_recv = receiver_vision_module

        self.use_hooks = use_hooks
        self.result = {}

        if self.use_hooks:
            self.register_hooks(self.encoder)

            if not self.shared:
                self.register_hooks(self.encoder_recv)

    def register_hooks(self, model):
        def save_outputs_hook(layer_id):
            def fn(_m, _input, output):
                output = output.view(output.size(0), -1)
                self.result[layer_id] = output
            return fn

        model.layer1.register_forward_hook(save_outputs_hook(1))
        model.layer2.register_forward_hook(save_outputs_hook(2))
        model.layer3.register_forward_hook(save_outputs_hook(3))
        model.layer4.register_forward_hook(save_outputs_hook(4))
        model.fc.register_forward_hook(save_outputs_hook(5))

    def forward(self, x_i, x_j):
        encoded_input_sender = self.encoder(x_i)
        if self.use_hooks:
            encoded_input_sender = self.result.copy()

        if self.shared:
            encoded_input_recv = self.encoder(x_j)
        else:
            encoded_input_recv = self.encoder_recv(x_j)
        if self.use_hooks:
            encoded_input_recv = self.result.copy()

        return encoded_input_sender, encoded_input_recv


class VisionGameWrapper(nn.Module):
    def __init__(
        self,
        game: nn.Module,
        vision_module: nn.Module,
    ):
        super(VisionGameWrapper, self).__init__()
        self.game = game
        self.vision_module = vision_module

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        x_i, x_j = sender_input
        sender_encoded_input, receiver_encoded_input = self.vision_module(x_i, x_j)

        return self.game(
            sender_input=sender_encoded_input,
            labels=labels,
            receiver_input=receiver_encoded_input,
        )


class SimCLRSender(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        discrete_evaluation: bool = False,
    ):
        super(SimCLRSender, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=False)

        self.discrete_evaluation = discrete_evaluation

    def forward(self, resnet_output, sender=False):
        first_projection = self.fc(resnet_output)

        if self.discrete_evaluation and (not self.training) and sender:
            logits = first_projection
            size = logits.size()
            indexes = logits.argmax(dim=-1)
            one_hot = torch.zeros_like(logits).view(-1, size[-1])
            one_hot.scatter_(1, indexes.view(-1, 1), 1)
            one_hot = one_hot.view(*size)
            first_projection = one_hot

        out = self.fc_out(first_projection)
        return out, first_projection.detach(), resnet_output.detach()


class EmSSLSender(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        temperature: float = 1.0,
        trainable_temperature: bool = False,
        straight_through: bool = False,
    ):
        super(EmSSLSender, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True
            )
        self.straight_through = straight_through

        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, resnet_output):
        first_projection = self.fc(resnet_output)
        message = gumbel_softmax_sample(
            first_projection, self.temperature, self.training, self.straight_through
        )
        out = self.fc_out(message)
        return out, message.detach(), resnet_output.detach()


class Receiver(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 2048):
        super(Receiver, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, _x, resnet_output):
        return self.fc(resnet_output), resnet_output.detach()


class FixedLengthFCNSender(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        temperature: float = 1.0,
        trainable_temperature: bool = False,
        straight_through: bool = False,
        shared_embedding: bool = False,
        nos: int = 4,
        structured_comm: bool = False,
    ):
        super(FixedLengthFCNSender, self).__init__()

        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True
            )
        self.straight_through = straight_through
        self.shared_embedding = shared_embedding
        self.structured_comm = structured_comm

        self.nos = nos

        self.fc_in_layers = []
        dims = [802816, 401408, 200704, 100352, input_dim]
        for i in range(self.nos):
            if self.structured_comm:
                in_layer = nn.Sequential(
                    nn.Linear(dims[i], hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                )
            else:
                in_layer = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                )
            self.fc_in_layers.append(in_layer)
        self.fc_in_layers = nn.ModuleList(self.fc_in_layers)

        if self.shared_embedding:
            self.fc_out = nn.Linear(hidden_dim, output_dim, bias=False)
        else:
            self.fc_out_layers = []
            for _ in range(self.nos):
                self.fc_out_layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
            self.fc_out_layers = nn.ModuleList(self.fc_out_layers)

    def forward(self, resnet_output):
        final_message, messages = [], []
        for i in range(self.nos):
            if self.structured_comm:
                first_projection = self.fc_in_layers[i](resnet_output[i+1])
            else:
                first_projection = self.fc_in_layers[i](resnet_output)
            message = gumbel_softmax_sample(
                first_projection, self.temperature, self.training, self.straight_through
            )
            if self.shared_embedding:
                out = self.fc_out(message)
            else:
                out = self.fc_out_layers[i](message)
            messages.append(message)
            final_message.append(out)
        messages = torch.concat(messages, dim=1)
        out = torch.concat(final_message, dim=1)
        if self.structured_comm:
            resnet_output = resnet_output[5]
        return out, messages.detach(), resnet_output.detach()


class FixedLengthFCNReceiver(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
        nos: int = 4,
        structured_comm: bool = False
    ):
        super(FixedLengthFCNReceiver, self).__init__()
        self.fc_out = []
        self.nos = nos
        self.structured_comm = structured_comm

        dims = [802816, 401408, 200704, 100352, input_dim]
        for i in range(self.nos):
            if structured_comm:
                fc_layer = nn.Sequential(
                    nn.Linear(dims[i], hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim, bias=False),
                )
            else:
                fc_layer = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim, bias=False),
                )
            self.fc_out.append(fc_layer)
        self.fc_out = nn.ModuleList(self.fc_out)

    def forward(self, _x, resnet_output):
        out = []
        for i in range(self.nos):
            if self.structured_comm:
                out.append(self.fc_out[i](resnet_output[i+1]))
            else:
                out.append(self.fc_out[i](resnet_output))
        out = torch.concat(out, dim=1)
        if self.structured_comm:
            resnet_output = resnet_output[5]
        return out, resnet_output.detach()


class EmComSSLSymbolGame(SenderReceiverContinuousCommunication):
    def __init__(self, *args, **kwargs):
        super(EmComSSLSymbolGame, self).__init__(*args, **kwargs)

    def forward(self, sender_input, labels, receiver_input):
        if isinstance(self.sender, SimCLRSender):
            message, message_like, resnet_output_sender = self.sender(
                sender_input, sender=True
            )
            receiver_output, _, resnet_output_recv = self.receiver(receiver_input)
        else:
            message, message_like, resnet_output_sender = self.sender(sender_input)
            receiver_output, resnet_output_recv = self.receiver(message, receiver_input)

        loss, aux_info = self.loss(
            sender_input, message, receiver_input, receiver_output, labels, None
        )

        if hasattr(self.sender, "temperature"):
            if isinstance(self.sender.temperature, torch.nn.Parameter):
                temperature = self.sender.temperature.detach()
            else:
                temperature = torch.Tensor([self.sender.temperature])
            aux_info["temperature"] = temperature

        if not self.training:
            aux_info["message_like"] = message_like
            aux_info["resnet_output_sender"] = resnet_output_sender
            aux_info["resnet_output_recv"] = resnet_output_recv

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            aux_input=None,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=torch.ones(message.size(0)),
            aux=aux_info,
        )

        return loss.mean(), interaction
