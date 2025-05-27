"""
Models for the project

Drafted by: Juntang Wang @ May 27, 2025

Copyright (c) 2025, Reserved
"""
import math
from typing import List, Optional, Callable
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import MLP
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar


class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(disable=True)
        return bar


class LitBase(pl.LightningModule):
    """
    Base class for all LightningModules.
    """
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y)
        self.train_losses.append(loss.cpu().item())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_func(y_pred, y)
        score = self.score(y, y_pred)
        self.val_losses.append(loss.cpu().item())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_" + self.metric_name, score, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_func(y_pred, y)
        score = self.score(y, y_pred)
        self.log("test_loss", loss)
        self.log("test_" + self.metric_name, score)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.forward(x)

    def score(self, y: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Score the model."""
        if self.is_classifier: y_pred = torch.argmax(y_pred, dim=-1)
        score = self.score_func(y.cpu().numpy(), y_pred.cpu().numpy())
        return score


class CustomMLP(nn.Module):
    """This block implements a custom MLP that handles both classification and regression tasks.
    Based on torchvision.ops.misc.MLP implementation.

    Args:
        input_dim (int): Number of channels of the input.
        hidden_dims (List[int]): List of the hidden channel dimensions.
        output_dim (int): Number of output dimensions.
        norm_layer (Optional[Callable[..., nn.Module]], optional): Norm layer that will be stacked on top of the linear layer. 
            If ``None`` this layer won't be used. Default: ``None``.
        activation_layer (Optional[Callable[..., nn.Module]], optional): Activation function which will be stacked on top of the 
            normalization layer (if not None), otherwise on top of the linear layer. Default: ``nn.ReLU``.
        inplace (Optional[bool], optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``.
        bias (bool): Whether to use bias in the linear layer. Default is ``True``.
        dropout (float): The probability for the dropout layer. Default: 0.2.
        is_classifier (bool): Whether this is a classification model. Default: False.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        output_dim: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.2,
        is_classifier: bool = False,
    ):
        super().__init__()
        self.is_classifier = is_classifier

        # Create the MLP layers
        layer_dims = hidden_dims + [output_dim]  # including output layer
        self.mlp = MLP(
            in_channels=input_dim,
            hidden_channels=layer_dims,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            inplace=inplace,
            bias=bias,
            dropout=dropout
        )

        # If not a classifier, remove the last layer
        if not self.is_classifier:
            layers = list(self.mlp.children())
            del layers[-1]  # Remove the last layer
            self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class LitCNN(LitBase):
    def __init__(
        self,
        model: nn.Module,
        loss_func: nn.Module,
        score_func: Callable,
        metric_name: str,
        is_classifier: bool = False,
    ):
        super().__init__()
        self.model = model
        self.loss_func = loss_func
        self.train_losses = []
        self.val_losses = []
        
        self.loss_func = loss_func
        self.score_func = score_func
        self.metric_name = metric_name
        self.is_classifier = is_classifier
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.model(x)
        if not self.is_classifier:
            y_hat = y_hat.squeeze(-1)
        return y_hat


class CustomCNN(nn.Module):
    """
    A custom CNN model with configurable architecture.
    
    Args:
        input_channels (int): Number of input channels.
        num_classes (int): Number of output classes for classification.
        conv_channels (List[int]): List of channel dimensions for conv layers. Default: [8, 16, 32].
        kernel_sizes (List[int]): List of kernel sizes for conv layers. Default: [3, 3, 3].
        pool_sizes (List[int]): List of pool sizes. Default: [2, 2].
        dropout (float): Dropout probability. Default: 0.2.
        is_classifier (bool): Whether this is a classification model. Default: True.
    """
    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = None,
        conv_channels: List[int] = [8, 16, 32],
        kernel_sizes: List[int] = [3, 3, 3],
        pool_sizes: List[int] = [2, 2],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.is_classifier = num_classes is not None
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, conv_channels[0], kernel_size=kernel_sizes[0], padding="same"),
            nn.BatchNorm1d(conv_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_sizes[0])
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=kernel_sizes[1], padding="same"),
            nn.BatchNorm1d(conv_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_sizes[1])
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=kernel_sizes[2], padding="same"),
            nn.BatchNorm1d(conv_channels[2]),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        if self.is_classifier:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(conv_channels[-1], num_classes),
                # nn.Softmax(dim=1)
            )
        else:
            self.regressor = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(4096, 1)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        if self.is_classifier:
            return self.classifier(x)
        else:
            return self.regressor(x)
