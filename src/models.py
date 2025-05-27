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
        # from the bangzi group
        optimizer = optim.Adam(
            self.parameters(), 
            lr=0.0005, 
            betas=(0.9, 0.999), 
            eps=1e-8, 
            weight_decay=0.0, 
            amsgrad=False
        )
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

# ---------------------------------------------------------------------------- #
#                      https://github.com/MbetterLife-dsp/                     #
# ---------------------------------------------------------------------------- #
class CustomCNN(nn.Module):
    """
    A custom CNN model with configurable architecture inspired by DCRNN.
    Uses depthwise separable convolutions followed by LSTM for sequence modeling.
    
    Args:
        input_channels (int): Number of input channels.
        num_classes (int): Number of output classes for classification.
        depthwise_kernels (List[int]): List of kernel sizes for depthwise conv layers. Default: [128, 64, 64].
        pointwise_channels (List[int]): List of channel dimensions for pointwise conv layers. Default: [16, 16, 16].
        pool_sizes (List[int]): List of pool sizes. Default: [2, 5].
        lstm_hidden (int): LSTM hidden dimension. Default: 10.
        dropout (float): Dropout probability. Default: 0.2.
    """
    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = None,
        depthwise_kernels: List[int] = [128, 64, 64],
        pointwise_channels: List[int] = [16, 16, 16],
        pool_sizes: List[int] = [2, 5],
        lstm_hidden: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.is_classifier = num_classes is not None
        
        # Separable Conv 1: Depthwise + Pointwise
        self.separable_conv1 = nn.Sequential(
            nn.Conv1d(input_channels, input_channels, kernel_size=depthwise_kernels[0], 
                     padding='same', groups=input_channels),  # Depthwise
            nn.Conv1d(input_channels, pointwise_channels[0], kernel_size=1, 
                     padding='same'),  # Pointwise
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(pointwise_channels[0]),
            nn.MaxPool1d(pool_sizes[0])
        )
        
        # Separable Conv 2: Depthwise + Pointwise
        self.separable_conv2 = nn.Sequential(
            nn.Conv1d(pointwise_channels[0], pointwise_channels[0], kernel_size=depthwise_kernels[1], 
                     padding='same', groups=pointwise_channels[0]),  # Depthwise
            nn.Conv1d(pointwise_channels[0], pointwise_channels[1], kernel_size=1, 
                     padding='same'),  # Pointwise
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(pointwise_channels[1]),
            nn.MaxPool1d(pool_sizes[1])
        )
        
        # Separable Conv 3: Depthwise + Pointwise (no pooling)
        self.separable_conv3 = nn.Sequential(
            nn.Conv1d(pointwise_channels[1], pointwise_channels[1], kernel_size=depthwise_kernels[2], 
                     padding='same', groups=pointwise_channels[1]),  # Depthwise
            nn.Conv1d(pointwise_channels[1], pointwise_channels[2], kernel_size=1, 
                     padding='same'),  # Pointwise
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(pointwise_channels[2])
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(pointwise_channels[2], lstm_hidden, batch_first=True)
        
        # Output head
        if self.is_classifier:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(lstm_hidden, num_classes)
            )
        else:
            self.regressor = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(lstm_hidden, 1)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply separable convolutions
        x = self.separable_conv1(x)
        x = self.separable_conv2(x)
        x = self.separable_conv3(x)
        
        # Transpose for LSTM (batch_first=True expects [batch, seq_len, features])
        x = x.transpose(1, 2)  # [batch, channels, length] -> [batch, length, channels]
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use the last output of the sequence
        x = lstm_out[:, -1, :]  # [batch, lstm_hidden]
        
        if self.is_classifier:
            return self.classifier(x)
        else:
            return self.regressor(x)
