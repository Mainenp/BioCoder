"""Trainable 1D XIC encoder and projector for Qwen hidden-space sensor tokens."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SensorProjectorSpec:
    input_points: int = 160
    hidden_size: int = 2560
    sensor_tokens: int = 4
    base_channels: int = 32
    dropout: float = 0.1

    def validate(self) -> None:
        if self.input_points < 32 or self.input_points % 8:
            raise ValueError("input_points must be at least 32 and divisible by eight")
        if self.hidden_size < 64:
            raise ValueError("hidden_size must be at least 64")
        if self.sensor_tokens < 1:
            raise ValueError("sensor_tokens must be positive")
        encoded_points = self.input_points // 8
        if encoded_points % self.sensor_tokens:
            raise ValueError("Encoded points must divide evenly into sensor tokens")
        if self.base_channels < 8:
            raise ValueError("base_channels must be at least eight")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def build_sensor_projector(spec: SensorProjectorSpec) -> Any:
    """Encode normalized XIC traces as a short sequence of Qwen-width tokens."""

    spec.validate()
    import torch
    from torch import nn

    def norm(channels: int) -> Any:
        groups = next(value for value in (8, 4, 2, 1) if channels % value == 0)
        return nn.GroupNorm(groups, channels)

    class ResidualDownsample(nn.Module):
        def __init__(self, input_channels: int, output_channels: int) -> None:
            super().__init__()
            self.main = nn.Sequential(
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    bias=False,
                ),
                norm(output_channels),
                nn.GELU(),
                nn.Dropout(spec.dropout),
                nn.Conv1d(
                    output_channels,
                    output_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                norm(output_channels),
            )
            self.skip = nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=2, bias=False),
                norm(output_channels),
            )
            self.activation = nn.GELU()

        def forward(self, inputs: Any) -> Any:
            return self.activation(self.main(inputs) + self.skip(inputs))

    class SensorProjector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = spec.base_channels
            self.encoder = nn.Sequential(
                nn.Conv1d(1, channels, kernel_size=7, padding=3, bias=False),
                norm(channels),
                nn.GELU(),
                ResidualDownsample(channels, channels * 2),
                ResidualDownsample(channels * 2, channels * 3),
                ResidualDownsample(channels * 3, channels * 4),
            )
            self.projector = nn.Sequential(
                nn.LayerNorm(channels * 4),
                nn.Linear(channels * 4, spec.hidden_size),
                nn.GELU(),
                nn.Linear(spec.hidden_size, spec.hidden_size),
                nn.LayerNorm(spec.hidden_size),
            )
            self.gate_logit = nn.Parameter(torch.tensor(-4.0))

        def forward(self, signals: Any) -> Any:
            if signals.ndim != 2 or signals.shape[1] != spec.input_points:
                raise ValueError(
                    f"Expected [batch, {spec.input_points}] signals, got {tuple(signals.shape)}"
                )
            encoded = self.encoder(signals.unsqueeze(1))
            points = int(encoded.shape[-1])
            width = points // spec.sensor_tokens
            pooled = encoded.reshape(
                encoded.shape[0], encoded.shape[1], spec.sensor_tokens, width
            ).mean(dim=-1)
            tokens = self.projector(pooled.transpose(1, 2))
            return tokens * self.gate_logit.sigmoid()

    return SensorProjector()
