"""Residual 1D peak detector with an optional audited-metadata branch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SequenceModelSpec:
    input_points: int = 160
    scalar_features: int = 7
    modality: str = "sequence"
    base_channels: int = 32
    position_bins: int = 10
    dropout: float = 0.15

    def validate(self) -> None:
        if self.input_points < 32:
            raise ValueError("Sequence model requires at least 32 input points")
        if self.scalar_features < 1:
            raise ValueError("At least one scalar feature must be declared")
        if self.modality not in {"sequence", "sequence_metadata"}:
            raise ValueError(f"Unsupported sequence modality: {self.modality}")
        if self.base_channels < 8:
            raise ValueError("base_channels must be at least eight")
        if self.position_bins < 2:
            raise ValueError("position_bins must be at least two")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def build_sequence_peak_net(spec: SequenceModelSpec) -> Any:
    """Build the PyTorch module while keeping torch optional for data-only users."""

    spec.validate()
    import torch
    from torch import nn

    def normalization(channels: int) -> Any:
        groups = next(value for value in (8, 4, 2, 1) if channels % value == 0)
        return nn.GroupNorm(num_groups=groups, num_channels=channels)

    class ResidualBlock(nn.Module):
        def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
            super().__init__()
            self.main = nn.Sequential(
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size=5,
                    stride=stride,
                    padding=2,
                    bias=False,
                ),
                normalization(output_channels),
                nn.GELU(),
                nn.Dropout(spec.dropout),
                nn.Conv1d(
                    output_channels,
                    output_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                normalization(output_channels),
            )
            if stride == 1 and input_channels == output_channels:
                self.skip = nn.Identity()
            else:
                self.skip = nn.Sequential(
                    nn.Conv1d(
                        input_channels,
                        output_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    normalization(output_channels),
                )
            self.activation = nn.GELU()

        def forward(self, inputs: Any) -> Any:
            return self.activation(self.main(inputs) + self.skip(inputs))

    class SequencePeakNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = spec.base_channels
            self.encoder = nn.Sequential(
                nn.Conv1d(1, channels, kernel_size=7, padding=3, bias=False),
                normalization(channels),
                nn.GELU(),
                ResidualBlock(channels, channels, stride=1),
                ResidualBlock(channels, channels * 2, stride=2),
                ResidualBlock(channels * 2, channels * 3, stride=2),
                ResidualBlock(channels * 3, channels * 4, stride=2),
            )
            self.average_pool = nn.AdaptiveAvgPool1d(spec.position_bins)
            self.maximum_pool = nn.AdaptiveMaxPool1d(spec.position_bins)
            encoded_features = channels * 8 * spec.position_bins
            self.metadata_encoder = None
            if spec.modality == "sequence_metadata":
                self.metadata_encoder = nn.Sequential(
                    nn.LayerNorm(spec.scalar_features),
                    nn.Linear(spec.scalar_features, 32),
                    nn.GELU(),
                    nn.Dropout(spec.dropout),
                    nn.Linear(32, 32),
                    nn.GELU(),
                )
                encoded_features += 32
            self.fusion = nn.Sequential(
                nn.Linear(encoded_features, 128),
                nn.GELU(),
                nn.Dropout(spec.dropout),
                nn.Linear(128, 64),
                nn.GELU(),
            )
            self.presence_head = nn.Linear(64, 1)
            self.boundary_head = nn.Linear(64, 2)
            with torch.no_grad():
                self.boundary_head.bias.copy_(torch.tensor([0.0, -1.5]))

        def forward(self, signals: Any, scalar_features: Any) -> tuple[Any, Any]:
            encoded = self.encoder(signals.unsqueeze(1))
            pooled = torch.cat(
                [self.average_pool(encoded), self.maximum_pool(encoded)],
                dim=1,
            ).flatten(start_dim=1)
            if self.metadata_encoder is not None:
                pooled = torch.cat(
                    [pooled, self.metadata_encoder(scalar_features)],
                    dim=1,
                )
            fused = self.fusion(pooled)
            presence_logits = self.presence_head(fused).squeeze(1)
            raw_boundaries = self.boundary_head(fused)
            start = torch.sigmoid(raw_boundaries[:, 0])
            width_fraction = torch.sigmoid(raw_boundaries[:, 1])
            end = start + (1.0 - start) * width_fraction
            return presence_logits, torch.stack([start, end], dim=1)

    return SequencePeakNet()
