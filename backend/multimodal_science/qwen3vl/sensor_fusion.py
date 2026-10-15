"""Insert projected XIC measurements into Qwen's continuous embedding stream."""

from __future__ import annotations

from typing import Any, Sequence


def insert_sensor_embeddings(
    model: Any,
    input_ids: Any,
    labels: Any,
    sensor_embeddings: Any,
    prompt_lengths: Sequence[int],
    *,
    ignore_index: int = -100,
    position_token_id: int = 0,
) -> tuple[Any, Any, Any, Any]:
    """Insert XIC embeddings immediately before each assistant response.

    The inserted positions are never language-model targets. Keeping the original
    tokenizer and vocabulary unchanged prevents sensor slots from becoming
    meaningless output classes during generation.
    """

    import torch

    if input_ids.ndim != 2 or labels.ndim != 2 or sensor_embeddings.ndim != 3:
        raise ValueError("Bad input, label, or sensor tensor rank")
    if input_ids.shape != labels.shape:
        raise ValueError("Input and label shapes disagree")
    batch_size, sequence_length = input_ids.shape
    if sensor_embeddings.shape[0] != batch_size:
        raise ValueError("Sensor batch size disagrees with language batch size")
    if len(prompt_lengths) != batch_size:
        raise ValueError("One prompt length is required per example")
    if not isinstance(position_token_id, int) or position_token_id < 0:
        raise ValueError("position_token_id must be a nonnegative integer")

    token_embeddings = model.get_input_embeddings()(input_ids)
    if sensor_embeddings.shape[2] != token_embeddings.shape[2]:
        raise ValueError("Sensor and language hidden widths disagree")
    sensor_embeddings = sensor_embeddings.to(
        device=token_embeddings.device,
        dtype=token_embeddings.dtype,
    )

    fused_embeddings = []
    fused_labels = []
    shadow_input_ids = []
    for row_index, boundary_value in enumerate(prompt_lengths):
        boundary = int(boundary_value)
        if not 0 <= boundary <= sequence_length:
            raise ValueError(f"Prompt boundary {boundary} is outside the sequence")
        fused_embeddings.append(
            torch.cat(
                (
                    token_embeddings[row_index, :boundary],
                    sensor_embeddings[row_index],
                    token_embeddings[row_index, boundary:],
                ),
                dim=0,
            )
        )
        ignored = torch.full(
            (sensor_embeddings.shape[1],),
            ignore_index,
            dtype=labels.dtype,
            device=labels.device,
        )
        fused_labels.append(
            torch.cat(
                (labels[row_index, :boundary], ignored, labels[row_index, boundary:]),
                dim=0,
            )
        )
        shadow_tokens = torch.full(
            (sensor_embeddings.shape[1],),
            position_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        shadow_input_ids.append(
            torch.cat(
                (
                    input_ids[row_index, :boundary],
                    shadow_tokens,
                    input_ids[row_index, boundary:],
                ),
                dim=0,
            )
        )

    embeddings = torch.stack(fused_embeddings)
    targets = torch.stack(fused_labels)
    shadow_ids = torch.stack(shadow_input_ids)
    attention_mask = torch.ones(
        targets.shape,
        dtype=torch.long,
        device=targets.device,
    )
    return embeddings, targets, attention_mask, shadow_ids
