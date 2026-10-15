"""CLI for publishing path-free multimodal development evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_science.qwen3vl.public_development_evidence import (
    build_public_development_evidence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-dossier-root", type=Path, required=True)
    parser.add_argument("--lora-training-root", type=Path, required=True)
    parser.add_argument("--fusion-training-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--auxiliary-pretraining-root", type=Path)
    parser.add_argument("--random-projector-evaluation-root", type=Path)
    parser.add_argument("--auxiliary-projector-evaluation-root", type=Path)
    parser.add_argument("--xic-only-evaluation-root", type=Path)
    parser.add_argument("--sequence-prompt-evaluation-root", type=Path)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    result = build_public_development_evidence(
        development_dossier_root=arguments.development_dossier_root,
        lora_training_root=arguments.lora_training_root,
        fusion_training_root=arguments.fusion_training_root,
        output_dir=arguments.output_dir,
        auxiliary_pretraining_root=arguments.auxiliary_pretraining_root,
        random_projector_evaluation_root=arguments.random_projector_evaluation_root,
        auxiliary_projector_evaluation_root=(
            arguments.auxiliary_projector_evaluation_root
        ),
        xic_only_evaluation_root=arguments.xic_only_evaluation_root,
        sequence_prompt_evaluation_root=arguments.sequence_prompt_evaluation_root,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "markdown_path": str(result.markdown_path),
                "manifest_path": str(result.manifest_path),
                "internal_test_accessed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
