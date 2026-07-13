"""Publish a released bare tensor state dict in the project artifact format."""

from __future__ import annotations

import argparse
import json

from models.world_model.publication import publish_released_lewm_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config")
    parser.add_argument("source_weights")
    parser.add_argument("output_path")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    metadata = publish_released_lewm_artifact(
        source_weights=arguments.source_weights,
        model_config=arguments.model_config,
        output_path=arguments.output_path,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
