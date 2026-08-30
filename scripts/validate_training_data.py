#!/usr/bin/env python3
"""Validate packed training payload sizes, checksums, mixture, and order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.data import DOMAIN_ORDER, validate_packed_manifest, validate_training_order


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("order_manifest", type=Path)
    parser.add_argument(
        "--skip-payload-checksums",
        action="store_true",
        help="validate sizes and metadata but skip the expensive binary SHA-256 pass",
    )
    args = parser.parse_args()
    try:
        order = validate_training_order(args.order_manifest)
        datasets = {}
        for domain in DOMAIN_ORDER:
            payload = order["dataset_manifests"][domain]
            path = args.order_manifest.parent / payload["path"]
            datasets[domain] = validate_packed_manifest(
                path,
                verify_checksums=not args.skip_payload_checksums,
            )
        summary = {
            "order_rows": order["rows"],
            "order_sha256": order["order"]["sha256"],
            "seed": order["seed"],
            "shuffle": order["shuffle"],
            "training_consumption": order["training_consumption"],
            "input_token_budget": order["input_token_budget"],
            "domains": {
                domain: {
                    "rows": datasets[domain]["rows"],
                    "input_tokens": datasets[domain]["input_tokens"],
                    "valid_loss_tokens": datasets[domain]["valid_loss_tokens"],
                    "masked_boundary_labels": datasets[domain]["masked_boundary_labels"],
                }
                for domain in DOMAIN_ORDER
            },
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, IOError, KeyError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
