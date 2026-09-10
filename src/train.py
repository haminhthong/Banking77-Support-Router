"""CLI huấn luyện Banking77 Support Router."""

from __future__ import annotations

import argparse

from src.banking_router.config import SEED
from src.banking_router.modeling.training import train_and_optimize
from src.banking_router.utils import set_seed, setup_logging


def main() -> None:
    setup_logging()
    set_seed(SEED)
    parser = argparse.ArgumentParser(description="Train Banking77 Support Router")
    parser.add_argument(
        "--benchmark", choices=["official", "strict_decontaminated"], default="official"
    )
    parser.add_argument(
        "--artifacts-dir", default="artifacts", help="Nơi lưu bộ artifact canonical"
    )
    parser.add_argument(
        "--reports-dir", default="reports/training", help="Nơi lưu validation report"
    )
    args = parser.parse_args()
    train_and_optimize(
        seed=SEED,
        benchmark=args.benchmark,
        artifacts_dir=args.artifacts_dir,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()
