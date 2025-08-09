import argparse
import numpy as np
import pandas as pd
import logging

from analysis import ks_test, sensitivity_over_lambda

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce analysis checks")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a lightweight smoke test",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    size = 50 if args.smoke else 1000
    sample_a = rng.normal(0.0, 1.0, size)
    sample_b = rng.normal(1.0, 1.0, size)

    res = ks_test(sample_a, sample_b)
    logger.info("KS statistic: %.3f, p-value: %.3g", res['statistic'], res['pvalue'])
    if res["pvalue"] >= 0.05:
        raise SystemExit("Failed to reject null hypothesis in KS test")

    df = pd.DataFrame({"ai_exact": sample_a[:5], "ai_surrogate": sample_b[:5]})
    medians = sensitivity_over_lambda(df)
    logger.info("Sensitivity medians: %s", medians)


if __name__ == "__main__":
    main()
