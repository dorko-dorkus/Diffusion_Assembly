import json, os, subprocess, sys, tempfile, yaml, shutil, logging
from pathlib import Path

logger = logging.getLogger(__name__)

def run(name):
    out = subprocess.check_output([sys.executable, "scripts/experiment.py", "--name", name]).decode()
    logger.info(out)
    # extract last line path
    for line in out.splitlines()[::-1]:
        if "Wrote manifest" in line:
            return line.split()[-1]
    raise RuntimeError("Run output dir not found")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    reg = yaml.safe_load(open("configs/registry.yaml"))
    A = "exp01_baseline"
    B = "exp02_guided_lambda1"

    dirA = run(A)
    dirB = run(B)

    mA = json.load(open(Path(dirA)/"metrics.json"))
    mB = json.load(open(Path(dirB)/"metrics.json"))

    summary = {
        "A": A, "B": B,
        "validity_delta": mB["valid_fraction"] - mA["valid_fraction"],
        "uniqueness_delta": mB["uniqueness"] - mA["uniqueness"],
        "diversity_delta": mB["diversity"] - mA["diversity"],
        "median_AI_delta": mB["median_ai"] - mA["median_ai"],
    }

    rows = []
    metrics = [
        ("valid_fraction", "Validity"),
        ("uniqueness", "Uniqueness"),
        ("diversity", "Diversity"),
        ("median_ai", "Median AI"),
    ]
    for key, label in metrics:
        delta = mB[key] - mA[key]
        ciA = mA.get(f"{key}_ci")
        ciB = mB.get(f"{key}_ci")
        if ciA and ciB:
            delta_ci = [ciB[0] - ciA[1], ciB[1] - ciA[0]]
        else:
            delta_ci = None
        rows.append((label, delta, delta_ci))

    logger.info("| Metric | Δ(B−A) | 95% CI |")
    logger.info("|---|---|---|")
    for label, delta, ci in rows:
        if ci:
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        else:
            ci_str = "N/A"
        logger.info("| %s | %.3f | %s |", label, delta, ci_str)

    logger.info(json.dumps(summary, indent=2))
    with open("results/ab_summary.json","w") as f:
        json.dump(summary,f,indent=2)
    logger.info("[OK] Wrote results/ab_summary.json")
