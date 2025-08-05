import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import statsmodels.api as sm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    HAS_RDKIT = True
except Exception:  # pragma: no cover - rdkit may not be installed
    HAS_RDKIT = False


def load_data(path: str = "qm9_chon_ai.csv") -> pd.DataFrame:
    """Load dataset. If empty, generate synthetic data for demonstration."""
    df = pd.read_csv(path)
    if df.empty:
        rng = np.random.default_rng(0)
        n = 100
        smiles = ["C" * (i % 3 + 1) for i in range(n)]
        df = pd.DataFrame(
            {
                "smiles": smiles,
                "ai_exact": rng.normal(size=n),
                "ai_surrogate": rng.normal(size=n),
                "scaffold_id": rng.integers(0, 10, n),
                "mol_wt": rng.normal(300, 10, n),
                "logp": rng.normal(2, 1, n),
            }
        )
        df["ai_conflict"] = (np.abs(df["ai_exact"] - df["ai_surrogate"]) > 0.5).astype(int)
    return df


def ks_ai(df: pd.DataFrame):
    """Kolmogorov–Smirnov test comparing exact vs surrogate AI."""
    return ks_2samp(df["ai_exact"], df["ai_surrogate"])


def bemis_murcko_diversity(df: pd.DataFrame) -> float:
    """Compute Bemis–Murcko scaffold diversity."""
    if not HAS_RDKIT:
        raise ImportError("RDKit is required for scaffold diversity analysis.")
    scaffolds = []
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
    unique = set(scaffolds)
    return len(unique) / len(scaffolds) if scaffolds else float("nan")


def mixed_effects_logistic(
    df: pd.DataFrame, response: str, predictors: list[str], group: str
):
    """Fit a mixed-effects logistic regression using statsmodels."""
    df = df.copy()
    df[group] = df[group].astype("category")
    formula = response + " ~ " + " + ".join(predictors)
    vc_formula = {"scaffold": "0 + C(" + group + ")"}
    model = BinomialBayesMixedGLM.from_formula(formula, vc_formula, df)
    return model.fit_vb()


def bootstrap_delta_median(
    df: pd.DataFrame, n_boot: int = 1000, random_state: int | None = 0
) -> np.ndarray:
    """Bootstrap the difference in medians between exact and surrogate AI."""
    rng = np.random.default_rng(random_state)
    x = df["ai_exact"].to_numpy()
    y = df["ai_surrogate"].to_numpy()
    n = len(df)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas.append(np.median(x[idx]) - np.median(y[idx]))
    return np.asarray(deltas)


def sensitivity_over_lambda(
    df: pd.DataFrame, lambdas=(0.25, 0.75, 1.0)
):
    """Run mixed-effects logistic regression for different λ thresholds."""
    results = {}
    for lam in lambdas:
        col = f"conflict_lambda_{str(lam).replace('.', '_')}"
        df[col] = (np.abs(df["ai_exact"] - df["ai_surrogate"]) > lam).astype(int)
        res = mixed_effects_logistic(df, col, ["mol_wt", "logp"], "scaffold_id")
        results[lam] = res.params
    return results


def main() -> None:
    try:
        df = load_data()
    except Exception as exc:  # pragma: no cover - I/O error
        print(f"Failed to load dataset: {exc}")
        return

    stat, pvalue = ks_ai(df)
    print(f"KS statistic: {stat:.4f}, p-value: {pvalue:.4g}")

    if HAS_RDKIT:
        try:
            diversity = bemis_murcko_diversity(df)
            print(f"Scaffold diversity: {diversity:.4f}")
        except Exception as exc:  # pragma: no cover - rdkit failure
            print(f"Scaffold diversity failed: {exc}")
    else:
        print("RDKit not available; skipping scaffold diversity.")

    deltas = bootstrap_delta_median(df)
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    print(
        f"Δmedian AI (mean): {deltas.mean():.4f}, 95% CI: ({ci_low:.4f}, {ci_high:.4f})"
    )

    print("Sensitivity analysis over λ thresholds:")
    try:
        sens = sensitivity_over_lambda(df)
        for lam, params in sens.items():
            print(f"λ={lam}:\n{params}\n")
    except Exception as exc:  # pragma: no cover - regression failure
        print(f"Mixed-effects logistic regression failed: {exc}")


if __name__ == "__main__":
    main()
