from pathlib import Path
import pandas as pd


def update_results_csv(
    results: pd.DataFrame,
    key_columns: list,
    csv_path="./results/results.csv"
):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Work on a copy to avoid modifying the original DataFrame.
    results = results.copy()

    # Normalize key dtypes.
    results["Model"] = results["Model"].astype(str)
    results["N. Classes"] = pd.to_numeric(results["N. Classes"])
    results["N. Examples"] = pd.to_numeric(results["N. Examples"])
    results["Seed"] = pd.to_numeric(results["Seed"])

    # Create the CSV if it does not exist yet.
    if not csv_path.exists():
        results.to_csv(csv_path, index=False)
        return

    old_results = pd.read_csv(csv_path)

    # Normalize the same columns in the existing results.
    old_results["Model"] = old_results["Model"].astype(str)
    old_results["N. Classes"] = pd.to_numeric(old_results["N. Classes"])
    old_results["N. Examples"] = pd.to_numeric(old_results["N. Examples"])
    old_results["Seed"] = pd.to_numeric(old_results["Seed"])

    # Configurations contained in the new batch.
    new_keys = results[key_columns].drop_duplicates()

    # Remove old rows corresponding to configurations being updated.
    merged = old_results.merge(
        new_keys,
        on=key_columns,
        how="left",
        indicator=True,
    )

    old_results = old_results[
        merged["_merge"].eq("left_only").values
    ]

    # Append updated/new results.
    updated_results = pd.concat(
        [old_results, results],
        ignore_index=True,
    )

    updated_results.to_csv(csv_path, index=False)