import os
import pandas as pd
from datasets import Dataset

def export_manifest(df, name="final_manifest", formats=["csv", "json", "hf"]):
    """
    Export df in selected formats.

    Args:
        df: dataframe which will be exported
        name: name of file without extension
        formats to export: csv, json, hf
    """
    os.makedirs(os.path.dirname(name), exist_ok=True)

    if "csv" in formats:
        csv_path = f"{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV to {csv_path}")

    if "json" in formats:
        json_path = f"{name}.json"
        df.to_json(json_path, orient="records", lines=True)
        print(f"Saved JSON to {json_path}")

    if "hf" in formats:
        try:
            hf_dataset = Dataset.from_pandas(df)
            hf_path = f"{name}_hf"
            hf_dataset.save_to_disk(hf_path)
            print(f"Saved Hugging Face Dataset to {hf_path}")
        except Exception as e:
            print(f"Hugging Face export failed: {e}")

if __name__ == "__main__":
    df = pd.read_csv("final_manifest.csv")
    formats = ["csv", "json", "hf"]
    export_manifest(df, "exports/final_manifest", formats)
