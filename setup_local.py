"""One-time setup script to download the encoder locally and pre-build doc_artifacts.pt."""

from pathlib import Path

from transformers import AutoModel, AutoTokenizer

from engine import PerpresSearchEngine


def main() -> None:
    root = Path(__file__).resolve().parent
    local_model_dir = root / "models" / "multilingual-e5-small"
    local_model_dir.mkdir(parents=True, exist_ok=True)

    model_id = "intfloat/multilingual-e5-small"
    print(f"Downloading {model_id} to {local_model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    tokenizer.save_pretrained(local_model_dir)
    model.save_pretrained(local_model_dir)
    print("Base encoder saved locally.")

    print("Building/verifying doc_artifacts.pt ...")
    PerpresSearchEngine(base_dir=root)
    print("Done! Uvicorn can now start offline without downloading.")


if __name__ == "__main__":
    main()
