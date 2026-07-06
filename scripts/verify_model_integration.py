"""
Verify that the trained BiLSTM and BERT models are ready for
application integration.

Checks:
- Required model files are present.
- Both models load successfully.
- Both models can perform a sample inference.
- Label ordering is consistent across both models.

Run:
    python scripts/verify_model_integration.py

Output:
    models/integration_readiness_report.txt

NOTE:
The verification script loads each model in a separate subprocess.
This avoids potential framework conflicts and mirrors the intended
deployment architecture. On some environments TensorFlow and PyTorch
can coexist in the same process, while others may experience issues
when both frameworks initialize simultaneously.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.label_mapping import TARGET_CLASSES

BILSTM_DIR = Path("models/bilstm")
BERT_DIR = Path("models/bert_emotion_model_final")
REPORT_PATH = Path("models/integration_readiness_report.txt")

BILSTM_REQUIRED_FILES = ["model.keras", "tokenizer.pickle", "label_encoder.json"]
SAMPLE_TEXT = "I don't understand this concept at all, can you explain it again?"

WORKER_DIR = Path(__file__).parent


def log(lines: list, text: str = "") -> None:
    print(text)
    lines.append(text)


def check_bilstm_files(lines: list) -> bool:
    log(lines, "\n== BiLSTM file inventory ==")
    all_present = True
    for filename in BILSTM_REQUIRED_FILES:
        path = BILSTM_DIR / filename
        status = "OK" if path.exists() else "MISSING"
        log(lines, f"  {filename:<25}: {status}")
        all_present = all_present and path.exists()
    return all_present


def check_bert_files(lines: list) -> bool:
    log(lines, "\n== BERT file inventory ==")
    if not BERT_DIR.exists():
        log(lines, f"  [MISSING] {BERT_DIR} does not exist at all")
        return False

    actual_files = sorted(p.name for p in BERT_DIR.iterdir() if p.is_file())
    log(lines, f"  Found {len(actual_files)} files: {actual_files}")

    has_config = "config.json" in actual_files
    has_weights = any(f in actual_files for f in ("model.safetensors", "pytorch_model.bin"))
    has_tokenizer = any(f in actual_files for f in ("tokenizer.json", "vocab.txt", "tokenizer_config.json"))

    log(lines, f"  Has config.json                : {has_config}")
    log(lines, f"  Has model weights file         : {has_weights}")
    log(lines, f"  Has at least one tokenizer file: {has_tokenizer}")

    if len(actual_files) < 7:
        log(
            lines,
            f"  [NOTE] Doc reference mentions 7 BERT components; found {len(actual_files)}. "
            "Not necessarily a problem -- exact file count depends on your transformers "
            "version and tokenizer type -- but worth a manual glance if it looks low.",
        )

    return has_config and has_weights and has_tokenizer


def run_worker(script_name: str) -> dict:
    """
    Runs a check worker in its own subprocess and parses its JSON
    result. Isolation is required here -- see module docstring.
    """
    result_proc = subprocess.run(
        [sys.executable, str(WORKER_DIR / script_name), SAMPLE_TEXT],
        capture_output=True, text=True,
    )
    for line in result_proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])

    # Worker crashed before printing its result line (e.g. import error) --
    # surface the raw stderr rather than failing silently.
    return {"status": "fail", "error": f"worker produced no result; stderr tail: {result_proc.stderr[-500:]}"}


def report_model_test(lines: list, name: str, result: dict):
    log(lines, f"\n== {name} load + inference test (isolated subprocess) ==")
    if result["status"] == "ok":
        log(lines, f"  Sample input   : {SAMPLE_TEXT!r}")
        log(lines, f"  Predicted class: {result['predicted_class']} (confidence {result['confidence']:.3f})")
        log(lines, "  Status: OK")
        return result["label_order"]
    else:
        log(lines, f"  [FAIL] {result['error']}")
        return None


def check_label_consistency(lines: list, bilstm_order, bert_order) -> bool:
    log(lines, "\n== Cross-model label order consistency ==")
    log(lines, f"  TARGET_CLASSES (source of truth): {TARGET_CLASSES}")
    log(lines, f"  BiLSTM label order              : {bilstm_order}")
    log(lines, f"  BERT label order                : {bert_order}")

    if bilstm_order is None or bert_order is None:
        log(lines, "  [SKIPPED] Cannot compare -- one or both models failed to load above.")
        return False

    consistent = (bilstm_order == TARGET_CLASSES) and (bert_order == TARGET_CLASSES)
    if consistent:
        log(lines, "  Status: OK -- all three orderings match exactly.")
    else:
        log(
            lines,
            "  [CRITICAL] Label orderings do NOT match. If deployed as-is, a "
            "Streamlit app comparing BiLSTM and BERT predictions would "
            "silently mislabel one of them -- no error, just wrong answers. "
            "Do not deploy until this is fixed.",
        )
    return consistent


def main() -> None:
    lines = []
    log(lines, "=" * 70)
    log(lines, "MODEL INTEGRATION READINESS REPORT")
    log(lines, "=" * 70)
    log(
        lines,
        "\nNOTE: TensorFlow and PyTorch/transformers cannot coexist in one "
        "process (confirmed -- causes a hard segfault). Each model below "
        "is loaded in its own isolated subprocess. Your Streamlit app will "
        "need the same isolation strategy.",
    )

    bilstm_files_ok = check_bilstm_files(lines)
    bert_files_ok = check_bert_files(lines)

    bilstm_result = run_worker("_integration_check_bilstm.py") if bilstm_files_ok else {"status": "fail", "error": "required files missing"}
    bilstm_order = report_model_test(lines, "BiLSTM", bilstm_result)

    bert_result = run_worker("_integration_check_bert.py") if bert_files_ok else {"status": "fail", "error": "required files missing"}
    bert_order = report_model_test(lines, "BERT", bert_result)

    labels_consistent = check_label_consistency(lines, bilstm_order, bert_order)

    ready = bilstm_files_ok and bert_files_ok and bilstm_order is not None and bert_order is not None and labels_consistent

    log(lines, "\n" + "=" * 70)
    log(lines, f"VERDICT: {'READY for Streamlit integration' if ready else 'NOT READY'}")
    log(lines, "=" * 70)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved full report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
