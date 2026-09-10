from pathlib import Path
import pandas as pd
from guardrails.validators import FailResult

from app.core.validators.pii_remover import PIIRemover
from app.evaluation.common.helper import (
    Profiler,
    build_evaluation_report,
    compute_binary_metrics,
    write_csv,
    write_json,
)
from app.evaluation.pii.entity_metrics import compute_entity_metrics

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs" / "pii_remover"

df = pd.read_csv(BASE_DIR / "datasets" / "pii_detection_testing_dataset.csv")

validator = PIIRemover()

config = {
    "entity_types": validator.entity_types,
    "threshold": validator.threshold,
    "nlp_engine_type": validator.nlp_engine_type,
    "model_name": validator.model_name,
    "on_fail": validator.on_fail,
    "language": "en",  # hardcoded in PIIRemover._validate; not a constructor param
}


def run_pii(text: str) -> tuple[str, int]:
    result = validator._validate(text)
    if isinstance(result, FailResult):
        return result.fix_value, 1
    return text, 0


with Profiler() as p:
    results = df["source_text"].astype(str).apply(lambda x: p.record(run_pii, x))
    df["anonymized"] = results.apply(lambda r: r[0])
    df["pii_detected"] = results.apply(lambda r: r[1])

entity_report = compute_entity_metrics(
    df["target_text"],
    df["anonymized"],
)

y_true = (df["label"] == "pii").astype(int)
binary_report = compute_binary_metrics(y_true, df["pii_detected"])

# ---- Save outputs ----
write_csv(df, OUT_DIR / "predictions.csv")

write_json(
    build_evaluation_report(
        guardrail="pii_remover",
        num_samples=len(df),
        profiler=p,
        config=config,
        entity_metrics=entity_report,
        binary_metrics=binary_report,
    ),
    OUT_DIR / "metrics.json",
)
