"""Render a concise Markdown report from measured API results."""


def generate_report(result: dict) -> str:
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("A measured experiment result with metrics is required")
    lines = [
        "# Q-Agri Lab experiment report",
        "",
        f"- Model: {result.get('model_type', 'unknown')}",
        f"- Split: {result.get('split', 'unknown')}",
        f"- Seed: {result.get('seed', 'unknown')}",
        f"- Training rows: {result.get('training_rows', 'not recorded')}",
        f"- Runtime (seconds): {result.get('runtime_seconds', 'not recorded')}",
        f"- Accuracy: {metrics.get('accuracy', 'not reported')}",
        f"- Macro-F1: {metrics.get('macro_f1', 'not reported')}",
        "",
        "## Per-class metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in metrics.get("per_class", {}).items():
        lines.append(
            f"| {label} | {values.get('precision')} | {values.get('recall')} | "
            f"{values.get('f1')} | {values.get('support')} |"
        )
    lines.extend(["", "## Confusion matrix", "", "```json"])
    import json
    lines.append(json.dumps(metrics.get("confusion_matrix"), indent=2))
    lines.extend(["```", "", "Test labels are not supplied by the challenge; this report uses validation results."])
    return "\n".join(lines)
