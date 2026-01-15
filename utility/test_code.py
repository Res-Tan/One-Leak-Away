import json
import re
import ast
from typing import Optional


def extract_python(pred: str) -> Optional[str]:
    if not isinstance(pred, str):
        return None

    match = re.search(r"```python(.*?)```", pred, re.S | re.I)
    if match:
        return match.group(1).strip()
    return None


def ast_syntax_valid(code: str) -> bool:
    """
    Check whether the given string is syntactically valid Python.
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def ast_valid_from_prediction(pred: str) -> bool:
    code = extract_python(pred)
    if code is None:
        return False
    return ast_syntax_valid(code)

def evaluate_ast_validity(data_list):
    total = len(data_list)

    python_extractable = 0
    ast_valid = 0

    for data in data_list:
        pred = data.get("prediction", "")

        code = extract_python(pred)
        if code is not None:
            python_extractable += 1
            if ast_syntax_valid(code):
                ast_valid += 1

    results = {
        "total_samples": total,
        "python_extractable": python_extractable,
        "python_extractable_rate": python_extractable / total if total > 0 else 0.0,
        "ast_valid": ast_valid,
        "ast_valid_rate_overall": ast_valid / total if total > 0 else 0.0,
        "ast_valid_rate_conditional": (
            ast_valid / python_extractable if python_extractable > 0 else 0.0
        ),
    }
    return results

if __name__ == "__main__":
    path = "k"
    target_path = "results_new/utility/codeEvol/llama2_7b/bs32_pm50.json"

    with open(f"{path}/{target_path}", "r") as f:
        data_list = json.load(f)

    stats = evaluate_ast_validity(data_list)

    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
