import json
import re
import ast
from typing import Optional, Tuple, Dict


FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.S)

def extract_first_fenced_block(text: str) -> Optional[Tuple[str, str]]:
    if not isinstance(text, str):
        return None
    m = FENCE_RE.search(text)
    if not m:
        return None
    lang = (m.group(1) or "").strip().lower()
    code = (m.group(2) or "").strip()
    return lang, code


def strip_leading_explanations(text: str) -> str:
    if not isinstance(text, str):
        return ""
    lines = text.splitlines()
    code_start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        # simple signals for code start
        if (
            s.startswith("def ")
            or s.startswith("class ")
            or s.startswith("import ")
            or s.startswith("from ")
            or s.startswith("@")
            or s.startswith("if ")
            or s.startswith("for ")
            or s.startswith("while ")
            or s.startswith("try:")
            or s.startswith("except")
            or s.endswith(":")
        ):
            code_start = i
            break
    return "\n".join(lines[code_start:]).strip()

def is_code_like(pred: str) -> bool:
    if not isinstance(pred, str):
        return False
    s = pred.strip()
    if "```" in s:
        return True
    keywords = ["def ", "class ", "import ", "{", "};", ";", "SELECT ", "<html", "#include", "public class"]
    return any(k.lower() in s.lower() for k in keywords)


def looks_like_python(code: str) -> bool:
    if not isinstance(code, str):
        return False
    s = code.strip()
    if s.startswith("<") or "SELECT " in s.upper() or "#include" in s or "public class" in s:
        return False
    py_signals = ["def ", "class ", "import ", "from ", "print(", "elif ", "None", "True", "False", "__name__"]
    return any(tok in s for tok in py_signals)


def python_ast_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def evaluate_codealpaca_syntax(data_list) -> Dict[str, float]:
    total = len(data_list)

    code_like = 0
    fenced = 0

    python_candidate = 0
    python_ast_ok = 0

    for item in data_list:
        pred = item.get("prediction", "")

        if not is_code_like(pred):
            continue
        code_like += 1

        block = extract_first_fenced_block(pred)
        if block is not None:
            fenced += 1
            lang, code = block
            if lang in ("python", "py"):
                python_candidate += 1
                if python_ast_valid(code):
                    python_ast_ok += 1
            else:
                if lang == "" and looks_like_python(code):
                    python_candidate += 1
                    if python_ast_valid(code):
                        python_ast_ok += 1
        else:
            code = strip_leading_explanations(pred)
            if looks_like_python(code):
                python_candidate += 1
                if python_ast_valid(code):
                    python_ast_ok += 1

    return {
        "total_samples": total,
        "code_like": code_like,
        "code_like_rate": code_like / total if total else 0.0,
        "fenced_code": fenced,
        "fenced_code_rate": fenced / total if total else 0.0,
        "python_candidate": python_candidate,
        "python_candidate_rate": python_candidate / total if total else 0.0,
        "python_ast_valid": python_ast_ok,
        "python_ast_valid_rate_overall": python_ast_ok / total if total else 0.0,
        "python_ast_valid_rate_conditional": (python_ast_ok / python_candidate) if python_candidate else 0.0,
    }


if __name__ == "__main__":
    path = ""
    target_path = "results_new/utility/codeEvol/llama2_7b/clean.json"  # 你按实际路径改

    with open(f"{path}/{target_path}", "r") as f:
        data_list = json.load(f)

    stats = evaluate_codealpaca_syntax(data_list)

    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
