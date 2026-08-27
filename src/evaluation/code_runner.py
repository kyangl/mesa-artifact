"""Run HumanEval code in a timed subprocess.

The evaluator captures output and rejects imports or builtins that expose the
host. Correctness is binary: all supplied assertions must pass.
"""

import subprocess
import tempfile
import time
import re
from pathlib import Path


# Modules that are forbidden in submitted code (security)
_FORBIDDEN_IMPORTS = [
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "importlib", "builtins", "ctypes", "pickle", "shelve",
    "__import__", "open", "eval", "exec", "compile",
]


def _extract_code_block(text: str) -> str:
    """Extract Python code from markdown code fences if present."""
    # Match ```python ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # If no fences, try to find a def statement and take everything from there
    def_match = re.search(r"(def\s+\w+\s*\(.*)", text, re.DOTALL)
    if def_match:
        return def_match.group(1).strip()

    return text.strip()


def _check_safety(code: str) -> tuple[bool, str]:
    """Basic static check for dangerous imports/calls.

    Returns (is_safe, reason).
    """
    code_lower = code.lower()
    for forbidden in _FORBIDDEN_IMPORTS:
        # Check for "import os", "from os", "os.path", "__import__('os')", etc.
        patterns = [
            f"import {forbidden}",
            f"from {forbidden}",
            f"{forbidden}.",
            f'__import__("{forbidden}")',
            f"__import__('{forbidden}')",
        ]
        for pat in patterns:
            if pat in code_lower:
                return False, f"Forbidden: '{pat}' detected in submitted code"
    return True, ""


def evaluate_code(
    code: str,
    test_suite: list[str],
    imports: str = "",
    timeout: int = 10,
) -> dict:
    """Run submitted code against a list of assert statements.

    Args:
        code:       The Python function implementation (string).
        test_suite: List of assert statements to run as tests.
        imports:    Optional import line(s) to prepend (e.g. "from typing import List").
        timeout:    Max seconds for the subprocess (default 10).

    Returns:
        {
            "passed":   bool,      # True iff all assertions pass
            "n_passed": int,       # Number of assertions that passed
            "n_total":  int,       # Total assertions
            "error":    str|None,  # First error message if any
            "elapsed":  float,     # Wall-clock seconds
        }
    """
    # Extract code from possible markdown fences
    clean_code = _extract_code_block(code)

    # Safety check
    safe, reason = _check_safety(clean_code)
    if not safe:
        return {
            "passed": False, "n_passed": 0, "n_total": len(test_suite),
            "error": f"SAFETY_BLOCK: {reason}", "elapsed": 0.0,
        }

    # Build the test script line-by-line to avoid indentation issues with f-strings
    lines = []
    if imports.strip():
        lines.append(imports.strip())
    lines.append("")
    lines.append(clean_code)
    lines.append("")
    lines.append("_n_passed = 0")
    lines.append("_errors = []")
    lines.append("")

    for i, assertion in enumerate(test_suite):
        safe_assertion = assertion.replace('"', '\\"')
        lines.append("_passed = False")
        lines.append("try:")
        lines.append(f"    {assertion}")
        lines.append("    _passed = True")
        lines.append("except AssertionError as _e:")
        lines.append(f'    _errors.append("Test {i+1} FAILED: {safe_assertion} -- " + str(_e))')
        lines.append("except Exception as _e:")
        lines.append(f'    _errors.append("Test {i+1} ERROR: " + type(_e).__name__ + ": " + str(_e))')
        lines.append("if _passed:")
        lines.append("    _n_passed += 1")
        lines.append("")

    lines.append("import json as _json")
    lines.append(f'print(_json.dumps({{"n_passed": _n_passed, "n_total": {len(test_suite)}, "errors": _errors}}))')

    script = "\n".join(lines)

    start = time.time()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="he_eval_"
        ) as f:
            f.write(script)
            tmp_path = f.name

        proc = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start

        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            return {
                "passed": False, "n_passed": 0, "n_total": len(test_suite),
                "error": f"RUNTIME_ERROR: {stderr[:500]}", "elapsed": elapsed,
            }

        import json
        output = proc.stdout.strip()
        result = json.loads(output)
        n_passed = result["n_passed"]
        errors = result["errors"]

        return {
            "passed": n_passed == len(test_suite),
            "n_passed": n_passed,
            "n_total": len(test_suite),
            "error": errors[0] if errors else None,
            "elapsed": elapsed,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        Path(tmp_path).unlink(missing_ok=True)
        return {
            "passed": False, "n_passed": 0, "n_total": len(test_suite),
            "error": f"TIMEOUT: exceeded {timeout}s", "elapsed": elapsed,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "passed": False, "n_passed": 0, "n_total": len(test_suite),
            "error": f"RUNNER_ERROR: {e}", "elapsed": elapsed,
        }


def evaluate_resolution(resolution: str, task: dict) -> dict:
    """Adapter for MAS runner: extract code from agent resolution and run tests.

    This replaces llm_judge() for the software engineering scenario.

    Returns a scores dict compatible with the rest of the pipeline:
        {"decision_accuracy": 0|1, "action_correctness": 0.0-1.0,
         "reasoning_quality": -1}   # not applicable for code
    """
    ground_truth = task.get("ground_truth", {})
    test_suite = ground_truth.get("test_suite", [])
    imports = task.get("mock_data", {}).get("imports", "")

    if not test_suite:
        return {"decision_accuracy": -1, "action_correctness": -1,
                "reasoning_quality": -1, "code_eval": None}

    result = evaluate_code(resolution, test_suite, imports=imports)

    return {
        "decision_accuracy": 1 if result["passed"] else 0,
        "action_correctness": result["n_passed"] / result["n_total"] if result["n_total"] > 0 else 0,
        "reasoning_quality": -1,  # not applicable — code either works or it doesn't
        "code_eval": result,
    }


if __name__ == "__main__":
    # Quick self-test
    code = """
def has_close_elements(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
"""
    tests = [
        "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False",
        "assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True",
        "assert has_close_elements([], 0.5) == False",
    ]
    r = evaluate_code(code, tests, imports="from typing import List")
    print(f"Result: {r}")
    assert r["passed"], f"Self-test failed: {r}"
    print("Self-test passed.")
