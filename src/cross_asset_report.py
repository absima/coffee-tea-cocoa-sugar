from __future__ import annotations

import argparse
import json
import os

import requests


SYSTEM_PROMPT = """You are an analytical assistant writing concise, professional cross-commodity comparison notes.

Rules:
- Use only the provided numbers and facts.
- Do not speculate on causes.
- Do not give investment advice or trading recommendations.
- Compare relative behaviour across commodities directly.
- Call out uncertainty when the model appears weak or compressed.
- Write dates in full, unambiguous form.

Output format (markdown):
1) Cross-commodity behaviour (3-5 bullets)
2) Model comparison (3-5 bullets)
3) Key risks & follow-ups (2-4 bullets)
"""


def CallOllama(model: str, prompt: str, host: str) -> str:
    """
    Send a prompt to Ollama and return the generated comparison note.

    Args:
        model: str.
        Ollama model name.
        prompt: str.
        Full prompt string sent to the model.
        host: str.
        Ollama base URL.

    Returns:
        response_text: str.
        Generated markdown response from the local model.
    """
    url = f"{host}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Failed to call Ollama at {host}. Is Ollama running and is the host correct? Underlying error: {e}"
        ) from e

    data = response.json()
    text = (data.get("response") or "").strip()
    if not text:
        raise RuntimeError(f"Ollama returned an empty response. Raw payload keys: {list(data.keys())}")
    return text


def main(summary_json: str, out_md: str, model: str, host: str) -> None:
    """
    Generate the cross-asset markdown note from the summary bundle.

    Args:
        summary_json: str.
        Input JSON path containing grounded cross-asset facts.
        out_md: str.
        Output markdown path.
        model: str.
        Ollama model name.
        host: str.
        Ollama base URL.

    Returns:
        None.
        Writes the generated cross-asset note to disk.
    """
    with open(summary_json, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Task: Write a short note comparing commodity behaviour and model quality across assets.\n"
        "Use the JSON bundle below as your only source of truth.\n\n"
        f"JSON bundle:\n{json.dumps(bundle, indent=2)}\n"
    )

    note = CallOllama(model=model, prompt=prompt, host=host)

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(note + "\n")

    print(f"Wrote cross-asset LLM note -> {out_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", required=True)
    args = parser.parse_args()

    main(
        summary_json=args.summary_json,
        out_md=args.out_md,
        model=args.model,
        host=args.host,
    )
