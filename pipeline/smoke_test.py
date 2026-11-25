"""Call each provider once, log latency to runs.jsonl. Tells you early what works.

Model ids come from .env because they change often. Anything missing is reported, not crashed on,
and the failure itself is a log line (useful evidence for FAILURES.md).
"""
import base64
import os

import requests

from common import load_env, logged_call

load_env()


def gemini_text():
    key, model = os.environ.get("GEMINI_API_KEY"), os.environ.get("GEMINI_TEXT_MODEL")
    if not (key and model):
        raise RuntimeError("GEMINI_API_KEY / GEMINI_TEXT_MODEL not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = requests.post(url, headers={"x-goog-api-key": key}, timeout=60,
                      json={"contents": [{"parts": [{"text": "Reply with the single word: ready"}]}]})
    r.raise_for_status()
    return r.json()


def gemini_tts():
    key, model = os.environ.get("GEMINI_API_KEY"), os.environ.get("GEMINI_TTS_MODEL")
    if not (key and model):
        raise RuntimeError("GEMINI_API_KEY / GEMINI_TTS_MODEL not set")
    url = "https://generativelanguage.googleapis.com/v1beta/interactions"
    body = {
        "model": model,
        "input": [{
            "type": "user_input",
            "content": [{
                "type": "text",
                "text": "hello",
                "annotations": [{"type": "speech_metadata", "style": "calm"}],
            }],
        }],
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": [{"voice": "Kore"}]},
    }
    r = requests.post(url, headers={"x-goog-api-key": key, "Content-Type": "application/json"}, json=body, timeout=120)
    r.raise_for_status()
    resp = r.json()
    audio_b64 = None
    for step in reversed(resp.get("steps", [])):
        for item in reversed(step.get("content", [])):
            if item.get("type") == "audio" and item.get("data"):
                audio_b64 = item["data"]
                break
    if not audio_b64:
        for item in reversed(resp.get("outputs", [])):
            if item.get("type") == "audio" and item.get("data"):
                audio_b64 = item["data"]
                break
    assert audio_b64 and len(base64.b64decode(audio_b64)) > 1000, "audio too short"
    return True


def cloudflare_image():
    acct, tok = os.environ.get("CLOUDFLARE_ACCOUNT_ID"), os.environ.get("CLOUDFLARE_API_TOKEN")
    model = os.environ.get("CF_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell")
    if not (acct and tok):
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN not set")
    url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{model}"
    r = requests.post(url, headers={"Authorization": f"Bearer {tok}"}, timeout=120,
                      json={"prompt": "a small clay teapot on a wooden table, soft window light", "steps": 4})
    r.raise_for_status()
    return r.json().get("success", True)


CHECKS = [("gemini-text", gemini_text), ("gemini-tts", gemini_tts), ("cloudflare-image", cloudflare_image)]

if __name__ == "__main__":
    for name, fn in CHECKS:
        try:
            logged_call("smoke", name, fn)
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {name}: {e}")
