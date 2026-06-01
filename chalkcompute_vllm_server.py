#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute>=1.5.13"]
# ///
"""Long-running vLLM model server — refund-abuse Summit demo.

Serves Qwen2.5-7B-Instruct on a single L4 GPU via an OpenAI-compatible HTTP
API. The agent in chalkcompute_agent_demo.py points at this server instead
of any third-party LLM provider — inference stays inside Chalk Compute, so
prompts and responses never leave the customer's cloud.

Setup (once):
  - HF_TOKEN in .env (free read-token from huggingface.co/settings/tokens)
  - CHALK_* credentials in .env

Run:
  ./chalkcompute_vllm_server.py
  # → prints VLLM_URL; copy it into your .env so the agent can find the server
"""

import chalkcompute


container = chalkcompute.Container(
    image=(
        chalkcompute.Image.base("vllm/vllm-openai:latest")
        .env({
            "HF_HOME": "/root/.cache/huggingface",
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
        })
    ),
    name="qwen-vllm-server",
    gpu="nvidia-l4:1",
    cpu="4",
    memory="16Gi",
    port=8000,
    lifetime="28800s",  # 8 hours; rerun this script if the server expires
    secrets=[chalkcompute.Secret.from_local_env("HF_TOKEN")],
    entrypoint=[
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen2.5-7B-Instruct",
        "--port", "8000",
        "--trust-remote-code",
        "--max-model-len", "4096",
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.90",
    ],
)


if __name__ == "__main__":
    # Reuse an already-running server rather than erroring on the duplicate-name constraint.
    existing = next(
        (c for c in chalkcompute.Container.list_all() if c.name == "qwen-vllm-server"),
        None,
    )
    if existing:
        print("Reusing existing 'qwen-vllm-server' container.")
        handle = chalkcompute.Container.from_name("qwen-vllm-server")
    else:
        # First-time deploy needs ~10–15 min: GPU node provisioning + 6GB image
        # pull + 15GB weight download. 1800s gives plenty of headroom.
        handle = container.run(ready_timeout=1800)

    url = handle.info.web_url
    print(f"\nvLLM server is up:  {url}")
    print(f"Test it:            curl {url}/v1/models")
    print(f"\nNow add this to your .env so the agent can find it:")
    print(f"  VLLM_URL={url}")
