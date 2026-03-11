"""
Unified experiment runner for VTC.
Runs a workload and logs all data (client results + server metrics) to a single JSON file.
"""

import argparse
import asyncio
import json
import time
import sys
import os
import aiohttp
import numpy as np
from tokenizers import Tokenizer

# currently using the same experiment config as in main
# should copy in any experiments you want to run (like Cary's new exp)

FIGURE3_CLIENTS = {
    "client1": {"req_rate": 1.5, "input_len": 256, "output_len": 256},
    "client2": {"req_rate": 3.0, "input_len": 256, "output_len": 256},
}

FIGURE4_CLIENTS = {
    "client1": {"req_rate": 0.25, "input_len": 256, "output_len": 256},
    "client2": {"req_rate": 0.5,  "input_len": 256, "output_len": 256},
    "client3": {"req_rate": 1.5,  "input_len": 256, "output_len": 256},
}

EXP_1 = {
    'client1': {'req_rate': 2.0, 'input_len': 1, 'output_len': 256},
    'client2': {'req_rate': 2.0 / 16.0, 'input_len': 1, 'output_len': 4096},
}

EXPERIMENTS = {
    "1": EXP_1,
    "3": FIGURE3_CLIENTS,
    "4": FIGURE4_CLIENTS,
}

MODEL_DIR = "huggyllama/llama-7b"
METRICS_DUMP = "fairness_metrics_dump.json"

async def client_sender(session, server_url, client_name, cfg, duration, results):
    rate = cfg["req_rate"]
    prompt = cfg['prompt']
    input_len = cfg['input_len']
    output_len = cfg["output_len"]
    start = time.time()
    req_id = 0

    while time.time() - start < duration:
        data = {
            "model_dir": MODEL_DIR,
            "lora_dir": client_name,
            "inputs": prompt,
            "parameters": {
                "do_sample": False,
                "ignore_eos": True,
                "max_new_tokens": output_len,
            },
        }

        req_start = time.time()
        first_token_latency = None
        try:
            async with session.post(f"{server_url}/generate_stream",
                                    headers={"User-Agent": "FigureClient"},
                                    json=data) as response:
                async for chunk, _ in response.content.iter_chunks():
                    if first_token_latency is None:
                        first_token_latency = time.time() - req_start
            total_latency = time.time() - req_start
            results.append({
                "client": client_name,
                "req_id": req_id,
                "input_len": input_len,
                "output_len": output_len,
                "total_latency": total_latency,
                "first_token_latency": first_token_latency,
                "timestamp": req_start - start,
                "wall_clock_start": req_start,
            })
        except Exception as e:
            print(f"[{client_name}] Request {req_id} failed: {e}")

        req_id += 1
        wait = np.random.exponential(1.0 / rate)
        sleep_until = req_start + wait
        now = time.time()
        if sleep_until > now:
            await asyncio.sleep(sleep_until - now)

async def run_workload(clients_cfg, duration, server_url):
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    results = []
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        try:
            async with session.get(f"{server_url}/health") as resp:
                if resp.status != 200:
                    print(f"Server health check failed: {resp.status}")
                    return None
        except Exception as e:
            print(f"Cannot reach server: {e}")
            return None

        tokenizer = Tokenizer.from_pretrained(MODEL_DIR)
        tasks = []
        for client_name, cfg in clients_cfg.items():
            cfg['prompt'] = tokenizer.decode([1234] * cfg['input_len'])
            tasks.append(client_sender(session, server_url, client_name, cfg, duration, results))
        await asyncio.gather(*tasks)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, choices=EXPERIMENTS.keys())
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--output-log", type=str, default=None)
    args = parser.parse_args()

    clients_cfg = EXPERIMENTS[args.exp]
    results = asyncio.run(run_workload(clients_cfg, args.duration, args.server))
    if results is None: return

    print("Waiting for server to flush metrics...")
    time.sleep(2)

    server_metrics = {}
    if os.path.exists(METRICS_DUMP):
        with open(METRICS_DUMP, "r") as f:
            server_metrics = json.load(f)
    else:
        print(f"Warning: {METRICS_DUMP} not found.")

    combined_log = {
        "exp_id": args.exp,
        "duration": args.duration,
        "clients_config": clients_cfg,
        "client_results": results,
        "server_metrics": server_metrics
    }

    out_path = args.output_log or f"exp{args.exp}_log_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(combined_log, f, indent=2)
    print(f"Experiment log saved to {out_path}")

if __name__ == "__main__":
    main()
