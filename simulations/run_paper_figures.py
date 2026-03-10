"""
Workload driver and plotter for VTC paper Figures 3 & 4.

Usage:
  # Figure 3 (index 3): start server with --lora-dirs client1 --lora-dirs client2
  python simulations/run_paper_figures.py --exp 3 --duration 300

  # Figure 4 (index 4): start server with --lora-dirs client1 --lora-dirs client2 --lora-dirs client3
  python simulations/run_paper_figures.py --exp 4 --duration 300
"""

import argparse
import asyncio
import json
import time
import sys
import os

import aiohttp
import numpy as np

# ---------------------------------------------------------------------------
# Paper workload parameters
# ---------------------------------------------------------------------------

# Figure 3: 2 clients, equal weight, equal request rate
FIGURE3_CLIENTS = {
    "client1": {"req_rate": 1.5, "input_len": 256, "output_len": 256},
    "client2": {"req_rate": 3.0, "input_len": 256, "output_len": 256},
}

# Figure 4: 3 clients — paper specifies 15, 30, 90 req/min = 0.25, 0.5, 1.5 req/s
# Fixed input=256, output=256. Client 3 is backlogged.
FIGURE4_CLIENTS = {
    "client1": {"req_rate": 0.25, "input_len": 256, "output_len": 256},
    "client2": {"req_rate": 0.5,  "input_len": 256, "output_len": 256},
    "client3": {"req_rate": 1.5,  "input_len": 256, "output_len": 256},
}

EXP_0 = {  # simple exp to find the capacity of the hardware
        'client1': {'req_rate': 100.0, 'input_len': 1, 'output_len': 1},
        'client2': {'req_rate': 100.0, 'input_len': 1, 'output_len': 1},
        'client3': {'req_rate': 100.0, 'input_len': 1, 'output_len': 1},
        'client4': {'req_rate': 100.0, 'input_len': 1, 'output_len': 1},
}

EXP_1 = {  # exp to test difference in service for long output length
         'client1': {'req_rate': 2.0, 'input_len': 1, 'output_len': 256},
         'client2': {'req_rate': 2.0 / 16.0, 'input_len': 1, 'output_len': 4096},
}

# experiment 10: 2 clients, equal amount of requested service, but client2 has much larger-size queries
EXP_10 = {
    "client1": {"req_rate": 3, "input_len": 256, "output_len": 256, }, 
    "client2": {"req_rate": 0.375, "input_len": 2048, "output_len": 2048}, # 2047 since 
}

# experiment 11: 2 clients, equal amount of requested service, but client2 has much larger-size queries
# and client1 has much smaller queries
EXP_11 = {
    "client1": {"req_rate": 48, "input_len": 16, "output_len": 16},
    "client2": {"req_rate": 0.375, "input_len": 2048, "output_len": 2048},
}

EXPERIMENTS = [EXP_0, EXP_1, None, FIGURE3_CLIENTS, FIGURE4_CLIENTS, None, None, None, None, None,
               EXP_10, EXP_11]

# Plot types: "svc_diff", "svc_rate", "resp_time"
PLOT_CONFIGS = [
    ["svc_rate", "resp_time"],  # EXP_1 (index 0)
    None,                       # (index 1)
    None,                       # (index 2)
    ["svc_diff", "svc_rate"],   # Figure 3 (index 3)
    ["svc_rate", "resp_time"],  # Figure 4 (index 4)
    None,                       # (index 5)
    None,                       # (index 6)
    None,                       # (index 7)
    None,                       # (index 8)
    None,                       # (index 9)
    ["svc_diff", "svc_rate"],   # Experiment 10 (index 10)
    ["svc_diff", "svc_rate"]    # Experiment 11 (index 11)
]

SERVER_URL = "http://127.0.0.1:8000"
MODEL_DIR = "huggyllama/llama-7b"
METRICS_DUMP = "fairness_metrics_dump.json"

# ---------------------------------------------------------------------------
# Client coroutine — sends requests at specified rate
# ---------------------------------------------------------------------------

async def client_sender(session, client_name, cfg, duration, results):
    """Send requests at a Poisson-like rate for `duration` seconds."""
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
            async with session.post(f"{SERVER_URL}/generate_stream",
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
            })
        except Exception as e:
            print(f"[{client_name}] Request {req_id} failed: {e}")

        req_id += 1

        # Poisson inter-arrival: exponential with mean 1/rate
        wait = np.random.exponential(1.0 / rate)
        sleep_until = req_start + wait
        now = time.time()
        if sleep_until > now:
            await asyncio.sleep(sleep_until - now)

    print(f"[{client_name}] Finished: sent {req_id} requests in {time.time() - start:.1f}s")


async def run_workload(clients_cfg, duration, server_url):
    """Launch per-client coroutines and wait for all to finish."""
    global SERVER_URL
    SERVER_URL = server_url

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    results = []

    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        # Quick health check
        try:
            async with session.get(f"{SERVER_URL}/health") as resp:
                if resp.status != 200:
                    print(f"Server health check failed with status {resp.status}")
                    return results
        except aiohttp.ClientError as e:
            print(f"Cannot reach server at {SERVER_URL}: {e}")
            return results

        print(f"Server is healthy. Starting workload with {len(clients_cfg)} clients for {duration}s ...")

        tasks = []
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_pretrained('huggyllama/llama-7b')
        for client_name, cfg in clients_cfg.items():
            cfg['prompt'] = tokenizer.decode([1234] * cfg['input_len'])
            # generate the prompt ahead of time with the correct number of tokens
            assert len(tokenizer.encode(cfg['prompt'], add_special_tokens=False).ids) == cfg['input_len']
            tasks.append(client_sender(session, client_name, cfg, duration, results))

        await asyncio.gather(*tasks)

    print(f"Workload complete. Total requests: {len(results)}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_experiment(metrics_path, exp_idx, client_results, output_path=None):
    """Generic plotting: reads PLOT_CONFIGS[exp_idx] and generates subplots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_types = PLOT_CONFIGS[exp_idx]
    if not plot_types:
        print(f"No plot configuration for experiment {exp_idx}.")
        return

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from slora.server.fairness.fairness_metrics import FairnessMetrics

    if not os.path.exists(metrics_path):
        print(f"Metrics file {metrics_path} not found.")
        return

    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    metrics = FairnessMetrics(w_p_input=data["w_p"], w_q_output=data["w_q"],
                              window_T=data["window_T"])
    metrics._start_time = data["start_time"]
    metrics._prompt_events = [tuple(e) for e in data["prompt_events"]]
    metrics._decode_events = [tuple(e) for e in data["decode_events"]]
    metrics._ftl_events = [tuple(e) for e in data["ftl_events"]]
    metrics._arrival_events = [tuple(e) for e in data["arrival_events"]]

    all_times = ([e[0] for e in metrics._prompt_events] +
                 [e[0] for e in metrics._decode_events])
    if not all_times:
        print("No events recorded in metrics file.")
        return

    t_start = min(all_times)
    t_end = max(all_times)
    clients = sorted(set([e[1] for e in metrics._prompt_events] + 
                         [e[1] for e in metrics._decode_events]))

    # Styling
    markers = ["v", "s", "D", "o", "^", "p"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    linestyles = ["-", "--", ":", "-."]

    # Prepare figure
    num_plots = len(plot_types)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    # Shared time calculations
    WINDOW_HALF = 30.0
    SAMPLE_STEP = 10.0
    time_points = np.arange(t_start + WINDOW_HALF, t_end - WINDOW_HALF, SAMPLE_STEP)
    rel_times = time_points - t_start

    for idx, ptype in enumerate(plot_types):
        ax = axes[idx]
        title_prefix = f"({chr(97+idx)}) "

        if ptype == "svc_diff":
            cumul_diff = []
            for t in time_points:
                svc = metrics._service_in_range(t_start, t)
                vals = [svc.get(c, 0.0) for c in clients]
                cumul_diff.append(max(vals) - min(vals) if len(vals) >= 2 else 0.0)
            
            ax.plot(rel_times, cumul_diff, marker="v", markersize=5, color="tab:blue", markevery=2, label="VTC")
            ax.set_ylabel("Absolute Difference in Service")
            ax.set_title(f"{title_prefix}Accumulated service difference")
            ax.legend()

        elif ptype == "svc_rate":
            service_rate_by_client = {c: [] for c in clients}
            for t in time_points:
                svc = metrics._service_in_range(t - WINDOW_HALF, t + WINDOW_HALF)
                for c in clients:
                    service_rate_by_client[c].append(svc.get(c, 0.0) / (2 * WINDOW_HALF))
            
            for i, c in enumerate(clients):
                ax.plot(rel_times, service_rate_by_client[c],
                        marker=markers[i % len(markers)], markersize=5,
                        linestyle=linestyles[i % len(linestyles)],
                        color=colors[i % len(colors)],
                        label=c.replace("client", "Client "), markevery=2)
            ax.set_ylabel("Service (tokens/s)")
            ax.set_title(f"{title_prefix}Received service rate (60s window)")
            ax.legend()

        elif ptype == "resp_time":
            if client_results:
                WINDOW = 30.0
                STEP = 10.0
                t_max_c = max(r["timestamp"] for r in client_results)
                t_pts = np.arange(WINDOW, t_max_c - WINDOW, STEP)

                for i, c in enumerate(clients):
                    c_res = [r for r in client_results if r["client"] == c]
                    avg_rt = []
                    rt_times = []
                    for t in t_pts:
                        in_window = [r["total_latency"] for r in c_res if t - WINDOW <= r["timestamp"] <= t + WINDOW]
                        if in_window:
                            avg_rt.append(np.mean(in_window))
                            rt_times.append(t)
                    if rt_times:
                        ax.plot(rt_times, avg_rt,
                                marker=markers[i % len(markers)], markersize=5,
                                linestyle=linestyles[i % len(linestyles)],
                                color=colors[i % len(colors)],
                                label=c.replace("client", "Client "), markevery=2)
                ax.set_ylabel("Response Time (s)")
                ax.set_title(f"{title_prefix}Response time")
                ax.legend()
            else:
                ax.text(0.5, 0.5, "No client results", ha="center", va="center", transform=ax.transAxes)

        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_path or f"exp{exp_idx}.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def _plot_client_side_only(results, exp_idx, output_path=None):
    """Fallback: plot client-side latency data when server metrics are unavailable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        print("No client results to plot.")
        return

    clients = sorted(set(r["client"] for r in results))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for i, c in enumerate(clients):
        c_results = [r for r in results if r["client"] == c]
        timestamps = [r["timestamp"] for r in c_results]
        total_lat = [r["total_latency"] for r in c_results]
        ftl = [r["first_token_latency"] for r in c_results]
        color = colors[i % len(colors)]
        ax1.scatter(timestamps, total_lat, label=c, color=color, s=10, alpha=0.6)
        ax2.scatter(timestamps, ftl, label=c, color=color, s=10, alpha=0.6)

    ax1.set_ylabel("Total Latency (s)")
    ax1.set_title(f"Experiment {exp_idx}: Per-Client Total Latency (client-side)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("First-Token Latency (s)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title(f"Experiment {exp_idx}: Per-Client First-Token Latency (client-side)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_path or f"exp{exp_idx}_client_only.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VTC Paper Figure Workload Driver & Plotter")
    parser.add_argument("--exp", type=int, required=True,
                        help="Index of the experiment in EXPERIMENTS list")
    parser.add_argument("--duration", type=int, default=300,
                        help="Workload duration in seconds (default: 300)")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000",
                        help="Server URL (default: http://127.0.0.1:8000)")
    parser.add_argument("--metrics-file", type=str, default=METRICS_DUMP,
                        help="Path to fairness_metrics_dump.json (to plot from existing)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output figure path (default: exp{N}.png)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip workload, just plot from existing metrics dump")
    args = parser.parse_args()

    if args.exp < 0 or args.exp >= len(EXPERIMENTS) or EXPERIMENTS[args.exp] is None:
        valid_indices = [i for i, e in enumerate(EXPERIMENTS) if e is not None]
        print(f"Error: --exp must be one of {valid_indices}")
        return

    clients_cfg = EXPERIMENTS[args.exp]

    if args.plot_only:
        # Try to load saved client-side results for response time plots
        client_results_path = args.metrics_file.replace(".json", "_client_results.json")
        saved_results = []
        if os.path.exists(client_results_path):
            with open(client_results_path, "r") as f:
                saved_results = json.load(f)
            print(f"Loaded {len(saved_results)} client results from {client_results_path}")
        plot_experiment(args.metrics_file, args.exp, saved_results, args.output)
        return

    # Run workload
    results = asyncio.run(run_workload(clients_cfg, args.duration, args.server))

    # Wait a moment for the server to flush metrics
    print("Waiting 2s for server to flush metrics ...")
    time.sleep(2)

    # Save timestamped results
    ts = int(time.time())
    metrics_path = f"fairness_metrics_exp{args.exp}_{ts}.json"
    client_results_path = metrics_path.replace(".json", "_client_results.json")

    # Rename the server's dump file to our timestamped version
    if os.path.exists(METRICS_DUMP):
        os.rename(METRICS_DUMP, metrics_path)
        print(f"Server metrics saved to {metrics_path}")
    else:
        print(f"Warning: {METRICS_DUMP} not found. Server-side plotting may fail.")
        metrics_path = METRICS_DUMP

    with open(client_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Client-side results saved to {client_results_path}")

    # Plot
    plot_experiment(metrics_path, args.exp, results, args.output)


if __name__ == "__main__":
    main()
