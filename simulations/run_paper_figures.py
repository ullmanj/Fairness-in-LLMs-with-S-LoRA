"""
Workload driver and plotter for VTC paper Figures 3 & 4.

Usage:
  # Figure 3 (2 clients): start server with --lora-dirs client1 --lora-dirs client2
  python simulations/run_paper_figures.py --figure 3 --duration 300

  # Figure 4 (3 clients): start server with --lora-dirs client1 --lora-dirs client2 --lora-dirs client3
  python simulations/run_paper_figures.py --figure 4 --duration 300
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
    "client1": {"req_rate": 2.0, "input_range": (64, 256), "output_range": (32, 128)},
    "client2": {"req_rate": 2.0, "input_range": (64, 256), "output_range": (32, 128)},
}

# Figure 4: 3 clients — paper specifies 15, 30, 90 req/min = 0.25, 0.5, 1.5 req/s
# Fixed input=256, output=256. Client 3 is backlogged.
FIGURE4_CLIENTS = {
    "client1": {"req_rate": 0.25, "input_range": (256, 256), "output_range": (256, 256)},
    "client2": {"req_rate": 0.5,  "input_range": (256, 256), "output_range": (256, 256)},
    "client3": {"req_rate": 1.5,  "input_range": (256, 256), "output_range": (256, 256)},
}

SERVER_URL = "http://127.0.0.1:8000"
MODEL_DIR = "huggyllama/llama-7b"
METRICS_DUMP = "fairness_metrics_dump.json"

# ---------------------------------------------------------------------------
# Client coroutine — sends requests at specified rate
# ---------------------------------------------------------------------------

async def client_sender(session, client_name, cfg, duration, results):
    """Send requests at a Poisson-like rate for `duration` seconds."""
    rate = cfg["req_rate"]
    input_lo, input_hi = cfg["input_range"]
    output_lo, output_hi = cfg["output_range"]
    start = time.time()
    req_id = 0

    while time.time() - start < duration:
        # Random prompt length and output length
        input_len = int(np.random.randint(input_lo, input_hi + 1))
        output_len = int(np.random.randint(output_lo, output_hi + 1))

        prompt = "Hello " * input_len  # dummy prompt tokens

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
        for client_name, cfg in clients_cfg.items():
            tasks.append(client_sender(session, client_name, cfg, duration, results))

        await asyncio.gather(*tasks)

    print(f"Workload complete. Total requests: {len(results)}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_figures(metrics_path, figure_num, client_results, output_path=None):
    """Load server-side metrics and plot to match the paper's figure layout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from slora.server.fairness.fairness_metrics import FairnessMetrics

    if not os.path.exists(metrics_path):
        print(f"Metrics file {metrics_path} not found — skipping server-side plots.")
        print("Plotting client-side results only ...")
        _plot_client_side_only(client_results, figure_num, output_path)
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

    t_first_event = min(all_times)
    t_max = max(all_times)

    clients = sorted(set(
        [e[1] for e in metrics._prompt_events] +
        [e[1] for e in metrics._decode_events]
    ))

    # --- Compute windowed service rate at regular sample points ---
    WINDOW_HALF = 30.0   # 60s window (±30s), matching the paper
    SAMPLE_STEP = 10.0   # sample every 10s
    time_points = np.arange(t_first_event + WINDOW_HALF, t_max - WINDOW_HALF, SAMPLE_STEP)

    service_rate_by_client = {c: [] for c in clients}
    cumul_diff = []
    relative_times = []

    for t in time_points:
        rel = t - t_first_event
        relative_times.append(rel)

        svc = metrics._service_in_range(t - WINDOW_HALF, t + WINDOW_HALF)
        for c in clients:
            service_rate_by_client[c].append(svc.get(c, 0.0) / (2 * WINDOW_HALF))

        cumul_svc = metrics._service_in_range(t_first_event, t)
        vals = [cumul_svc.get(c, 0.0) for c in clients]
        cumul_diff.append(max(vals) - min(vals) if len(vals) >= 2 else 0.0)

    relative_times = np.array(relative_times)

    markers = ["v", "s", "D", "o"]
    linestyles = ["-", "--", ":"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    if figure_num == 3:
        _plot_figure3(relative_times, cumul_diff, service_rate_by_client,
                      clients, markers, colors, output_path)
    else:
        _plot_figure4(relative_times, service_rate_by_client,
                      client_results, clients, markers, linestyles, colors,
                      output_path)


def _plot_figure3(relative_times, cumul_diff, service_rate_by_client,
                  clients, markers, colors, output_path):
    """Paper Figure 3: (a) accumulated service difference, (b) service rate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Absolute difference in accumulated service
    ax1.plot(relative_times, cumul_diff, marker="v", markersize=5,
             color="tab:blue", label="VTC", markevery=2)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Absolute Difference in Service")
    ax1.set_title("(a) Accumulated service difference")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (b) Received service rate per client
    for i, c in enumerate(clients):
        ax2.plot(relative_times, service_rate_by_client[c],
                 marker=markers[i % len(markers)], markersize=5,
                 color=colors[i % len(colors)],
                 label=c.replace("client", "Client "), markevery=2)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Service")
    ax2.set_title("(b) Received service rate (avg of 60s windows, VTC)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Figure 3", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = output_path or "figure3.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def _plot_figure4(relative_times, service_rate_by_client, client_results,
                  clients, markers, linestyles, colors, output_path):
    """Paper Figure 4: (a) service rate, (b) response time per client."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Received service rate per client
    for i, c in enumerate(clients):
        ax1.plot(relative_times, service_rate_by_client[c],
                 marker=markers[i % len(markers)], markersize=5,
                 linestyle=linestyles[i % len(linestyles)],
                 color=colors[i % len(colors)],
                 label=c.replace("client", "Client "), markevery=2)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Service")
    ax1.set_title("(a) Received service rate (VTC)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (b) Response time per client — windowed average of total latency
    # Compute from client-side results
    if client_results:
        WINDOW = 30.0
        STEP = 10.0
        t_max_client = max(r["timestamp"] for r in client_results)
        t_points = np.arange(WINDOW, t_max_client - WINDOW, STEP)

        for i, c in enumerate(clients):
            c_results = [r for r in client_results if r["client"] == c]
            avg_rt = []
            rt_times = []
            for t in t_points:
                in_window = [r["total_latency"] for r in c_results
                             if t - WINDOW <= r["timestamp"] <= t + WINDOW]
                if in_window:
                    avg_rt.append(np.mean(in_window))
                    rt_times.append(t)
            if rt_times:
                ax2.plot(rt_times, avg_rt,
                         marker=markers[i % len(markers)], markersize=5,
                         linestyle=linestyles[i % len(linestyles)],
                         color=colors[i % len(colors)],
                         label=c.replace("client", "Client "), markevery=2)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Response Time (s)")
        ax2.set_title("(b) Response time (VTC)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No client-side results\n(use --plot-only with client results)",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("(b) Response time (VTC)")

    fig.suptitle("Figure 4", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = output_path or "figure4.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def _plot_client_side_only(results, figure_num, output_path=None):
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
    ax1.set_title(f"Figure {figure_num}: Per-Client Total Latency (client-side)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("First-Token Latency (s)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title(f"Figure {figure_num}: Per-Client First-Token Latency (client-side)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_path or f"figure{figure_num}.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VTC Paper Figure Workload Driver & Plotter")
    parser.add_argument("--figure", type=int, required=True, choices=[3, 4],
                        help="Which figure to reproduce (3 or 4)")
    parser.add_argument("--duration", type=int, default=300,
                        help="Workload duration in seconds (default: 300)")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000",
                        help="Server URL (default: http://127.0.0.1:8000)")
    parser.add_argument("--metrics-file", type=str, default=METRICS_DUMP,
                        help="Path to fairness_metrics_dump.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output figure path (default: figure{N}.png)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip workload, just plot from existing metrics dump")
    args = parser.parse_args()

    clients_cfg = FIGURE3_CLIENTS if args.figure == 3 else FIGURE4_CLIENTS

    if args.plot_only:
        # Try to load saved client-side results for response time plots
        client_results_path = f"figure{args.figure}_client_results.json"
        saved_results = []
        if os.path.exists(client_results_path):
            with open(client_results_path, "r") as f:
                saved_results = json.load(f)
            print(f"Loaded {len(saved_results)} client results from {client_results_path}")
        plot_figures(args.metrics_file, args.figure, saved_results, args.output)
        return

    # Run workload
    results = asyncio.run(run_workload(clients_cfg, args.duration, args.server))

    # Save client-side results
    client_results_path = f"figure{args.figure}_client_results.json"
    with open(client_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Client-side results saved to {client_results_path}")

    # Wait a moment for the server to flush metrics
    print("Waiting 2s for server to flush metrics ...")
    time.sleep(2)

    # Plot
    plot_figures(args.metrics_file, args.figure, results, args.output)


if __name__ == "__main__":
    main()
