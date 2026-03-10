"""
Unified plotter for VTC experiment results.
Reads a consolidated experiment log and generates specified subplots.
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def load_log(log_path):
    with open(log_path, 'r') as f:
        return json.load(f)

def plot_svc_diff(ax, metrics_data, title_prefix=""):
    """Accumulated service difference."""
    prompt_events = metrics_data.get("prompt_events", [])
    decode_events = metrics_data.get("decode_events", [])
    w_p = metrics_data.get("w_p", 1)
    w_q = metrics_data.get("w_q", 2)
    
    all_times = [e[0] for e in prompt_events] + [e[0] for e in decode_events]
    if not all_times:
        ax.text(0.5, 0.5, "No service events", ha="center", va="center", transform=ax.transAxes)
        return

    t_start = min(all_times)
    t_end = max(all_times)
    clients = sorted(set([e[1] for e in prompt_events] + [e[1] for e in decode_events]))
    
    SAMPLE_STEP = 5.0
    time_points = np.arange(t_start, t_end, SAMPLE_STEP)
    rel_times = time_points - t_start
    
    cumul_diff = []
    for t in time_points:
        svc = {}
        for ts, c, tokens in prompt_events:
            if ts <= t:
                svc[c] = svc.get(c, 0.0) + w_p * tokens
        for ts, c, tokens in decode_events:
            if ts <= t:
                svc[c] = svc.get(c, 0.0) + w_q * tokens
        
        vals = [svc.get(c, 0.0) for c in clients]
        cumul_diff.append(max(vals) - min(vals) if len(vals) >= 2 else 0.0)
    
    ax.plot(rel_times, cumul_diff, marker="v", markersize=5, color="tab:blue", markevery=max(1, len(rel_times)//15), label="VTC")
    ax.set_ylabel("Absolute Difference in Service")
    ax.set_title(f"{title_prefix}Accumulated service difference")
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_svc_rate(ax, metrics_data, title_prefix=""):
    """Windowed service rate."""
    prompt_events = metrics_data.get("prompt_events", [])
    decode_events = metrics_data.get("decode_events", [])
    w_p = metrics_data.get("w_p", 1)
    w_q = metrics_data.get("w_q", 2)
    WINDOW_HALF = 30.0
    
    all_times = [e[0] for e in prompt_events] + [e[0] for e in decode_events]
    if not all_times:
        ax.text(0.5, 0.5, "No service events", ha="center", va="center", transform=ax.transAxes)
        return

    t_start = min(all_times)
    t_end = max(all_times)
    clients = sorted(set([e[1] for e in prompt_events] + [e[1] for e in decode_events]))
    
    SAMPLE_STEP = 5.0
    time_points = np.arange(t_start + WINDOW_HALF, t_end - WINDOW_HALF, SAMPLE_STEP)
    rel_times = time_points - t_start
    
    markers = ["v", "s", "D", "o", "^", "p"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    
    for i, c in enumerate(clients):
        rates = []
        for t in time_points:
            window_svc = 0.0
            for ts, client, tokens in prompt_events:
                if t - WINDOW_HALF <= ts <= t + WINDOW_HALF and client == c:
                    window_svc += w_p * tokens
            for ts, client, tokens in decode_events:
                if t - WINDOW_HALF <= ts <= t + WINDOW_HALF and client == c:
                    window_svc += w_q * tokens
            rates.append(window_svc / (2 * WINDOW_HALF))
        
        ax.plot(rel_times, rates, label=c.replace("client", "Client "),
                marker=markers[i % len(markers)], color=colors[i % len(colors)],
                markersize=5, markevery=max(1, len(rel_times)//15))
                
    ax.set_ylabel("Service (tokens/s)")
    ax.set_title(f"{title_prefix}Received service rate (60s window)")
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_resp_time(ax, client_results, title_prefix=""):
    """Response time plot from client-side results."""
    if not client_results:
        ax.text(0.5, 0.5, "No client results", ha="center", va="center", transform=ax.transAxes)
        return

    clients = sorted(set(r["client"] for r in client_results))
    WINDOW = 30.0
    STEP = 10.0
    t_max = max(r["timestamp"] for r in client_results)
    t_pts = np.arange(WINDOW, t_max - WINDOW, STEP)

    markers = ["v", "s", "D", "o", "^", "p"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

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
            ax.plot(rt_times, avg_rt, label=c.replace("client", "Client "),
                    marker=markers[i % len(markers)], color=colors[i % len(colors)],
                    markersize=5, markevery=max(1, len(rt_times)//10))

    ax.set_ylabel("Response Time (s)")
    ax.set_title(f"{title_prefix}Response time")
    ax.legend()
    ax.grid(True, alpha=0.3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True, help="Path to consolidated experiment log")
    parser.add_argument("--plots", type=str, nargs=2, required=True, 
                        choices=["svc_diff", "svc_rate", "resp_time"],
                        help="Two plot types to generate")
    parser.add_argument("--output", type=str, default="experiment_results.png")
    args = parser.parse_args()

    data = load_log(args.log)
    metrics_data = data.get("server_metrics", {})
    client_results = data.get("client_results", [])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    plot_map = {
        "svc_diff": lambda ax, pref: plot_svc_diff(ax, metrics_data, pref),
        "svc_rate": lambda ax, pref: plot_svc_rate(ax, metrics_data, pref),
        "resp_time": lambda ax, pref: plot_resp_time(ax, client_results, pref)
    }

    for idx, ptype in enumerate(args.plots):
        prefix = f"({chr(97+idx)}) "
        plot_map[ptype](axes[idx], prefix)
        axes[idx].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

if __name__ == "__main__":
    main()
