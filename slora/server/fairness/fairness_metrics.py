import json
import os
import tempfile
import time
from collections import defaultdict


"""
Note for understanding: While FairQueue implements the VTC scheduling algorithm
(Algorithm 2 in the paper), this class implements the evaluation metrics from
Section 3/5. The metrics here (service difference, demand-capped difference,
windowed service rate, FTL) are what the paper uses to measure whether the
scheduler is actually being fair — i.e. Figures 3 and 4. The RouterManager
calls the record_* methods during request processing.
"""
class FairnessMetrics:

    def __init__(self, w_p_input=1, w_q_output=2, window_T=30.0, log_interval=10):
        self.w_p = w_p_input
        self.w_q = w_q_output
        self.window_T = window_T
        self.log_interval = log_interval

        self._prompt_events = []   # (ts, client_id, input_tokens)
        self._decode_events = []   # (ts, client_id, output_tokens)
        self._ftl_events = []      # (ts, client_id, first_token_latency)
        self._arrival_events = []  # (ts, client_id)

        self._start_time = time.time()
        self._last_log_time = time.time()

    # ############ Event recording ############

    def record_arrival(self, client_id):
        self._arrival_events.append((time.time(), client_id))

    def record_prompt_service(self, client_id, input_tokens):
        self._prompt_events.append((time.time(), client_id, input_tokens))

    def record_first_token_latency(self, client_id, ftl):
        self._ftl_events.append((time.time(), client_id, ftl))

    def record_decode_service(self, client_id, output_tokens):
        self._decode_events.append((time.time(), client_id, output_tokens))

    # ############ Metric computation ############

    def _service_in_range(self, t_start, t_end):
        """Wi = w_p * input_tokens + w_q * output_tokens for events in [t_start, t_end]."""
        service_per_client = defaultdict(float)
        for timestamp, client, token_count in self._prompt_events:
            if t_start <= timestamp <= t_end:
                service_per_client[client] += self.w_p * token_count
        for timestamp, client, token_count in self._decode_events:
            if t_start <= timestamp <= t_end:
                service_per_client[client] += self.w_q * token_count
        return dict(service_per_client)

    def _windowed_service(self, t):
        return self._service_in_range(t - self.window_T, t + self.window_T)

    def _max_service_difference(self, t):
        """max |Wi(0,t) - Wj(0,t)| over all client pairs."""
        service_per_client = self._service_in_range(self._start_time, t)
        if len(service_per_client) < 2:
            return 0.0
        service_values = list(service_per_client.values())
        return max(service_values) - min(service_values)

    def _service_difference_capped(self, t):
        """min(s_high - s_low, |r_low - s_low|) using arrival count as demand proxy."""
        service_per_client = self._service_in_range(self._start_time, t)
        if len(service_per_client) < 2:
            return 0.0

        arrivals_per_client = defaultdict(int)
        for timestamp, client in self._arrival_events:
            if self._start_time <= timestamp <= t:
                arrivals_per_client[client] += 1

        total_arrivals = sum(arrivals_per_client.values()) if arrivals_per_client else 1
        total_service = sum(service_per_client.values()) if service_per_client else 0.0
        lowest_service = min(service_per_client.values())
        highest_service = max(service_per_client.values())
        lowest_service_client = min(service_per_client, key=service_per_client.get)
        demand_proxy = (arrivals_per_client.get(lowest_service_client, 0) / total_arrivals) * total_service if total_arrivals > 0 else 0.0

        return min(highest_service - lowest_service, abs(demand_proxy - lowest_service))

    def _avg_ftl_in_window(self, t):
        # ftl = first-token latency in this case
        window_start, window_end = t - self.window_T, t + self.window_T
        latencies_per_client = defaultdict(list)
        for timestamp, client, latency in self._ftl_events:
            if window_start <= timestamp <= window_end:
                latencies_per_client[client].append(latency)
        return { client: sum(latencies)/len(latencies) for client, latencies in latencies_per_client.items() }

    # ############ Logging ############

    def print_fairness_stats(self):
        now = time.time()
        if now - self._last_log_time < self.log_interval:
            return
        self._last_log_time = now

        cumulative = self._service_in_range(self._start_time, now)
        windowed = self._windowed_service(now)
        max_diff = self._max_service_difference(now)
        capped_diff = self._service_difference_capped(now)
        avg_ftl = self._avg_ftl_in_window(now)

        print("=" * 60)
        print("FAIRNESS METRICS")
        print("-" * 60)
        print(f"  Cumulative service per client: {cumulative}")
        print(f"  Windowed service (T={self.window_T}s):   {windowed}")
        print(f"  Max service difference:        {max_diff:.2f}")
        print(f"  Demand-capped service diff:    {capped_diff:.2f}")
        print(f"  Avg FTL per client (window):   {avg_ftl}")
        print("=" * 60)

    # ############ Persistence ############

    def save_events(self, filepath):
        data = {
            "start_time": self._start_time,
            "w_p": self.w_p, "w_q": self.w_q, "window_T": self.window_T,
            "prompt_events": self._prompt_events,
            "decode_events": self._decode_events,
            "ftl_events": self._ftl_events,
            "arrival_events": self._arrival_events,
        }
        dir_name = os.path.dirname(filepath) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f)
            os.replace(tmp_path, filepath)
        except:
            os.unlink(tmp_path)
            raise
