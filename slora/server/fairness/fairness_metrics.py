import time
from collections import defaultdict


class FairnessMetrics:
    """Evaluation metrics from Section 3/5 of the paper — service difference,
    windowed service rate, FTL, etc."""

    def __init__(self, w_p=1, w_q=2, window_size=30.0):
        self.w_p = w_p
        self.w_q = w_q
        self.window_size = window_size

        self._prompt_events = []   # (ts, client_id, input_tokens)
        self._decode_events = []   # (ts, client_id, output_tokens)
        self._arrival_events = []  # (ts, client_id)

        self._start_time = time.time()

    # ############ Event recording ############

    def record_arrival(self, client_id):
        self._arrival_events.append((time.time(), client_id))

    def record_prompt_service(self, client_id, input_tokens):
        self._prompt_events.append((time.time(), client_id, input_tokens))

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
        return self._service_in_range(t - self.window_size, t + self.window_size)

    def _max_service_difference(self, t):
        """max |Wi(0,t) - Wj(0,t)| over all client pairs."""
        service_per_client = self._service_in_range(self._start_time, t)
        if len(service_per_client) < 2:
            return 0.0
        service_values = list(service_per_client.values())
        return max(service_values) - min(service_values)
