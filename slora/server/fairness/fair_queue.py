from slora.server.router.req_queue import ReqQueue
from slora.server.io_struct import Batch
from slora.server.io_struct import Req
from typing import Deque, List
from collections import deque
import uuid

# number of input tokens at which the high tier cost applies
HIGH_TIER_START = 2000
# multiplier applied to base weights when in 'high' tier of tiered scheduling
HIGH_TIER_MULTIPLIER = 2
# number to divide base weight by for 2nd-degree term in quadratic cost function
QUADRATIC_DIVISOR = 1000 

"""
Note for understanding: While the algorithm lays out the logic in "while" loops,
in s-lora, these loops are already implemented by the RouterManager in
loop_for_netio_req() and _stepp(). So, this queue just needs to expose the
logic from within those loops for each of those call sites.
"""
class FairQueue(ReqQueue):
    """
    A version of the request queue that implements the VTC-based fairness
    scheduler. The outline for this is in Algorithm 2 in the paper.
    """
    # 'service_def' is a string argument that takes on one of three string values: "linear", "tiered", and "quadratic".
    # "linear" corresponds to the original algorithm in VTC, "tiered" uses a service definition based on modern, large-context
    # pricing models where input & output tokens cost 2x more if the # of input tokens is more than 2k, and "quadratic"
    # uses a quadratic function to most accurately estimate the system resources of a given job.
    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size, adapter_dirs, fair_weights, service_def):
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)

        self.service_def = service_def
        self.w_p_input = 1  # input token weights
        self.w_q_output = 2  # output token weights
        
        self.adapter_dirs = adapter_dirs  # WE ASSUME: 1:1 mapping between adapter_dir and client.

        self.counters_by_client: dict[str, int] = { adapter_dir: 0 for adapter_dir in self.adapter_dirs }  # initialize to 0 for all clients

        # Fair Queue uses it's own custom queue, as opposed to the parent `waiting_req_list`.
        self.queued_requests = QueuedRequests()

        # NOTE: This is unused. If we want to incoroprate tiered fariness, we can incorporate this in the implemenetation below.
        self.fair_weights = fair_weights
        if len(fair_weights) > 0:
            print("WARNING: Fair weights detected in input. These are not currently implemented in VTC algorithm, though it would be a quick fix to add them.")

        return

    # Add these getters so that we can pretend to still have waiting_req_list, as an interface.
    @property
    def waiting_req_list(self) -> List[Req]:
        # This is O(N), but is only called in the abort case, so doesn't matter. Just flatten the per-client queues. Order doesn't matter.
        return [request for client_requests in self.queued_requests.queued_requests_by_client.values() for request in client_requests]

    @waiting_req_list.setter
    def waiting_req_list(self, value):
        raise RuntimeError("ERROR: waiting_req_list is not used in FairQueue. It's setter should not be called.")

    #  ############# Monitoring stream (i.e. new request comes in? Append it. ) #############

    # This will have lines 7-14 of Alg2. See `loop_for_netio_req` from
    # manager.py for the prior steps in this loop.
    def append(self, req: Req):
        # if another request does NOT exist in the queue from the same client
        if self.queued_requests.get_earliest_request_from_client(req.adapter_dir) is None:
            # if the queue is empty
            if self.queued_requests.is_empty():
                # get the counter of the last client who left the Q (call it c_l)
                c_l = 0
                if self.queued_requests.client_that_made_Q_empty_when_last_request_left is not None:
                    c_l = self.counters_by_client[self.queued_requests.client_that_made_Q_empty_when_last_request_left]
                
                # set the counter for this request's client (call it c_u) to the max{ c_u, c_l }
                self.counters_by_client[req.adapter_dir] = max(self.counters_by_client[req.adapter_dir], c_l)
            # else
            else:
                # Build a set "P" of all the clients who have a request in the queue
                P = self.queued_requests.get_clients_in_queue()
                # Get the minimum counter for those clients in P and call it c_min
                c_min = min(self.counters_by_client[client] for client in P)
                # Update the counter for this request's client (call it c_u) to the max{ c_u, c_min }
                self.counters_by_client[req.adapter_dir] = max(self.counters_by_client[req.adapter_dir], c_min)

        # enqueue as normal
        self.queued_requests.push_request(req)
        return

    # ############# With Execution stream (i.e. whose turn is it next? Build a batch.) #############
    
    # This will have lines 18-26 of Alg2. See `_step` from manager.py for the
    # surrounding steps in this loop.
    def generate_new_batch(self, current_batch:Batch, lora_ranks: dict[str, int]):
        # Not in algorithm: current batch checks and cache init. Inspired by super()
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        self._init_cache_list(current_batch, lora_ranks)

        # Create an empty "batch"
        new_batch_reqs = []
        new_batch_total_tokens = 0

        # while true
        while True:
            # get the client (call it k) whose counter is the smallest among all of the clients in the queue.
            clients_in_queue = self.queued_requests.get_clients_in_queue()
            if len(clients_in_queue) == 0:
                break

            k = min(clients_in_queue, key=lambda client: self.counters_by_client[client])
            # let r be the earlist request in the queue from client k.
            r = self.queued_requests.get_earliest_request_from_client(k)

            # Not in algorithm: Drop the aborted requests. Inspired by super()
            if r.aborted:
                self.queued_requests.remove_request(r)
                continue
            
            # if r cannot fit into the batch, break. (because we have filled the batch maximally while staying fair)
            if not (self._can_add_new_req(r, lora_ranks) and  # Not in algorithm: can_add_new_req check. Inspired by super()
                new_batch_total_tokens + r.input_len <= self.batch_max_tokens):
                break
            # update the counter for client k to be it's current value + w_p_input times the input length of r.
            if self.service_def == "linear":
                self.counters_by_client[k] += self.w_p_input * r.input_len
            elif self.service_def == "tiered":
                self.counters_by_client[k] += 2 * self.w_p_input * r.input_len if r.input_len >= HIGH_TIER_START else self.w_p_input * r.input_len
            elif self.service_def == "quadratic":
                self.counters_by_client[k] += self.w_p_input * r.input_len + (self.w_p_input * r.input_len * r.input_len / QUADRATIC_DIVISOR)
            # Append r to the batch
            new_batch_reqs.append(r)
            new_batch_total_tokens += r.input_len
            # Remove r from the queue
            self.queued_requests.remove_request(r)
        
        if len(new_batch_reqs) == 0:
            return None
        new_batch = Batch(uuid.uuid4().hex, new_batch_reqs)
        return new_batch  # return the batch


    # Line 30 of Alg2.
    def update_counter(self, batch:Batch):
        # For each of the clients (call it k) in the the batch
        if self.service_def == "linear":
            requests_by_client = self._get_requests_by_client_from(batch.reqs)
            for client in requests_by_client.keys():
                # Find the set of requests in the batch that are from client k (call it "requests")
                # Get the length of "requests" (call it L)
                L = len(requests_by_client[client])
                # Add to the counter for this client the product of w_q_output and L
                self.counters_by_client[client] += self.w_q_output * L
            return
        elif self.service_def == "tiered":
            for req in batch.reqs:
                self.counters_by_client[req.adapter_dir] += 2 * self.w_q_output if req.input_len >= HIGH_TIER_START else self.w_q_output
        elif self.service_def == "quadratic":
            for req in batch.reqs:
                req.output_len += 1
                self.counters_by_client[req.adapter_dir] += self.w_q_output + (self.w_q_output * req.output_len / QUADRATIC_DIVISOR)
    
    # ############ OVERRIDE HELPER FUNCTIONS ############
    
    #  _init_cache_list(self, current_batch:Batch, lora_ranks and
    # _can_add_new_req(self, req, lora_ranks) are fine as is in super. They just
    # check resource limits given the selected requests by this VTC algorithm.
    # So they are independent.

    # ############ OVERRIDE OTHER FUNCTIONS ############
    def next_batch(self):
        print("WARNING: Next batch does not incorporate fairness logic in it's estimate, so should not be used.")
        return super().next_batch()
    

    # ############ HELPER FUNCTIONS ############
    def _get_requests_by_client_from(self, requests: List[Req]) -> dict[str, List[Req]]:
        requests_by_client = { }
        for req in requests:
            if req.adapter_dir not in requests_by_client:
                requests_by_client[req.adapter_dir] = [req]
            else:
                requests_by_client[req.adapter_dir].append(req)
        return requests_by_client


class QueuedRequests:
    def __init__(self):
        # We maintain per-client FIFO queues for efficient access at the high
        # level of requests that are needed to reach capacity on GPUs today.
        self.queued_requests_by_client: dict[str, Deque[Req]] = { }
        self.client_that_made_Q_empty_when_last_request_left = None
    
    def get_earliest_request_from_client(self, client: str) -> Req:
        requests = self.queued_requests_by_client.get(client, None)
        if requests is None:
            return None
        return requests[0]

    def is_empty(self) -> bool:
        # This works because we remove clients from the queue when they have no requests left.
        return len(self.queued_requests_by_client) == 0

    def push_request(self, request: Req):
        if self.queued_requests_by_client.get(request.adapter_dir, None) is None:
            self.queued_requests_by_client[request.adapter_dir] = deque([request])
        else:
            self.queued_requests_by_client[request.adapter_dir].append(request)

    def remove_request(self, request: Req):
        requests = self.queued_requests_by_client.get(request.adapter_dir, None)
        if requests is None:
            raise ValueError(f"Queue for client {request.adapter_dir} does not exist / is empty.")
        if requests[0] != request:
            requests.remove(request)
            print(f"WARNING: Request {request.request_id} is not in the 0th position of the client's queue as expected {request.adapter_dir}.")
        else:
            requests.popleft()

        # remove that client from our queue to reflect in the keys that there are no requests from this client in the queue.
        if len(requests) == 0:
            self.queued_requests_by_client.pop(request.adapter_dir)
        
        if self.is_empty():
            self.client_that_made_Q_empty_when_last_request_left = request.adapter_dir
        else:
            self.client_that_made_Q_empty_when_last_request_left = None
    
    def get_clients_in_queue(self) -> List[str]:
        return list(self.queued_requests_by_client.keys())
    