from logging import Logger
from slora.server.router.req_queue import ReqQueue
from slora.server.io_struct import Batch
from slora.server.io_struct import Req
from typing import List
import uuid

"""
Note for understanding: While the algorithm lays out the logic in "while" loops,
in s-lora, these loops are already implemented by the RouterManager in
loop_for_netio_req() and _stepp(). So, this queue just needs to expose the
logic from within those loops for each of those call sites.
"""

# REMAINING TODO:
# - Check the fair weights input — I made assumptions about what shape it would be, but have to confirm.
# - Popuplate the last_client_to_leave_Q when the client leaves the queue (maybe in making the batch?).
# - Test and confirm functionality.
# - Confirm the caching assumptions.

class FairQueue(ReqQueue):
    """
    A version of the request queue that implements the VTC-based fairness
    scheduler. The outline for this is in Algorithm 2 in the paper.
    """

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size, adapter_dirs, fair_weights):
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)
        
        if len(fair_weights) != 2:
            raise ValueError("fair_weights must be a list of two weights")
        if fair_weights[0] <= 0 or fair_weights[1] <= 0:
            raise ValueError("fair_weights must be positive")
        self.w_p_input = fair_weights[0]  # input token weights
        self.w_q_output = fair_weights[1]  # output token weights
        
        self.adapter_dirs = adapter_dirs  # As of now, this is unnecesssary. But it was passed into the predicessor implementation that we cut out without reading, so leaving for now incase we need it in the future for implementation.

        # WE ASSUME: 1:1 mapping between adapter_dir and client.
        self.counters_by_client: dict[str, int] = { adapter_dir: 0 for adapter_dir in self.adapter_dirs }  # initialize to 0 for all clients

        # Request queue is accessed from self.waiting_req_list in the parent class.

        self.last_client_to_leave_Q = None
        return

    #  ############# Monitoring stream (i.e. new request comes in? Append it. ) #############

    # This will have lines 7-14 of Alg2. See `loop_for_netio_req` from
    # manager.py for the prior steps in this loop.
    def append(self, req: Req):
        requests_by_client = self._get_requests_by_client()
        # if another request does NOT exist in the queue from the same client
        if len(requests_by_client[req.adapter_dir]) == 0:
            # if the queue is empty
            if len(self.waiting_req_list) == 0:
                # get the counter of the last client who left the Q (call it c_l)
                c_l = 0
                if self.last_client_to_leave_Q is not None:
                    c_l = self.counters_by_client[self.last_client_to_leave_Q]
                
                # set the counter for this request's client (call it c_u) to the max{ c_u, c_l }
                self.counters_by_client[req.adapter_dir] = max(self.counters_by_client[req.adapter_dir], c_l)
            # else
            else:
                # Build a set "P" of all the clients who have a request in the queue
                P = requests_by_client.keys()
                # Get the minimum counter for those clients in P and call it c_min
                c_min = min(self.counters_by_client[client] for client in P)
                # Update the counter for this request's client (call it c_u) to the max{ c_u, c_min }
                self.counters_by_client[req.adapter_dir] = max(self.counters_by_client[req.adapter_dir], c_min)

        # enqueue as normal into the waiting_req_list (queue source of truth)
        super().append(req)
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
            k = min(self.counters_by_client.keys(), key=lambda x: self.counters_by_client[x])
            # let r be the earlist request in the queue from client k.
            r = self._get_earliest_request_from_client(k)

            # Not in algorithm: Drop the aborted requests. Inspired by super()
            if r.aborted:
                self.waiting_req_list.remove(r)
                continue
            
            # if r cannot fit into the batch, break. (because we have filled the batch maximally while staying fair)
            if (self._can_add_new_req(r, lora_ranks) and  # Not in algorithm: can_add_new_req check. Inspired by super()
                new_batch_total_tokens + r.input_len <= self.batch_max_tokens):
                break
            # update the counter for client k to be it's current value + w_p_input times the input length of r.
            self.counters_by_client[k] += self.w_p_input * r.input_len
            # Append r to the batch
            new_batch_reqs.append(r)
            new_batch_total_tokens += r.input_len
            # Remove r from the queue
            self.waiting_req_list.remove(r)
        
        new_batch = Batch(uuid.uuid4().hex, new_batch_reqs)
        return new_batch  # return the batch


    # Line 30 of Alg2.
    def update_counter(self, batch:Batch):
        # For each of the clients (call it k) in the the batch
        requests_by_client = self._get_requests_by_client()
        for client in requests_by_client.keys():
            # Find the set of requests in the batch that are from client k (call it "requests")
            # Get the length of "requests" (call it L)
            L = len(requests_by_client[client])
            # Add to the counter for this client the product of w_q_output and L
            self.counters_by_client[client] += self.w_q_output * L
        return
    

    # ############ HELPER FUNCTIONS ############
    # Format the queue into a dictionary of requests by client.
    def _get_requests_by_client(self) -> dict[str, List[Req]]:
        requests_by_client = { adapter_dir: [] for adapter_dir in self.adapter_dirs }
        for req in self.waiting_req_list:
            requests_by_client[req.adapter_dir].append(req)
        return requests_by_client
    # Get the earliest request from a client.
    def _get_earliest_request_from_client(self, client: str) -> Req:
        for req in self.waiting_req_list:
            if req.adapter_dir == client:
                return req
        return None
    
    # ############ OVERRIDE HELPER FUNCTIONS ############
    
    #  _init_cache_list(self, current_batch:Batch, lora_ranks and
    # _can_add_new_req(self, req, lora_ranks) are fine as is in super. The cache
    # fairness logic does not rely on the caching (though maybe we could improve
    # the caching to reflect the VTC logic for more accurate caching? I have to
    # understand it more.

    # ############ OVERRIDE OTHER FUNCTIONS ############
    def next_batch(self):
        Logger.warning("Next batch does not incorporate fairness logic in it's estimate.")
        return super().next_batch()



