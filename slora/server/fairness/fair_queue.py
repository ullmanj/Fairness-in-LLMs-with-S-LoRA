from slora.server.router.req_queue import ReqQueue
from slora.server.io_struct import Batch
from slora.server.io_struct import Req
from typing import List
from collections import deque

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

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size, adapter_dirs, fair_weights):
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)
        
        if len(fair_weights) != 2:
            raise ValueError("fair_weights must be a list of two weights")
        if fair_weights[0] <= 0 or fair_weights[1] <= 0:
            raise ValueError("fair_weights must be positive")
        self.w_p_input = fair_weights[0]  # input token weights
        self.w_q_output = fair_weights[1]  # output token weights
        
        self.adapter_dirs = adapter_dirs  # As of now, this is unnecesssary. But it was passed into the predicessor implementation that we cut out without reading, so leaving for now incase we need it in the future for implementation.

        # ASSUME: 1:1 mapping between adapter_dir and client.
        self.requests_by_client: dict[str, List[Req]] = { adapter_dir: [] for adapter_dir in self.adapter_dirs }
        self.counters_by_client: dict[str, int] = { adapter_dir: 0 for adapter_dir in self.adapter_dirs }  # initialize to 0 for all clients

        # Request queue is accessed from self.waiting_req_list in the parent class.

        self.last_client_to_leave_Q = None
        return

    #  ############# Monitoring stream (i.e. new request comes in? Append it. ) #############

    # This will have lines 7-14 of Alg2. See `loop_for_netio_req` from
    # manager.py for the prior steps in this loop.
    def append(self, req: Req):
        # if another request does NOT exist in the queue from the same client
        if len(self.requests_by_client[req.adapter_dir]) == 0:
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
                P = self.requests_by_client.keys()
                # Get the minimum counter for those clients in P and call it c_min
                c_min = min(self.counters_by_client[client] for client in P)
                # Update the counter for this request's client (call it c_u) to the max{ c_u, c_min }
                self.counters_by_client[req.adapter_dir] = max(self.counters_by_client[req.adapter_dir], c_min)

        # enqueue as normal
        super().append(req)
        return

    # ############# With Execution stream (i.e. whose turn is it next? Build a batch.) #############
    
    # This will have lines 18-26 of Alg2. See `_step` from manager.py for the
    # surrounding steps in this loop.
    def generate_new_batch(self, current_batch:Batch, lora_ranks: dict[str, int]):
        # Create an empty batch
        new_batch = super().generate_new_batch(current_batch, lora_ranks) #TODO: Is this correct?

        # while true
        while True:
            # get the client (call it k) whose counter is the smallest among all of the clients in the queue.
            k = min(self.counters_by_client.keys(), key=lambda x: self.counters_by_client[x])
            # let r be the earlist request in the queue from client k.
            r = self.requests_by_client[k][0]
            # if r cannot fit into the batch, break. (because we have filled the batch maximally while staying fair)
            if new_batch.input_tokens() + r.input_len > self.batch_max_tokens:
                break
            # update the counter for client k to be it's current value + w_p_input times the input length of r.
            self.counters_by_client[k] += self.w_p_input * r.input_len
            # Append r to the batch
            new_batch.reqs.append(r)
            # Remove r from the queue
            self.requests_by_client[k].pop(0)
            self.waiting_req_list.remove(r)
        return None  # return the batch


    # Line 30 of Alg2.
    def update_counter(self, batch:Batch):
        # For each of the clients (call it k) in the the batch
            # Find the set of requests in the batch that are from client k (call it "requests")
            # Get the length of "requests" (call it L)
            # Add to the counter for this client the product of w_q_output and L
        return





