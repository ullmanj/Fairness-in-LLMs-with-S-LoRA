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
# - Test and confirm functionality.

class FairQueue(ReqQueue):
    """
    A version of the request queue that implements the VTC-based fairness
    scheduler. The outline for this is in Algorithm 2 in the paper.
    """

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size, adapter_dirs, fair_weights):
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)
        
        self.w_p_input = 1  # input token weights
        self.w_q_output = 2  # output token weights
        
        self.adapter_dirs = adapter_dirs  # WE ASSUME: 1:1 mapping between adapter_dir and client.

        self.counters_by_client: dict[str, int] = { adapter_dir: 0 for adapter_dir in self.adapter_dirs }  # initialize to 0 for all clients

        # Our own shadow queue whose functions will update the parent queue (self.waiting_req_list) as well.
        self.queued_requests = QueuedRequests()

        # NOTE: This is unused. If we want to incoroprate tiered fariness, we can incorporate this in the implemenetation below.
        self.fair_weights = fair_weights

        return

    #  ############# Monitoring stream (i.e. new request comes in? Append it. ) #############

    # This will have lines 7-14 of Alg2. See `loop_for_netio_req` from
    # manager.py for the prior steps in this loop.
    def append(self, req: Req):
        # if another request does NOT exist in the queue from the same client
        if self.queued_requests.get_earliest_request_from_client(req.adapter_dir) is None:
            # if the queue is empty
            if len(self.waiting_req_list) == 0:
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
        self.queued_requests.push_request(req, self.waiting_req_list)
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
                self.queued_requests.remove_request(r, self.waiting_req_list)
                continue
            
            # if r cannot fit into the batch, break. (because we have filled the batch maximally while staying fair)
            if not (self._can_add_new_req(r, lora_ranks) and  # Not in algorithm: can_add_new_req check. Inspired by super()
                new_batch_total_tokens + r.input_len <= self.batch_max_tokens):
                break
            # update the counter for client k to be it's current value + w_p_input times the input length of r.
            self.counters_by_client[k] += self.w_p_input * r.input_len
            # Append r to the batch
            new_batch_reqs.append(r)
            new_batch_total_tokens += r.input_len
            # Remove r from the queue
            self.queued_requests.remove_request(r, self.waiting_req_list)
        
        if len(new_batch_reqs) == 0:
            return None
        new_batch = Batch(uuid.uuid4().hex, new_batch_reqs)
        return new_batch  # return the batch


    # Line 30 of Alg2.
    def update_counter(self, batch:Batch):
        # For each of the clients (call it k) in the the batch
        requests_by_client = self._get_requests_by_client_from(batch.reqs)
        for client in requests_by_client.keys():
            # Find the set of requests in the batch that are from client k (call it "requests")
            # Get the length of "requests" (call it L)
            L = len(requests_by_client[client])
            # Add to the counter for this client the product of w_q_output and L
            self.counters_by_client[client] += self.w_q_output * L
        return
    
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
        # need to maintain an official queue because the ReqQueue official queue
        # is accessed by other parts of the code.
        # This should also stay ordered, so each list is a FIFO queue.
        self.queued_requests_by_client: dict[str, List[Req]] = { }
        self.client_that_made_Q_empty_when_last_request_left = None
    
    def get_earliest_request_from_client(self, client: str) -> Req:
        requests = self.queued_requests_by_client.get(client, None)
        if requests is None:
            return None
        return requests[0]

    def push_request(self, request: Req, official_queue: List[Req]):
        official_queue.append(request)
        if self.queued_requests_by_client.get(request.adapter_dir, None) is None:
            self.queued_requests_by_client[request.adapter_dir] = [request]
        else:
            self.queued_requests_by_client[request.adapter_dir].append(request)

    def remove_request(self, request: Req, official_queue: List[Req]):
        official_queue.remove(request)
        if self.queued_requests_by_client.get(request.adapter_dir, None) is None:
            raise ValueError(f"Shadow queue for client {request.adapter_dir} does not exist.")
        self.queued_requests_by_client[request.adapter_dir].remove(request)

        # remove that client from our shadow queue to reflect in the keys that there are no requests from this client in the queue.
        if len(self.queued_requests_by_client[request.adapter_dir]) == 0:
            self.queued_requests_by_client.pop(request.adapter_dir)
        
        if official_queue == []:
            self.client_that_made_Q_empty_when_last_request_left = request.adapter_dir
        else:
            self.client_that_made_Q_empty_when_last_request_left = None
    
    def get_clients_in_queue(self) -> List[str]:
        return list(self.queued_requests_by_client.keys())