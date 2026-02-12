from slora.server.router.req_queue import ReqQueue
from slora.server.io_struct import Batch

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
        
        return

    #  ############# Monitoring stream (i.e. new request comes in? Append it. ) #############

    # This will have lines 7-14 of Alg2. See `loop_for_netio_req` from
    # manager.py for the prior steps in this loop.
    def append(self, req):
        # if another request does NOT exist in the queue from the same client
            # if the queue is empty
                # get the counter of the last client who left the Q (call it c_l)
                # set the counter for this request's client (call it c_u) to the max{ c_u, c_l }
            # else
                # Build a set "P" of all the clients who have a request in the queue
                # Get the minimum counter for those clients in P and call it c_min
                # Update the counter for this request's client (call it c_u) to the max{ c_u, c_min }

        # enqueue as normal
        super().append(req)
        return

    # ############# With Execution stream (i.e. whose turn is it next? Build a batch.) #############
    
    # This will have lines 18-26 of Alg2. See `_step` from manager.py for the
    # surrounding steps in this loop.
    def generate_new_batch(self, current_batch:Batch, lora_ranks: dict[str, int]):
        # Create an empty batch
        # while true
            # get the client (call it k) whose counter is the smallest among alll of the clients in the queue.
            # let r be the earlist request in the queue from client k.
            # if r cannot fit into the batch, break. (because we have filled the batch maximally while staying fair)
            # update the counter for client k to be it's current value + w_p_input times the input length of r.
            # Append r to the batch
            # Remove r from the queue
       
        return None  # return the batch


    # Line 30 of Alg2.
    def update_counter(self, batch:Batch):
        # For each of the clients (call it k) in the the batch
            # Find the set of requests in the batch that are from client k (call it "requests")
            # Get the length of "requests" (call it L)
            # Add to the counter for this client the product of w_q_output and L
        return





