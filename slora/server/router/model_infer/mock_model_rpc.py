import random
import time
from collections import defaultdict


class MockModelRpcServer:

    def exposed_init_model(self, rank_id, world_size, weight_dir, adapter_dirs,
                           max_total_token_num, load_way, mode, input_params,
                           prefetch_stream):
        self.tp_rank = rank_id
        self.world_size = world_size
        self.input_params = input_params
        self.vocab_size = 32000
        self.cache = {}

    def exposed_load_adapters(self, adapter_dirs, prefetch=False):
        pass

    def exposed_offload_adapters(self, reserve_dirs=None, prefetch=False):
        pass

    def exposed_add_batch(self, batch_id, reqs, dtype):
        self.cache[batch_id] = {
            "reqs": list(reqs),
            "input_lengths": [len(r["input_id"]) for r in reqs],
        }

    def _forward(self, batch_id):
        time.sleep(0.01)  # small delay so tokens don't all arrive at once
        batch = self.cache[batch_id]
        output_dict = {}
        for i, req in enumerate(batch["reqs"]):
            token_id = random.randint(3, self.vocab_size - 1)  # skip BOS/EOS
            metadata = {"id": token_id, "logprob": -0.5}
            output_dict[req["request_id"]] = (token_id, metadata)
            batch["input_lengths"][i] += 1
            req["input_id"].append(token_id)
        return output_dict

    def exposed_prefill_batch(self, batch_id):
        return self._forward(batch_id)

    def exposed_decode_batch(self, batch_id):
        return self._forward(batch_id)

    def exposed_filter_batch(self, batch_id, req_id_list):
        batch = self.cache.pop(batch_id)
        req_id_set = set(req_id_list)
        batch["reqs"] = [r for r in batch["reqs"] if r["request_id"] in req_id_set]
        batch["input_lengths"] = [len(r["input_id"]) for r in batch["reqs"]]
        self.cache[batch_id] = batch

    def exposed_merge_batch(self, batch_id1, batch_id2):
        b1 = self.cache.pop(batch_id1)
        b2 = self.cache.pop(batch_id2)
        b1["reqs"].extend(b2["reqs"])
        b1["input_lengths"].extend(b2["input_lengths"])
        self.cache[batch_id1] = b1

    def exposed_remove_batch(self, batch_id):
        self.cache.pop(batch_id, None)

    def exposed_merge_adapter(self):
        pass

    def exposed_unmerge_adapter(self):
        pass

    def exposed_profile_prefill(self):
        return defaultdict(dict), defaultdict(dict)
