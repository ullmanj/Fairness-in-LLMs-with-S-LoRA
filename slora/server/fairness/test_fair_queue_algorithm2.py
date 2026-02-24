import pytest

from slora.server.fairness.fair_queue import FairQueue
from slora.server.io_struct import Batch, Req
from slora.server.sampling_params import SamplingParams


def _mk_req(client: str, req_id: str, input_len: int, max_new_tokens: int = 8) -> Req:
    prompt_ids = list(range(input_len))
    sp = SamplingParams(max_new_tokens=max_new_tokens)
    return Req(client, req_id, prompt_ids, sp)


@pytest.fixture
def fair_queue(monkeypatch):
    q = FairQueue(
        max_total_tokens=10_000,
        batch_max_tokens=10_000,
        running_max_req_size=64,
        adapter_dirs=["a", "b", "c"],
        fair_weights=[],
    )
    # Keep tests focused on Algorithm 2 scheduling logic.
    monkeypatch.setattr(q, "_init_cache_list", lambda current_batch, lora_ranks: None)
    monkeypatch.setattr(q, "_can_add_new_req", lambda req, lora_ranks: True)
    monkeypatch.setattr(q, "_sanity_check_shadow_queue", lambda: None)
    return q


def test_append_lift_uses_active_clients_only(fair_queue):
    fair_queue.counters_by_client.update({"a": 10, "b": 0, "c": -100})
    fair_queue.append(_mk_req("a", "a0", 2))
    fair_queue.append(_mk_req("b", "b0", 2))

    # Lift for b should use min counter among active queued clients (only "a"), not inactive "c".
    assert fair_queue.counters_by_client["b"] == 10


def test_append_does_not_lift_if_client_already_in_queue(fair_queue):
    fair_queue.counters_by_client["a"] = 7
    fair_queue.append(_mk_req("a", "a0", 2))
    fair_queue.append(_mk_req("a", "a1", 2))

    assert fair_queue.counters_by_client["a"] == 7


def test_generate_new_batch_selects_min_counter_client_and_updates_input_counter(fair_queue):
    req_a = _mk_req("a", "a0", 2)
    req_b = _mk_req("b", "b0", 3)
    fair_queue.append(req_a)
    fair_queue.append(req_b)
    # Set counters after append so this test targets line-20 selection behavior,
    # independent of counter-lift side effects in append().
    fair_queue.counters_by_client.update({"a": 5, "b": 1})

    new_batch = fair_queue.generate_new_batch(current_batch=None, lora_ranks={})
    assert new_batch is not None
    assert [r.request_id for r in new_batch.reqs] == ["b0", "a0"]

    assert fair_queue.counters_by_client["b"] == 4  # 1 + wp(=1) * 3
    assert fair_queue.counters_by_client["a"] == 7  # 5 + wp(=1) * 2
    assert fair_queue.waiting_req_list == []


def test_generate_new_batch_breaks_when_head_cannot_fit(fair_queue, monkeypatch):
    req_a = _mk_req("a", "a0", 2)
    req_b = _mk_req("b", "b0", 3)
    fair_queue.append(req_a)
    fair_queue.append(req_b)
    # Set counters after append so b0 is unambiguously selected first.
    fair_queue.counters_by_client.update({"a": 5, "b": 1})

    # The first selected request is b0; make it non-fit to force line-22 break behavior.
    monkeypatch.setattr(fair_queue, "_can_add_new_req", lambda req, lora_ranks: req.request_id != "b0")

    new_batch = fair_queue.generate_new_batch(current_batch=None, lora_ranks={})
    assert new_batch is None
    assert [r.request_id for r in fair_queue.waiting_req_list] == ["a0", "b0"]


def test_generate_new_batch_drops_aborted_then_continues(fair_queue):
    req_a = _mk_req("a", "a0", 2)
    req_a.aborted = True
    req_b = _mk_req("b", "b0", 3)
    fair_queue.counters_by_client.update({"a": 0, "b": 1})
    fair_queue.append(req_a)
    fair_queue.append(req_b)

    new_batch = fair_queue.generate_new_batch(current_batch=None, lora_ranks={})
    assert new_batch is not None
    assert [r.request_id for r in new_batch.reqs] == ["b0"]
    assert fair_queue.waiting_req_list == []


def test_update_counter_adds_output_cost_per_running_request(fair_queue):
    req_a0 = _mk_req("a", "a0", 2)
    req_a1 = _mk_req("a", "a1", 2)
    req_b0 = _mk_req("b", "b0", 2)
    batch = Batch("running", [req_a0, req_a1, req_b0])

    fair_queue.counters_by_client.update({"a": 0, "b": 10, "c": 0})
    fair_queue.update_counter(batch)

    assert fair_queue.counters_by_client["a"] == 4  # w_q(=2) * 2 reqs
    assert fair_queue.counters_by_client["b"] == 12  # w_q(=2) * 1 req
    assert fair_queue.counters_by_client["c"] == 0


def test_rejoin_after_queue_empty_uses_last_leaving_client_counter(fair_queue):
    fair_queue.counters_by_client.update({"a": 5, "b": 0})
    fair_queue.append(_mk_req("a", "a0", 3))

    first_batch = fair_queue.generate_new_batch(current_batch=None, lora_ranks={})
    assert first_batch is not None
    assert [r.request_id for r in first_batch.reqs] == ["a0"]
    assert fair_queue.waiting_req_list == []

    fair_queue.append(_mk_req("b", "b0", 1))
    # a's counter became 8 after admitting a0, so b should be lifted to 8.
    assert fair_queue.counters_by_client["b"] == 8
