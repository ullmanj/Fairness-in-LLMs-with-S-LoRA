# S-LoRA: Enhanced Cost Functions for Fair LLM Serving

This codebase is a fork off of S-LoRA replicating the VTC scheduler algorithm and introducing a new quadratic cost function and corresponding service definition.

See full paper [here](https://www.scs.stanford.edu/26wi-cs244c/proj/fair_llm.pdf).

## Overview

LLM providers receive requests of vastly different sizes and frequencies from a diverse array of clients. To ensure that no client is starved in responding to those requests, Sheng et al. (OSDI '24) introduced the Virtual Token Counter (VTC) scheduler. This algorithm for the LLM scheduler ensures fairness without artificially causing under-utilization.

"At What Cost?" presents a comprehensive replication and extension of the VTC scheduler to incorporate the needs of the present day. While the original VTC uses a linear cost function, we explore the use of a **quadratic cost function** to provide a more accurate measure of fairness, accounting for the inherent non-linearity of the attention mechanism.

## Key Contributions

* **VTC Replication:** Independently verified the fairness guarantees of the VTC algorithm on modern NVIDIA H100 GPUs.
* **Quadratic Service Definition:** Proposed a new definition of "service" more closely tuned to actual compute costs as context windows grow.
* **Modern Benchmarking:** Redefining the cost maintains a reduced service difference while being more resistant to asymmetric workloads.

---

## Background: The Cost of Attention

In the prefill stage, the model processes the entire input sequence in parallel to compute the initial Key-Value (KV) cache. This operation has $O(L_{in}^2)$ computational complexity. During the decode stage, although the use of a KV cache prevents re-computing previous states, each new token still must attend to every preceding token in the sequence. Consequently, generating the $k$-th output token requires $O(L_{in} + k)$ operations, making the entire cost **quadratic** relative to the total context length.

### VTC Linear vs. VTC Quadratic

The original VTC paper uses a linear heuristic:
$$Cost = w_{in} \cdot L_{in} + w_{out} \cdot L_{out}$$

We argue that this fails to account for the accelerating cost of attention as context windows reach the scales used in modern applications (e.g., $10^5$ to $10^7$ tokens).

We define **VTC Quadratic** by adding two additional components that scale with the square of the input and output length. The overall service $W$ received by a client $c$ for a query $q$ is:
$$W_c(q) = w_i l_i + w_o l_o + \frac{w_i}{d} l_i^2 + \frac{w_0}{2d} (l_i)(l_i + 1)$$
*(Note: We chose $d=1000$ so that a medium-length query receives roughly twice the service per token as a minimum-size query, matching tiered API pricing).*

---

## Replication Results

Using NVIDIA H100 GPUs, we scaled up request rates to ensure a growing backlog, allowing us to observe VTC's prioritization logic.

### Service Rate Stability
Our independent re-implementation matched the behavior of the original paper. Due to the H100 hardware, the service received by each client was around 2 times higher and the calculated service over time was more stable.

| (a) Accumulated Service Difference | (b) Received Service Rate (60s window) |
| :--- | :--- |
| <img width="302" height="259" alt="Screenshot 2026-03-16 at 2 33 45 PM" src="https://github.com/user-attachments/assets/cacaafba-a169-433e-b3d6-448e8f43158d" /> | <img width="299" height="244" alt="Screenshot 2026-03-16 at 2 34 05 PM" src="https://github.com/user-attachments/assets/c037f47f-8548-4178-a6a8-80c4347f5c03" /> |
| **Figure 1:** VTC maintains near-zero service difference compared to FCFS. | **Figure 1:** Received service rate for Client 1 and Client 2 using VTC. |

---

## Evaluation of Quadratic Scheduling

In "Experiment 2," we tested an asymmetric workload where Client 1 sends small queries (256 tokens) and Client 2 sends large queries (2048 tokens).

* **VTC Linear Issue:** Assigns constant service per token, underestimating Client 2's actual cost. This causes Client 2 to take up a majority of GPU utilization even if received service looks identical.
* **VTC Quadratic Solution:** Ensures that the weighted sum of tokens received by a client becomes more resistant to the workloads of other clients when all are backlogged.


<img width="337" height="284" alt="Screenshot 2026-03-16 at 2 34 35 PM" src="https://github.com/user-attachments/assets/0933948f-d429-4e82-aee7-4d93702274f0" />

> **Figure 4:** Running the VTC Quadratic scheduler results in Client 1 receiving service (as defined by VTC Linear) much closer to its performance in symmetric workloads, demonstrating improved resistance.

---

## Implementation Details

This implementation exists on top of the **S-LORA** scheduler. To view our code, see [github.com/ullmanj/Fairness-in-LLMs-with-S-LORA](https://github.com/ullmanj/Fairness-in-LLMs-with-S-LORA).

### Infrastructure Optimizations
* **Connection Pool Starvation:** Removed the default limit of 100 concurrent connections in `aiohttp` (limit=0) so arrival rates accurately reflect the workload.
* **In-Flight Request Throttling:** Introduced a per-client `asyncio.Semaphore` to maintain stable pressure on the server queue without crashing the service.
