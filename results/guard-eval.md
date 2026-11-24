Simulated agent, 300 seeded episodes. Misbehaviour rates: stuck 0.15, bad args 0.2, over-refund 0.15; tool faults: transient 0.3, slow (300 ms) 0.1.

| guards | task success | unsafe refunds | runaway (hit step/tool cap) | model calls / ep | tool calls / ep | cost / ep (USD) | wall ms / ep | stop reasons |
|---|---|---|---|---|---|---|---|---|
| off | 67.3% | 16.3% | 16.3% | 7.84 | 7.01 | 0.1292 | 134 | completed: 251, max_steps: 49 |
| on | 67.3% | 0.0% | 0.0% | 3.37 | 2.54 | 0.0219 | 12 | completed: 251, loop_detected: 49 |
