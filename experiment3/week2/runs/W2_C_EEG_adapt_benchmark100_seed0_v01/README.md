# W2 C benchmark: passed

Real EEG only; 100 diagnostic updates, batch=2, FP32. Original EMA reloaded in this process; condition14, target=v_prediction.

Only 54 training subjects are eligible; no sleep labels, validation or test signals were read. Gate/backbone gradient evidence, frozen checksums, exact parameter names and signal accesses are recorded.

The 100-update benchmark includes its first 10 warmup updates. Its weights are diagnostic only, not a selected S1 result. Formal adaptation must reload the original EMA and explicitly lock its total steps, stopping and new EMA policy; no downstream or N1 improvement was evaluated.
