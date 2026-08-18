# Figure captions

**Figure 0.** Project overview of the shared SmoothLLM execution. The input prompt is perturbed into multiple copies, each copy is sent through the target LLM and the internal Judge, and the resulting vote stream is consumed either by fixed SmoothLLM (all N=10 copies) or by SE-SmoothLLM (stop once the final verdict is mathematically locked). The verdict card uses all 600 fixed/SE pairs (300 harmful and 300 benign), while the compute cards use the 300 harmful pairs and summarize exact agreement and savings.

**Figure 1.** Harmful-request attack success rate (ASR) for SmoothLLM fixed and SE-SmoothLLM across seeds 42, 43, and 44. Points show the exact proportion of 100 harmful samples per seed; dotted horizontal lines show the three-seed mean, without inferential significance testing. The DeepSeek-V4-Flash auxiliary Judge gives 6.33% for fixed SmoothLLM and 6.00% for SE-SmoothLLM, so the observed ASR difference is small in this run.

**Figure 2.** Model-query cost per sample for undefended generation, fixed SmoothLLM, and SE-SmoothLLM on harmful and benign splits. Bars show mean queries and error bars show sample standard deviation (n=300 per method and split). SE-SmoothLLM uses fewer queries than the fixed budget method on both splits while preserving the fixed method as the efficiency reference.

**Figure 3.** Prompt and completion token cost per sample across methods and splits. Bars show mean tokens and error bars show sample standard deviation (n=300 per method and split). The token panels expose the compute cost associated with early stopping instead of presenting query savings alone.

**Figure 4.** Benign-request refusal rate across seeds for the three methods. Points show the exact rate among 100 benign samples per seed; no significance test is applied. The Llama-3-8B Refusal Judge reports 16.33% for fixed SmoothLLM and 17.33% for SE-SmoothLLM, which should be interpreted as a usability cost rather than a security gain.

The paired fixed-versus-SE comparison contains 600 sample pairs and 0 internal verdict mismatches.
