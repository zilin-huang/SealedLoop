# SealedLoop

**Keep or Roll Back? Out-of-Loop Judges for Self-Improving VLA and End-to-End Embodied Policies**

Zilin Huang, Zhengyang Wan, Kangan Qian, Zihao Sheng, Sikai Chen

[Project page](https://zilin-huang.github.io/SealedLoop/) · [Paper](https://zilin-huang.github.io/SealedLoop/assets/SealedLoop.pdf) · [Videos](https://zilin-huang.github.io/SealedLoop/#video)

![SealedLoop framework](https://zilin-huang.github.io/SealedLoop/assets/figures/framework.png)

Self-improving embodied policies are usually kept or rolled back by a judge that is trained or chosen inside the same loop. SealedLoop adds a sealed judge: an evaluator outside the loop that the improver cannot see or optimize against. A new policy is kept only when a sequential test on the sealed judge confirms the gain, with a bounded probability of keeping a worse one. We evaluate on LIBERO manipulation, NAVSIM driving, and a full-scale real vehicle.

## Code

This first release contains the sequential test that decides whether a candidate policy is kept or rolled back. The evaluation records of the paper and the scripts that produce its figures and tables from them, and the run scripts of the manipulation and driving loops, are coming soon.

### Install

```bash
git clone https://github.com/zilin-huang/SealedLoop.git
cd SealedLoop
pip install -e ".[test]"
pytest
```

The package depends only on NumPy.

### Keep or roll back

Score the current policy and the candidate on the same evaluation items with a judge that the improvement loop never sees, then read the test:

```python
from sealedloop import GateConfig, gate_path

cfg = GateConfig(alpha=0.05, budget=100, min_n=10, score_range=(0.0, 1.0))
reading = gate_path(candidate_scores, current_scores, cfg)

reading.keep        # True: the candidate replaces the current policy
reading.decision    # accept, reject, unresolved_at_budget or cs_empty
reading.stopped_at  # number of items used when the test decided
reading.lo, reading.hi  # confidence sequence for the mean paired gain
```

The test is a betting confidence sequence on the paired score differences (Waudby-Smith and Ramdas, 2024). It can be read after every item, and the probability of keeping a candidate that is not better than the current policy is at most `alpha`. A candidate is kept only on `accept`; every other decision rolls back. `python examples/keep_or_roll_back.py` runs the test on three synthetic candidates.

| File | Content |
|---|---|
| `sealedloop/confseq.py` | Betting confidence sequence for the mean of bounded observations |
| `sealedloop/gate.py` | The sequential test and its decisions |
| `sealedloop/power.py` | Power and error rate of the test by simulation |

## Citation

```bibtex
@article{huang2026sealedloop,
  title = {Keep or Roll Back? Out-of-Loop Judges for Self-Improving VLA and End-to-End Embodied Policies},
  author = {Huang, Zilin and Wan, Zhengyang and Qian, Kangan and Sheng, Zihao and Chen, Sikai},
  year = {2026},
  journal = {arXiv preprint},
  url = {https://zilin-huang.github.io/SealedLoop/}
}
```

## Contact

Sikai Chen (corresponding author): sikai.chen@wisc.edu
