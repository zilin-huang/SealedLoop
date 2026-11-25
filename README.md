# SealedLoop

**Keep or Roll Back? Out-of-Loop Judges for Self-Improving VLA and End-to-End Embodied Policies**

Zilin Huang, Zhengyang Wan, Kangan Qian, Zihao Sheng, Sikai Chen

[Project page](https://zilin-huang.github.io/SealedLoop/) · [Paper](https://zilin-huang.github.io/SealedLoop/assets/SealedLoop.pdf) · [Videos](https://zilin-huang.github.io/SealedLoop/#video)

![SealedLoop framework](https://zilin-huang.github.io/SealedLoop/assets/figures/framework.png)

Self-improving embodied policies are usually kept or rolled back by a judge that is trained or chosen inside the same loop. SealedLoop adds a sealed judge: an evaluator outside the loop that the improver cannot see or optimize against. A new policy is kept only when a sequential test on the sealed judge confirms the gain, with a bounded probability of keeping a worse one. We evaluate on LIBERO manipulation, NAVSIM driving, and a full-scale real vehicle.

## Code

Code coming soon.

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
