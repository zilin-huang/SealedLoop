"""Recompute every keep-or-roll-back decision of the paper from the per-item paired scores.

For each generation the test is rerun under both judges and compared with the recorded decision; the candidate is
kept exactly when the test of the deciding judge accepts.
"""
import collections

import records as rec


def main():
    a = rec.parser(__doc__).parse_args()
    rows = rec.generations(a.records)
    bad, kept = 0, collections.Counter()
    for r in rows:
        for judge in ("sealed", "inloop"):
            d = rec.decide(a.records, r, judge)
            ok = d.decision == r[judge + "_decision"] and d.n == r[judge + "_pairs_read"]
            if judge == r["deciding_judge"]:
                ok = ok and d.keep == bool(r["kept"])
            bad += not ok
            if not ok:
                print("MISMATCH", r["testbed"], r["run"], r["generation"], judge, d.decision, d.n)
        kept[(r["testbed"], r["variant"], "control" if r["control"] else "main")] += r["kept"]
    for k, v in sorted(kept.items()):
        n = sum(1 for r in rows if (r["testbed"], r["variant"], "control" if r["control"] else "main") == k)
        print("%-13s %-20s %-8s kept %2d of %3d" % (*k, v, n))
    print("%d generations, %d decisions recomputed, %d mismatches" % (len(rows), 2 * len(rows), bad))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
