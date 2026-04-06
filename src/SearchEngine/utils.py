"""
Shared utilities.
"""
import time


def make_marker(prefix: str, q: str):
    """Returns a mark(stage, detail) closure that prints per-stage timing."""
    print(f"\n[{prefix}] {'═'*6} q={q!r} {'═'*6}", flush=True)
    t0  = time.time()
    t_  = [t0]

    def mark(stage: str, detail: str = ""):
        now   = time.time()
        took  = now - t_[0]
        total = now - t0
        line  = f"[{prefix}] {stage:<22} | took={took:.3f}s | total={total:.3f}s"
        if detail:
            line += f" | {detail}"
        print(line, flush=True)
        t_[0] = now

    return mark
