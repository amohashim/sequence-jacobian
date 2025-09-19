# utilities/convergence.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass(frozen=True)
class PolicyDiagnostics:
    last_error: float
    best_error: float
    monotonicity_M: float
    stall_count: int
    converged: bool  # whether this policy ever satisfied the within-tol check

class PolicyConvergenceWarning(UserWarning):
    """
    Warning carrying PER-POLICY convergence diagnostics.

    Attributes
    ----------
    phase : 'backward' | 'forward'
    maxit : int
    tol   : float
    failed_policies : tuple[str, ...]           # subset of metrics.keys()
    metrics : dict[str, PolicyDiagnostics]      # per-policy stats (all tracked)
    """
    def __init__(
        self,
        *,
        phase: str,
        maxit: int,
        tol: float,
        failed_policies: Tuple[str, ...],
        metrics: Dict[str, PolicyDiagnostics]
        ):
        
        self.phase = phase
        self.maxit = int(maxit)
        self.tol = float(tol)
        self.failed_policies = tuple(failed_policies)
        self.metrics = dict(metrics)

        # Human-friendly message (keeps your logs readable)
        parts = [f"[{self.phase}] No convergence after {self.maxit} iterations."]
        if self.failed_policies:
            details = []
            for k in self.failed_policies:
                md = self.metrics[k]
                details.append(
                    f"{k}: last={md.last_error:.3e}, best={md.best_error:.3e}, "
                    f"M={md.monotonicity_M:.2f}, stall={md.stall_count}"
                )
            parts.append("Failed policies: " + "; ".join(details))
        else:
            parts.append("All tracked policies individually met tol at least once "
                         "(but not simultaneously).")
        super().__init__(" ".join(parts))
