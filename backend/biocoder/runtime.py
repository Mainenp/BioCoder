from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from biocoder.state import AgentState


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    allowed: bool
    reason: str | None = None


class BudgetGuard:
    @staticmethod
    def check(state: AgentState) -> BudgetDecision:
        budget = state.budget
        elapsed = (datetime.now(UTC) - state.start_time).total_seconds()
        checks = (
            (state.current_step > budget.max_steps, "max_steps"),
            (state.retries > budget.max_retries, "max_retries"),
            (elapsed > budget.timeout_seconds, "timeout"),
            (state.token_usage.total_tokens > budget.token_budget, "token_budget"),
            (state.estimated_cost > budget.cost_budget > 0, "cost_budget"),
        )
        for exceeded, reason in checks:
            if exceeded:
                return BudgetDecision(False, reason)
        return BudgetDecision(True)
