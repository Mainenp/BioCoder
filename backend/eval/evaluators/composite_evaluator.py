from __future__ import annotations

from eval.benchmark import AgentPrediction, BenchmarkCase
from eval.evaluators.llm_judge import LLMJudge
from eval.evaluators.rule_evaluator import EvaluationResult, RuleEvaluator
from eval.rubric import rubric_for_task


class CompositeEvaluator:
    def __init__(self, rule: RuleEvaluator | None = None, llm_judge: LLMJudge | None = None) -> None:
        self.rule = rule or RuleEvaluator()
        self.llm_judge = llm_judge

    async def evaluate(self, case: BenchmarkCase, prediction: AgentPrediction) -> EvaluationResult:
        result = self.rule.evaluate(case, prediction)
        if self.llm_judge is None:
            return result
        judge = await self.llm_judge.evaluate(case, prediction)
        judge_score = rubric_for_task(case.task_type, case.rubric).score(judge.scores)
        result.score = round(result.score * 0.75 + judge_score * 0.25, 4)
        result.details["llm_judge"] = judge.model_dump(mode="json")
        if not result.failure_reason and judge.failure_reason:
            result.failure_reason = judge.failure_reason
        if judge.improvement_suggestion:
            result.improvement_suggestion = judge.improvement_suggestion
        return result
