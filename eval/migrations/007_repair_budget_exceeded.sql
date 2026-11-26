BEGIN;

SELECT pg_advisory_xact_lock(hashtext('pharmaflow.eval.schema'));

ALTER TABLE eval.results
    DROP CONSTRAINT results_failure_type,
    DROP CONSTRAINT results_repair_attempts_check;

ALTER TABLE eval.results
    ADD CONSTRAINT results_failure_type CHECK (failure_type IN (
        'replay_miss', 'invalid_args', 'agent_error', 'step_limit', 'repair_budget_exceeded',
        'tool_error', 'wrong_tool', 'rule_failed', 'judge_call_error',
        'judge_parse_error', 'judge_below_threshold'
    )),
    ADD CONSTRAINT results_repair_attempts_check CHECK (repair_attempts >= 0);

COMMIT;
