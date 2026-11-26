CREATE SCHEMA IF NOT EXISTS eval;

CREATE TABLE IF NOT EXISTS eval.runs (
    run_id uuid PRIMARY KEY,
    tag text NOT NULL,
    git_commit text NOT NULL,
    user_id uuid UNIQUE NOT NULL,
    config jsonb NOT NULL,
    started_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    finished_at timestamptz,
    CONSTRAINT runs_time_order CHECK (finished_at >= started_at)
);

CREATE TABLE IF NOT EXISTS eval.cases (
    case_id text PRIMARY KEY,
    category text NOT NULL,
    CONSTRAINT cases_category CHECK (category IN ('single_tool', 'multi_hop', 'exception', 'self_healing'))
);

CREATE TABLE IF NOT EXISTS eval.results (
    run_id uuid NOT NULL REFERENCES eval.runs (run_id),
    case_id text NOT NULL REFERENCES eval.cases (case_id),
    thread_id uuid UNIQUE NOT NULL,
    success boolean NOT NULL,
    tools_correct boolean NOT NULL,
    params_valid boolean,
    judge_score smallint CHECK (judge_score BETWEEN 0 AND 3),
    failure_type text,
    error_detail jsonb,
    input_tokens bigint CHECK (input_tokens >= 0),
    elapsed_ms bigint NOT NULL CHECK (elapsed_ms >= 0),
    repair_attempts smallint CONSTRAINT results_repair_attempts_check CHECK (repair_attempts >= 0),
    trace_id text NOT NULL CHECK (trace_id ~ '^[0-9a-f]{32}$'),
    PRIMARY KEY (run_id, case_id),
    CONSTRAINT results_success_matches_failure CHECK (success = (failure_type IS NULL)),
    CONSTRAINT results_failure_type CHECK (failure_type IN (
        'replay_miss', 'invalid_args', 'agent_error', 'step_limit', 'repair_budget_exceeded',
        'tool_error', 'wrong_tool', 'rule_failed', 'judge_call_error',
        'judge_parse_error', 'judge_below_threshold'
    ))
);
