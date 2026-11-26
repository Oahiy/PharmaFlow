SELECT
    first.case_id,
    first.success AS first_success,
    second.success AS second_success,
    CASE
        WHEN first.failure_type IS DISTINCT FROM 'replay_miss'
            AND second.failure_type IS DISTINCT FROM 'replay_miss'
        THEN second.success::int - first.success::int
    END AS success_delta,
    first.failure_type AS first_failure_type,
    second.failure_type AS second_failure_type,
    first.judge_score AS first_judge_score,
    second.judge_score AS second_judge_score,
    second.judge_score - first.judge_score AS judge_delta,
    first.input_tokens AS first_input_tokens,
    second.input_tokens AS second_input_tokens,
    second.input_tokens - first.input_tokens AS input_tokens_delta,
    first.elapsed_ms AS first_elapsed_ms,
    second.elapsed_ms AS second_elapsed_ms,
    second.elapsed_ms - first.elapsed_ms AS elapsed_ms_delta
FROM eval.results AS first
JOIN eval.results AS second USING (case_id)
WHERE first.run_id = %s AND second.run_id = %s
ORDER BY first.case_id;
