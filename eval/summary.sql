WITH run AS (
    SELECT run_id, finished_at, config FROM eval.runs WHERE run_id = %s
), planned AS (
    SELECT run.run_id, planned.case_id, planned.category
    FROM run
    CROSS JOIN LATERAL jsonb_to_recordset(run.config->'cases')
        AS planned(case_id text, category text)
), metrics AS (
    SELECT
        planned.category,
        COUNT(*) AS planned_count,
        COUNT(result.case_id) AS stored_count,
        COUNT(result.case_id) FILTER (
            WHERE result.failure_type IS DISTINCT FROM 'replay_miss'
        ) AS scorable_count,
        COUNT(*) FILTER (WHERE result.failure_type = 'replay_miss') AS replay_miss_count,
        COUNT(result.case_id) FILTER (WHERE result.input_tokens IS NULL) AS missing_tokens_count,
        AVG(result.tools_correct::int) AS tool_accuracy,
        AVG(result.params_valid::int) AS parameter_accuracy,
        AVG(result.success::int) FILTER (
            WHERE result.failure_type IS DISTINCT FROM 'replay_miss'
        ) AS success_rate,
        AVG(result.input_tokens) AS average_input_tokens,
        AVG(result.elapsed_ms) AS average_elapsed_ms
    FROM planned
    LEFT JOIN eval.results AS result USING (run_id, case_id)
    GROUP BY GROUPING SETS ((), (planned.category))
)
SELECT run.run_id, run.finished_at IS NOT NULL AS complete, metrics.*
FROM run CROSS JOIN metrics
ORDER BY metrics.category NULLS FIRST;
