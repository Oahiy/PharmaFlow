SELECT
    method,
    COUNT(*) AS query_count,
    AVG(hit_count::numeric / relevant_count) AS recall_at_8,
    AVG(COALESCE(1.0 / first_relevant_rank, 0.0)) AS mrr
FROM eval.retrieval_results
WHERE run_id = %s
GROUP BY method
ORDER BY method;
