BEGIN;

SELECT pg_advisory_xact_lock(hashtext('pharmaflow.eval.schema'));

ALTER TABLE eval.cases DROP CONSTRAINT cases_category;
ALTER TABLE eval.cases ADD CONSTRAINT cases_category
    CHECK (category IN ('single_tool', 'multi_hop', 'exception', 'self_healing'));

ALTER TABLE eval.results ADD COLUMN IF NOT EXISTS
    repair_attempts smallint CHECK (repair_attempts BETWEEN 0 AND 3);

COMMIT;
