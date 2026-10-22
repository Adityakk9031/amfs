-- Migration 014: the covariates of the task on the outcome row
--
-- An outcome records what was done (`actions_taken`) in what kind of task
-- (`situation`) under what runtime (`environment`). It did not record what
-- the task *was like*: the customer's plan, the app version the lookup
-- showed, the card's issuer country. When a rule changes for part of a
-- situation — the export fix stops working on paid plans and keeps working
-- on the free one — the situation's record turns contested (wins and losses
-- interleaved, no streak) and nothing in it says which runs the old action
-- still wins. On the 2026-10-21 regime study the recommendation kept the
-- stale action as `act` on 59 % of the first searches after such a change.
--
-- The covariates are the scalar attributes a run declared about the task
-- (`set_session_attributes`, `Run.begin(covariates=...)`) less its
-- environment keys and the situation label, copied here at commit from the
-- record so `amfs_core.actions.split_contested` can read them off the same
-- rows the priors are aggregated from. Scalars only; at most the attribute
-- bag's 20 keys. Written only when the record has any.

ALTER TABLE amfs_outcomes ADD COLUMN IF NOT EXISTS covariates JSONB;
