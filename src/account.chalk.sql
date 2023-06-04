-- Incrementally ingest account data from Snowflake.
-- This comment will be searchable in the Chalk dashboard.
--
-- resolves: account
-- source: snowflake
-- type: offline
-- cron: 5m
-- incremental:
--   mode: row
--   lookback: 1h
select
    id,
    user_id,
    amount,
    updated_at
from accounts;
