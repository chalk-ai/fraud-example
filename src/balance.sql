-- Incrementally ingest account balance data from Snowflake.
-- The `balances` table in this example is an append-only table.
-- This comment will be searchable in the Chalk dashboard.
--
-- resolves: account
-- source: snowflake
-- type: offline
-- cron: 5m
-- incremental:
--   mode: row
--   incremental_column: updated_at
--   lookback: 1h
select
    id,
    user_id,
    balance,
    updated_at
from balances;
