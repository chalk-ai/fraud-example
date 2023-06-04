-- The transactions for all accounts.
-- This comment will be searchable in the Chalk dashboard.
--
-- resolves: transaction
-- source: postgres
select
    id,
    account_id,
    amount,
    status,
    date
from txns;
