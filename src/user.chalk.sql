-- The features given to us by the user.
-- This comment will be searchable in the Chalk dashboard.
--
-- resolves: user
-- source: postgres
select
    id,
    full_name as name,
    email
from users;
