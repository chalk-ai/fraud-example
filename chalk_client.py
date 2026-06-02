"""The Chalk Compute integration for the refund-abuse demo.

Resolve the deployed agent by name, call it, get its text back.
"""

from chalkcompute import RemoteFunction
from dotenv import load_dotenv

load_dotenv()  # reads .env for Chalk credentials for demo

# resolve deployed agent by name — no URL, no client wiring
investigate_refund = RemoteFunction.from_name("investigate_refund")

# call the agent — runs server-side in Chalk Compute, we get text back
def investigate(user_id: int, reason: str) -> str:
    # "".join(iterable) drains a stream of {trace}\n\n{verdict} strings
    return "".join(investigate_refund.remote(user_id, reason))
