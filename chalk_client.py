"""The Chalk Compute integration for the refund-abuse demo.

Resolve the deployed agent by name, call it, get its text back.
"""

from chalkcompute import RemoteFunction
from dotenv import load_dotenv

load_dotenv()  # CHALK_* creds for the from_name lookup below

# resolve deployed agent by name — no URL, no client wiring
investigate_refund = RemoteFunction.from_name("investigate_refund")

# call the agent — runs server-side in Chalk Compute, we get text back
def investigate(user_id: int, reason: str) -> str:
    return "".join(investigate_refund.remote(user_id, reason))   # -> "{trace}\n\n{verdict}"
