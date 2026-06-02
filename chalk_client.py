"""The entire Chalk Compute integration for the refund-abuse demo.

Resolve the deployed agent by name, call it, get its text back. The agent
(`investigate_refund`) runs its whole agentic loop *inside Chalk Compute* —
discovering and querying features on its own, secrets injected by Chalk, data
never leaving the VPC. From here it's one function call.
"""

from chalkcompute import RemoteFunction
from dotenv import load_dotenv

load_dotenv()  # CHALK_* creds for the from_name lookup below


# resolve deployed agent by name — no URL, no client wiring
investigate_refund = RemoteFunction.from_name("investigate_refund")


def investigate(user_id: int, reason: str) -> str:
    # call the agent — runs server-side in Chalk Compute, we get text back
    return "".join(investigate_refund.remote(user_id, reason))   # -> "{trace}\n\n{verdict}"
