"""Refund-abuse demo features.

Defines the Chalk context for an order under refund review. One features class,
one resolver — the simplest shape for the Summit demo's "one-line context
engine call" beat. Deployed via `chalk apply`.
"""

from chalk import online
from chalk.features import Features, Primary, features


@features
class Order:
    id: Primary[str]
    refund_risk_score: float
    customer_prior_refunds: int


_DEMO_ORDERS: dict[str, tuple[float, int]] = {
    "ORD-8823": (0.87, 4),
    "ORD-1001": (0.12, 0),
    "ORD-4242": (0.55, 2),
}


@online
def get_order_refund_context(
    order_id: Order.id,
) -> Features[Order.refund_risk_score, Order.customer_prior_refunds]:
    risk, claims = _DEMO_ORDERS.get(order_id, (0.05, 0))
    return Order(refund_risk_score=risk, customer_prior_refunds=claims)
