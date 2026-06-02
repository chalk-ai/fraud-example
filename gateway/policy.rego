package chalk.policy

import future.keywords.if
import future.keywords.in

# ── Allowlists ────────────────────────────────────────────────────────────────

allowed_features := {
    "order.refund_risk_score",
    "order.customer_prior_refunds",
}

allowed_slack_channels := {"#refund-demo"}

# ── Deny rules ────────────────────────────────────────────────────────────────

# Chalk: block any feature query not in the allowlist.
deny contains msg if {
    input.backend == "chalk"
    not input.tool in allowed_features
    msg := sprintf("'%v' is not an allowed feature", [input.tool])
}

# Slack: block posts to any channel outside the allowlist.
deny contains msg if {
    input.backend == "slack"
    not input.arguments.channel in allowed_slack_channels
    msg := sprintf("'%v' is not an allowed channel", [input.arguments.channel])
}
