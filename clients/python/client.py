# AUTO-GENERATED FILE. Do not edit.
# Run [chalk codegen](https://docs.chalk.ai/cli/codegen) to generate.
# chalk codegen python --out ./clients/python/client.py
# fmt: off
# isort: skip_file
from datetime import date, datetime
from typing import Dict, List, Optional, Union, Any
from typing_extensions import TypeAlias
from chalk.features import DataFrame, FeatureTime, Primary, feature, features, has_many, has_one
from chalk.streams import Windowed, windowed
from dataclasses import dataclass


@features
class Account:
    id: int = feature(max_staleness="infinity", primary=True)

    # The name of the owner of the account.
    title: str = feature(max_staleness="infinity")

    # The id of the user that owns this account.
    user_id: int = feature(max_staleness="infinity")

    # The balance of the account, in dollars.
    balance: float = feature(max_staleness="infinity")

    # The user that owns this account.
    user: "User" = has_one(lambda: User.id == Account.user_id)
    updated_at: FeatureTime = feature(max_staleness="infinity")


@features
class User:
    id: int = feature(primary=True)

    # The name the user provided to us at signup.
    # :tags: pii
    # :owner: identity@chalk.ai
    name: str

    # :tags: pii
    email: str

    # The account that this user owns.
    account: "Account" = has_one(lambda: User.id == Account.user_id)

    # The similarity between the user's name and the account's title.
    account_name_match: float

    # The fraud score, as provided by a third-party vendor.
    fico_score: int

    # Tags from our credit scoring vendor.
    credit_score_tags: List[str]
