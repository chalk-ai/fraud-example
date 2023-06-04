from chalk.features import features, has_one, FeatureTime, feature


@features
class User:
    id: int

    # The name the user provided to us at signup.
    # :owner: identity@chalk.ai
    # :tags: pii
    name: str

    # :tags: pii
    email: str

    # The account that this user owns.
    account: "Account"

    # The similarity between the user's name and the account's title.
    account_name_match: float

    # The fraud score, as provided by a third-party vendor.
    fico_score: int = feature(min=300, max=850, strict=True)

    # Tags from our credit scoring vendor.
    credit_score_tags: list[str]


@features(max_staleness="infinity", etl_offline_to_online=True)
class Account:
    id: int

    # The name of the owner of the account.
    title: str

    # The id of the user that owns this account.
    user_id: int

    # The balance of the account, in dollars.
    balance: float

    # The user that owns this account.
    user: User = has_one(lambda: User.id == Account.user_id)

    updated_at: FeatureTime
