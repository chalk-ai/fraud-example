from chalk.features import Features, online
from src.models import User
from src.mocks import experian


@online
def account_name_match(
    title: User.account.title,
    name: User.name,
) -> User.account_name_match:
    """Docstrings show up in the Chalk dashboard"""
    intersection = set(title) & set(name)
    union = set(title) | set(name)
    return len(intersection) / len(union)


@online
def get_fraud_score(
    name: User.name,
    email: User.email,
) -> Features[User.fico_score, User.credit_score_tags]:
    response = experian.get_credit_score(name, email)

    # We don't need to provide all the features for
    # the `User` class, only the ones that we want to update.
    return User(
        fico_score=response["fico_score"],
        credit_score_tags=response["tags"],
    )
