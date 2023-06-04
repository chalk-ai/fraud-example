from src.models import User
from chalk.client import ChalkClient


if __name__ == "__main__":
    # Create a new Chalk client. By default, this will
    # pick up the login credentials generated after running
    # `chalk login`.
    client = ChalkClient()

    result = client.query(
        input=User(id=1234),
        output=[
            User.id,
            User.name,
            User.fico_score,
            User.account.balance,
        ],
    )

    assert result.errors is None

    result.get_feature_value(User.fico_score)
    result.get_feature_value(User.account.balance)
