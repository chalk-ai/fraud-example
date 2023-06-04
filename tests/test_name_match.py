from src.resolvers import account_name_match


def test_names_match():
    """Resolvers can be unit tested exactly as you would expect.

    Here, the `account_name_match` resolver should return 1.0
    because the `title` and `name` are identical.
    """
    assert 1 == account_name_match(
        title="John Coltrane",
        name="John Coltrane",
    )


def test_names_completely_different():
    """The `account_name_match` resolver should return 0
    because the `title` and `name` don't share any characters.
    """
    assert 0 == account_name_match(
        title="John Coltrane",
        name="Zyx",
    )
