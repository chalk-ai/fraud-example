import random


def get_credit_score(name: str, email: str):
    return {
        "fico_score": random.randint(300, 850),
        "credit_score_tags": random.sample(["match", "no_match", "default"]),
    }
