from scipy.stats import norm


def calculate_ev(projection, line, sd, american_odds, side="Over"):
    if sd <= 0 or american_odds == 0:
        raise ValueError("Invalid standard deviation or odds")

    probability_over = norm.sf(line, loc=projection, scale=sd)

    if side == "Over":
        probability = probability_over
    elif side == "Under":
        probability = 1 - probability_over
    else:
        raise ValueError("Side must be Over or Under")

    if american_odds > 0:
        profit_per_dollar = american_odds / 100
    else:
        profit_per_dollar = 100 / abs(american_odds)

    ev = (
        probability * profit_per_dollar
        - (1 - probability)
    ) * 100

    return {
        "Probability": round(probability * 100, 2),
        "EV": round(ev, 2)
    }
