def percent_to_american(percent):
    """Convert a percentage price to American odds."""
    if not 0 < percent < 100:
        raise ValueError("Percentage must be between 0 and 100.")

    probability = percent / 100

    if probability < 0.5:
        return round(100 * (1 - probability) / probability)

    return round(-100 * probability / (1 - probability))
