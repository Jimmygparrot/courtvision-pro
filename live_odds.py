import requests
import tomli
from pathlib import Path


BASE_URL = "https://api.the-odds-api.com/v4"


def get_api_key():
    import streamlit as st
    return st.secrets["ODDS_API_KEY"]


def get_nba_events():
    response = requests.get(
        f"{BASE_URL}/sports/basketball_nba/events/",
        params={"apiKey": get_api_key()},
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def get_player_props(event_id, market="player_points"):
    response = requests.get(
        f"{BASE_URL}/sports/basketball_nba/events/{event_id}/odds",
        params={
            "apiKey": get_api_key(),
            "regions": "us",
            "markets": market,
            "oddsFormat": "american"
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def implied_probability(odds):
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def compare_player(player, data):
    results = []

    for book in data.get("bookmakers", []):
        for market in book.get("markets", []):
            outcomes = market.get("outcomes", [])

            for over in outcomes:
                if over.get("description") != player:
                    continue

                if over.get("name") != "Over":
                    continue

                under = next(
                    (
                        item for item in outcomes
                        if item.get("description") == player
                        and item.get("name") == "Under"
                        and item.get("point") == over.get("point")
                    ),
                    None
                )

                if under is None:
                    continue

                p_over = implied_probability(over["price"])
                p_under = implied_probability(under["price"])

                fair_probability = (
                    p_over / (p_over + p_under) * 100
                )

                results.append({
                    "Player": player,
                    "Sportsbook": book["title"],
                    "Statistic": market["key"],
                    "Line": over["point"],
                    "Over_Odds": over["price"],
                    "Under_Odds": under["price"],
                    "No_Vig_Over": round(fair_probability, 2)
                })

    return results


def best_prices(results):
    comparisons = {}

    for item in results:
        key = item["Line"]

        if key not in comparisons:
            comparisons[key] = {
                "Over": item,
                "Under": item
            }
        else:
            if item["Over_Odds"] > comparisons[key]["Over"]["Over_Odds"]:
                comparisons[key]["Over"] = item

            if item["Under_Odds"] > comparisons[key]["Under"]["Under_Odds"]:
                comparisons[key]["Under"] = item

    return comparisons


def find_team_event(team_name, events):
    """
    Find the earliest upcoming sportsbook event
    involving the selected NBA team.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    matches = []

    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")

        if team_name not in (home, away):
            continue

        try:
            start = datetime.fromisoformat(
                event["commence_time"].replace("Z", "+00:00")
            )
        except (KeyError, ValueError, TypeError):
            continue

        if start <= now:
            continue

        matches.append((start, event))

    if not matches:
        return None

    matches.sort(key=lambda item: item[0])

    return matches[0][1]
