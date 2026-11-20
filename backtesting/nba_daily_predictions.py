from datetime import datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent.parent
SEASON = "2026-27"
LIMITATIONS = (
    "Pregame team records exclude games played on the prediction date. "
    "The logistic model uses team win-rate difference only. It does not "
    "account for injuries, lineups, travel, rest, or sportsbook prices. "
    "Cross-season held-out accuracy was 67.64% on 2025-26; future results "
    "may differ. Probabilities are estimates, not guarantees."
)


def build_prediction_pdf(predictions, game_date, output_path=None, limitations=None):
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("COURTVISION PRO", styles["Title"]),
        Paragraph(f"NBA Game Predictions - {game_date}", styles["Heading2"]),
        Spacer(1, 18),
    ]

    data = [[
        Paragraph("Away", styles["BodyText"]),
        Paragraph("Home", styles["BodyText"]),
        Paragraph("Predicted winner", styles["BodyText"]),
        Paragraph("Win probability", styles["BodyText"]),
    ]]

    for row in predictions.to_dict("records"):
        away = row.get("away_team", row.get("away", ""))
        home = row.get("home_team", row.get("home", ""))
        data.append([
            Paragraph(str(away), styles["BodyText"]),
            Paragraph(str(home), styles["BodyText"]),
            Paragraph(str(row["predicted_winner"]), styles["BodyText"]),
            f'{float(row["win_probability"]):.1%}',
        ])

    table = Table(data, colWidths=[105, 105, 175, 95], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.whitesmoke, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.extend([
        table,
        Spacer(1, 15),
        Paragraph(limitations or LIMITATIONS, styles["BodyText"]),
    ])

    output = BytesIO()
    SimpleDocTemplate(
        output,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    ).build(elements)
    report = output.getvalue()
    if output_path is not None:
        Path(output_path).write_bytes(report)
    return report


def main():
    from nba_predictions import (
        build_pregame_predictions,
        load_current_season_team_games,
        load_previous_season_team_games,
        load_todays_matchups,
        save_prediction_snapshots,
    )

    today = datetime.now(ZoneInfo("America/New_York")).date()
    games, schedule_source, schedule_warning = load_todays_matchups(today)
    if games.empty:
        print(schedule_warning or f"No NBA games found for {today}.")
        return

    try:
        current_results = load_current_season_team_games()
    except Exception as error:
        print(
            "Current-season results unavailable; using the prior-season "
            f"baseline: {type(error).__name__}"
        )
        current_results = pd.DataFrame()

    predictions = build_pregame_predictions(
        games,
        current_results,
        load_previous_season_team_games(),
    )
    if predictions.empty:
        print(f"No unstarted NBA games available for prediction on {today}.")
        return

    save_prediction_snapshots(predictions)
    output_dir = ROOT / "reports"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / f"NBA_Predictions_{today}.csv"
    pdf_path = output_dir / f"CourtVision_Pro_{today}.pdf"
    predictions.to_csv(csv_path, index=False)
    build_prediction_pdf(
        predictions,
        today,
        output_path=pdf_path,
        limitations=(
            f"Schedule source: {schedule_source}. {LIMITATIONS}"
        ),
    )
    if schedule_warning:
        print(schedule_warning)
    print(predictions.to_string(index=False))
    print(f"\nCSV saved: {csv_path}")
    print(f"PDF saved: {pdf_path}")


if __name__ == "__main__":
    main()
