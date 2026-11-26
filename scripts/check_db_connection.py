"""Read-only check of the production prediction history (run by test_db_connection.ps1).

Connects through db.connect() (restricted courtvision_app role, sslmode=verify-full)
in a read-only transaction and compares public.prediction_accuracy with the
expected cumulative totals. Prints no credentials.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db  # noqa: E402

EXPECTED = {"total_predictions": 27, "completed": 27, "pending": 0,
            "correct": 19, "incorrect": 8, "accuracy": 0.7037}


def main():
    try:
        settings = db.load_settings(secrets={})
        with db.connect(settings) as connection:
            connection.read_only = True
            # Client-side TLS (this machine <-> pooler), per libpq. verify-full
            # makes libpq refuse unverified or plaintext connections outright.
            client_tls = bool(connection.pgconn.ssl_in_use)
            sslmode = connection.info.get_parameters().get("sslmode")
            who = connection.execute(
                "SELECT current_user AS user_name, "
                "(SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()) AS tls"
            ).fetchone()
            actual = connection.execute(
                "SELECT * FROM public.prediction_accuracy").fetchone()
    except db.HistoryUnavailable as error:
        print(f"FAILED: {error}")
        return 2

    actual = {k: (float(v) if k == "accuracy" and v is not None else v)
              for k, v in actual.items()}
    print(f"Connected as {who['user_name']}")
    print(f"Client TLS (this PC -> pooler): {client_tls}, sslmode={sslmode}")
    # pg_stat_ssl describes the pooler -> database hop, not the client connection.
    print(f"Pooler -> database hop (informational only): {who['tls']}")
    if not client_tls or sslmode != "verify-full":
        print("FAIL: client connection is not using verified TLS")
        return 3
    for key, want in EXPECTED.items():
        got = actual.get(key)
        ok = abs(got - want) < 1e-9 if key == "accuracy" and got is not None else got == want
        print(f"  {key:18} {got!s:>8}  expected {want!s:>8}  {'OK' if ok else 'MISMATCH'}")
    matches = all(
        (abs(actual[k] - v) < 1e-9 if k == "accuracy" else actual[k] == v)
        for k, v in EXPECTED.items())
    print(f"Accuracy {actual['accuracy']:.2%}")
    print("PASS" if matches else "FAIL: totals differ from expected")
    return 0 if matches else 1


if __name__ == "__main__":
    sys.exit(main())
