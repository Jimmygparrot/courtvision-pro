import socket
import time
import urllib.request
import urllib.error
import streamlit as st

st.title("CourtVision Pro — Network Diagnostics")
st.write("Tests connectivity to the NBA statistics API.")
st.info("Run this test locally and on Streamlit Cloud to compare results.")

HOST = "stats.nba.com"
TIMEOUT = 5

if st.button("Run Network Diagnostics"):
    # Test 1: DNS
    with st.spinner("Checking DNS..."):
        start = time.monotonic()
        try:
            addresses = socket.getaddrinfo(HOST, 443, type=socket.SOCK_STREAM)
            ip = addresses[0][4][0]
            st.success(f"DNS: PASS ({time.monotonic() - start:.2f}s)")
            st.write(f"Resolved IP: {ip}")
        except Exception as exc:
            st.error(f"DNS: FAIL — {type(exc).__name__}: {exc}")

    # Test 2: TCP port 443
    with st.spinner("Checking TCP connection..."):
        start = time.monotonic()
        try:
            with socket.create_connection((HOST, 443), timeout=TIMEOUT):
                pass
            st.success(f"TCP port 443: PASS ({time.monotonic() - start:.2f}s)")
        except Exception as exc:
            st.error(f"TCP port 443: FAIL ({time.monotonic() - start:.2f}s)")
            st.code(f"{type(exc).__name__}: {exc}")

    # Test 3: HTTPS
    with st.spinner("Checking HTTPS response..."):
        start = time.monotonic()
        try:
            request = urllib.request.Request(
                f"https://{HOST}/stats/",
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                status = response.status
            st.success(f"HTTPS: HTTP {status} ({time.monotonic() - start:.2f}s)")
        except urllib.error.HTTPError as exc:
            st.warning(f"HTTPS: HTTP {exc.code} ({time.monotonic() - start:.2f}s)")
        except Exception as exc:
            st.error(f"HTTPS: FAIL ({time.monotonic() - start:.2f}s)")
            st.code(f"{type(exc).__name__}: {exc}")

    st.write("Diagnostics complete.")

st.divider()
st.subheader("NBA PlayerGameLog API Test")

if st.button("Test NBA PlayerGameLog API"):
    import time
    from nba_api.stats.endpoints import playergamelog

    start = time.monotonic()

    try:
        response = playergamelog.PlayerGameLog(
            player_id=1628369,
            season="2025-26",
            timeout=10,
        )
        games = response.get_data_frames()[0]
        elapsed = time.monotonic() - start
        st.success(f"SUCCESS: {len(games)} games in {elapsed:.2f} seconds")
    except Exception as error:
        elapsed = time.monotonic() - start
        st.error(f"FAILED after {elapsed:.2f} seconds")
        st.code(f"{type(error).__name__}: {error}")
