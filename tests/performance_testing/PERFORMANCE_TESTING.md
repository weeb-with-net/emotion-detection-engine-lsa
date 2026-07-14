# Performance Testing — Instructions

Two files are provided: `perf_test_server.py` and `locustfile.py`. Copy both into
the **root** of your local `emotion-detection-engine-lsa` repo (same folder as
`app.py`) before running anything.

## 0. Install the extra tools (one-time)

In your existing project virtual environment:

```
pip install fastapi uvicorn locust
```

## 1. Why a wrapper instead of testing `streamlit run app.py` directly

Streamlit talks to the browser over WebSocket, not plain HTTP, so Locust can't
simulate an "Analyze" click against it directly. The actual computational cost
(BiLSTM + BERT inference, the decision engine, the LLM call) all lives inside
`run_analysis()` — the same function `app.py` calls — so `perf_test_server.py`
exposes that exact function as a real HTTP endpoint. The Streamlit rendering
layer itself isn't the bottleneck, so this tests the part that matters.

## 2. Run it in FULL mode (BiLSTM + BERT + real LLM)

In one terminal, with your normal `.env` in place (real `GEMINI_API_KEY` /
`OPENROUTER_API_KEY`, `DISABLE_BERT` and `DISABLE_AI_RESPONSE` **not** set):

```
python perf_test_server.py
```

Leave this running. In a **second** terminal, from the same folder:

```
# Single-user response time (no concurrency)
locust -f locustfile.py --host=http://127.0.0.1:8500 --users 1 --spawn-rate 1 --run-time 1m --headless --csv=results_full_single

# Concurrent-session test (10 simulated simultaneous students)
locust -f locustfile.py --host=http://127.0.0.1:8500 --users 10 --spawn-rate 2 --run-time 2m --headless --csv=results_full_concurrent
```

While the concurrent run is going, open Task Manager → Performance tab and
jot down the peak CPU % and peak Memory usage you see during that ~2 minutes.

Stop the server (Ctrl+C in the first terminal) once both Locust runs finish.

## 3. Run it in DEGRADED mode (matches the public Streamlit Cloud deployment)

Close the terminal from step 2, open a new one, and set the same flags the
real app checks before starting the server again:

```
set DISABLE_BERT=true
set DISABLE_AI_RESPONSE=true
python perf_test_server.py
```

Then, in a second terminal, repeat the same two Locust commands but with
different `--csv` names so they don't overwrite the full-mode results:

```
locust -f locustfile.py --host=http://127.0.0.1:8500 --users 1 --spawn-rate 1 --run-time 1m --headless --csv=results_degraded_single

locust -f locustfile.py --host=http://127.0.0.1:8500 --users 10 --spawn-rate 2 --run-time 2m --headless --csv=results_degraded_concurrent
```

Again, note peak CPU % and Memory during the concurrent run.

## 4. What to send back to me

Each `--csv=NAME` run produces `NAME_stats.csv`, `NAME_stats_history.csv`,
and `NAME_failures.csv` (that last one may be empty/absent if nothing failed
— that's fine, just say so). So in total:

- `results_full_single_stats.csv`
- `results_full_concurrent_stats.csv` (+ failures file if non-empty)
- `results_degraded_single_stats.csv`
- `results_degraded_concurrent_stats.csv` (+ failures file if non-empty)
- The peak CPU % and peak Memory you noted from Task Manager for the two
  concurrent runs (full mode and degraded mode)

Paste the CSV contents directly in chat, or upload the files — either works.
I'll turn those into the Performance Testing document once I have them.
