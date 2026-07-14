"""
Locust load test for the emotion-detection analysis pipeline.

Run against perf_test_server.py (NOT against `streamlit run app.py`
directly - Streamlit's WebSocket protocol isn't something plain Locust
HttpUser speaks). See PERFORMANCE_TESTING.md for full instructions.

Example (single simulated user, i.e. plain response-time measurement):
    locust -f locustfile.py --host=http://127.0.0.1:8500 \
        --users 1 --spawn-rate 1 --run-time 1m --headless \
        --csv=results_single_user

Example (concurrent-session test, 10 simultaneous "browser tabs"):
    locust -f locustfile.py --host=http://127.0.0.1:8500 \
        --users 10 --spawn-rate 2 --run-time 2m --headless \
        --csv=results_concurrent_10
"""
import random

from locust import HttpUser, between, task

# one real example per field, matching src/ui/field_problem_capture.py's
# own placeholders, so the test payloads look like real student input
SAMPLE_PROBLEMS = [
    ("Computer Science", "I don't get how recursion works, I've read the explanation five times."),
    ("Mathematics", "I can't solve this integration problem, I keep getting the wrong sign."),
    ("Physics", "I don't understand torque at all, none of this is clicking."),
    ("Chemistry", "Balancing equations is confusing and I don't know where to even start."),
    ("Biology", "I already understand the Krebs cycle, can we move on to something harder?"),
    ("Psychology", "This theory feels kind of interesting, I want to know how it applies elsewhere."),
]


class StudentUser(HttpUser):
    # 2-5s between requests per simulated user, roughly how long a
    # student would spend reading a response before submitting another
    wait_time = between(2, 5)

    @task
    def analyze(self):
        field, problem = random.choice(SAMPLE_PROBLEMS)
        self.client.post("/analyze", json={"field": field, "problem": problem, "ai_enabled": True})
