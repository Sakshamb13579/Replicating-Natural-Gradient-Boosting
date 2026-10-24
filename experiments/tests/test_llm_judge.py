"""Meaningful scoring/protocol tests without loading model weights."""
import sys
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_judge import (JudgeError, OllamaJudge, classification_summary,
                       extract_verdict, order_summary, validate_verdict)
import run_llm_judge


class JudgeTests(unittest.TestCase):
    def test_swap_maps_answers_not_letters(self):
        self.assertEqual(order_summary("A", "B", "first", "second")["consensus_winner_id"], "first")
        result = order_summary("A", "A", "first", "second")
        self.assertFalse(result["consistent_under_swap"])
        self.assertEqual(result["consensus_status"], "order_sensitive")
        self.assertEqual(order_summary("Tie", "Tie", "first", "second")["consensus_status"], "tie")

    def test_agreement_handles_degenerate_and_imperfect_labels(self):
        self.assertIsNone(classification_summary([True, True], [True, True])["cohen_kappa"])
        result = classification_summary([True, True, False, False], [True, False, False, False])
        self.assertEqual(result["accuracy"], .75)
        self.assertAlmostEqual(result["cohen_kappa"], .5)

    def test_reject_string_boolean_and_unexpected_fields(self):
        value = {"clarity": "false", "interval_consistency": True, "evidence_support": True, "reason": "test"}
        with self.assertRaises(JudgeError):
            validate_verdict(value, "pointwise")
        with self.assertRaises(JudgeError):
            validate_verdict({"winner": "A", "reason": "test", "confidence": .99}, "pairwise")

    def test_reject_truncation_and_unstructured_content(self):
        options = {"num_ctx": 2048, "num_predict": 160}
        with self.assertRaises(JudgeError):
            extract_verdict({"done": True, "done_reason": "length", "response": '{"winner":"A"}'}, "pairwise", options)
        with self.assertRaises(JudgeError):
            extract_verdict({"done": True, "done_reason": "stop", "response": 'Here is {"winner":"A","reason":"ok"}'}, "pairwise", options)

    def test_payload_excludes_fixture_expectations(self):
        payload = OllamaJudge().payload("pointwise", {"mean": 1}, "some explanation")
        self.assertNotIn("engineering_expectation", payload["prompt"])
        self.assertIn("<think>\n\n</think>", payload["prompt"])
        self.assertTrue(payload["raw"])

    def test_http_pipeline_persists_and_resumes_without_rejudging(self):
        class Handler(BaseHTTPRequestHandler):
            posts = 0

            def log_message(self, *args):
                pass

            def reply(self, data):
                body = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/version":
                    self.reply({"version": "test-server"})
                else:
                    self.reply({"models": [{"name": "fixture:latest", "digest": "fixed",
                                            "details": {"quantization_level": "Q8_0"}}]})

            def do_POST(self):
                Handler.posts += 1
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if "winner" in data["format"]["required"]:
                    verdict = {"winner": "A", "reason": "Transport test only"}
                else:
                    verdict = {"clarity": True, "interval_consistency": True,
                               "evidence_support": True, "reason": "Transport test only"}
                self.reply({"done": True, "done_reason": "stop", "prompt_eval_count": 500,
                            "eval_count": 40, "response": json.dumps(verdict)})

        root = Path(__file__).resolve().parents[1]
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory(prefix="judge-transport-test-", dir=root / "artifacts") as directory:
                args = SimpleNamespace(cases=root / "data/judge/judge-smoke-v1.json", max_cases=None,
                    model="fixture", base_url=f"http://127.0.0.1:{server.server_port}", timeout=5,
                    repeat_first=True, output=Path(directory) / "output")
                with contextlib.redirect_stdout(io.StringIO()):
                    run_llm_judge.run(args)
                result = json.loads((args.output / "summary.json").read_text())
                self.assertEqual(Handler.posts, 13)
                self.assertEqual(result["valid_pointwise_explanations"], 6)
                self.assertEqual(result["valid_pairwise_cases"], 3)
                self.assertEqual(result["order_consistent_cases"], 0)
                self.assertFalse(result["reportable"])
                self.assertTrue(result["repeatability"][0]["same_scores"])
                self.assertTrue((args.output / "checksums.txt").exists())
                with contextlib.redirect_stdout(io.StringIO()):
                    run_llm_judge.run(args)
                self.assertEqual(Handler.posts, 13, "resume must not submit fresh judgements")
                args.repeat_first = False
                with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
                    run_llm_judge.run(args)
                self.assertEqual(Handler.posts, 13)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    unittest.main()
