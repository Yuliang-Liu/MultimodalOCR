import ast
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

EVALUATOR = (
    Path(__file__).resolve().parents[1] / "OCRBench_v2" / "eval_scripts" / "eval.py"
)
NON_STRING_PREDICTIONS = (None, 0, 42, 0.5, True, ["A"], {"A": "option"})
STRING_CASES = (
    ("A", "A", 1),
    ("(A)", "A", 1),
    ("A.", "A", 1),
    ("B", "A", 0),
    ("", "A", 0),
    ("?!", "A", 0),
    ("A/B", "A", 0),
    ("a", "A", 0),
    ("答案是 A", "A", 0),
    ("（甲）", "甲", 1),
)


def load_process_predictions():
    # Run the production scorer without importing dependencies for unrelated metrics.
    nodes = ast.parse(EVALUATOR.read_text(encoding="utf-8")).body
    function = next(
        node for node in nodes
        if isinstance(node, ast.FunctionDef) and node.name == "process_predictions"
    )
    scope = {
        "json": json,
        "tqdm": lambda records: records,
        "TEDS": lambda **kwargs: None,
    }
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(EVALUATOR), "exec"), scope)  # noqa: S102
    return scope["process_predictions"]


def record(category, prediction, answer="A"):
    return {
        "type": category,
        "eval": "multiple choice",
        "answers": [answer],
        "predict": prediction,
        "question": "请选择正确答案",
    }


class ChineseMultipleChoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.process_predictions = staticmethod(load_process_predictions())

    def check_scoring(self, records, scores):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "predictions.json"
            output_path = Path(directory) / "scores.json"
            # ASCII escapes keep input portable with the scorer's default text encoding.
            input_json = json.dumps(records)
            input_path.write_text(input_json, encoding="ascii")

            output = io.StringIO()
            with redirect_stdout(output):
                self.process_predictions(input_path, output_path)

            self.assertEqual(input_path.read_text(encoding="ascii"), input_json)
            output_json = output_path.read_text(encoding="utf-8")
            self.assertIn("请选择正确答案", output_json)
            expected = [
                {**item, "score": score} for item, score in zip(records, scores)
            ]
            self.assertEqual(json.loads(output_json), expected)
        return output.getvalue()

    def check_non_string_predictions(self, category):
        for prediction in NON_STRING_PREDICTIONS:
            with self.subTest(prediction=prediction):
                self.check_scoring([record(category, prediction)], [0])

    def check_string_predictions(self, category):
        for prediction, answer, score in STRING_CASES:
            with self.subTest(prediction=prediction, answer=answer):
                self.check_scoring([record(category, prediction, answer)], [score])

    def test_cognition_non_string_predictions(self):
        self.check_non_string_predictions("cognition VQA cn")

    def test_reasoning_non_string_predictions(self):
        self.check_non_string_predictions("reasoning VQA cn")

    def test_cognition_string_matching(self):
        self.check_string_predictions("cognition VQA cn")

    def test_reasoning_string_matching(self):
        self.check_string_predictions("reasoning VQA cn")

    def test_english_multiple_choice(self):
        records = [
            record("reasoning VQA en", prediction)
            for prediction in (None, 0, "(A)", "B")
        ]
        self.check_scoring(records, [0, 0, 1, 0])

    def test_mixed_file_preserves_records_and_task_averages(self):
        records = [
            record("cognition VQA cn", None),
            record("reasoning VQA cn", 42),
            record("cognition VQA cn", "(A)"),
            record("reasoning VQA en", None),
        ]
        output = self.check_scoring(records, [0, 0, 1, 0])
        self.assertIn(
            "Task cognition VQA cn, total instructions: 2.0, average score: 0.500",
            output,
        )
        self.assertIn(
            "Task reasoning VQA cn, total instructions: 1.0, average score: 0.000",
            output,
        )


if __name__ == "__main__":
    unittest.main()
