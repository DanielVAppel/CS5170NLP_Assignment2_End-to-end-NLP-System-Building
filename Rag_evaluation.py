from __future__ import annotations

import json
import re
import string
import textwrap
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal

# Root folder containing the test/ and train/ folders.
DATA_ROOT = Path("Anotated_Game_Reviews_Data/data")


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------

@dataclass
class EvalSample:
    """One question/reference-answer pair, optionally enriched with annotation data."""
    index: int
    question: str
    reference_answer: str
    annotated_document: str = ""
    annotated_existing_answer: str = ""
    split: str = ""


@dataclass
class EvalResult:
    """Result for a single evaluated sample."""
    index: int
    question: str
    reference_answer: str
    predicted_answer: str
    is_correct: bool

    # Official-rubric-style token metrics.
    exact_match: float = 0.0
    token_precision: float = 0.0
    answer_recall: float = 0.0
    f1: float = 0.0

    annotated_document: str = ""


@dataclass
class EvalReport:
    """Aggregate report for a full evaluation run."""
    mode: str
    split: str
    total: int
    correct: int
    results: list[EvalResult] = field(default_factory=list)

    @property
    def precision(self) -> float:
        """
        Legacy project metric.

        This is kept for backwards compatibility with your older code, but it is
        really closer to substring-based accuracy than traditional precision.
        """
        return self.correct / self.total if self.total else 0.0

    @property
    def legacy_accuracy(self) -> float:
        return self.precision

    @property
    def exact_match(self) -> float:
        return sum(r.exact_match for r in self.results) / self.total if self.total else 0.0

    @property
    def token_precision(self) -> float:
        return sum(r.token_precision for r in self.results) / self.total if self.total else 0.0

    @property
    def answer_recall(self) -> float:
        return sum(r.answer_recall for r in self.results) / self.total if self.total else 0.0

    @property
    def f1(self) -> float:
        return sum(r.f1 for r in self.results) / self.total if self.total else 0.0

    def summary(self) -> str:
        lines = [
            f"\n{'=' * 60}",
            f"  Evaluation Mode         : {self.mode}",
            f"  Split                   : {self.split}",
            f"  Total Samples           : {self.total}",
            f"  Legacy Correct          : {self.correct}/{self.total}",
            f"  Legacy Accuracy         : {self.legacy_accuracy:.2%}",
            f"  Exact Match             : {self.exact_match:.2%}",
            f"  Token Precision         : {self.token_precision:.2%}",
            f"  Answer Recall           : {self.answer_recall:.2%}",
            f"  F1                      : {self.f1:.2%}",
            f"{'=' * 60}",
        ]
        return "\n".join(lines)

    def detailed(self, max_width: int = 120) -> str:
        """Full per-sample breakdown."""
        sections = [self.summary()]

        for result in self.results:
            mark = "O" if result.is_correct else "X"
            predicted = textwrap.shorten(
                result.predicted_answer.replace("\n", " "),
                width=max_width,
                placeholder="...",
            )

            block = (
                f"\n[{mark}] Sample {result.index}\n"
                f"  Q                 : {result.question}\n"
                f"  Reference         : {result.reference_answer}\n"
                f"  Predicted         : {predicted}\n"
                f"  Exact Match       : {result.exact_match:.2%}\n"
                f"  Token Precision   : {result.token_precision:.2%}\n"
                f"  Answer Recall     : {result.answer_recall:.2%}\n"
                f"  F1                : {result.f1:.2%}\n"
            )

            if result.annotated_document:
                doc_preview = textwrap.shorten(
                    result.annotated_document.replace("\n", " "),
                    width=max_width,
                    placeholder="...",
                )
                block += f"  Doc snippet       : {doc_preview}\n"

            sections.append(block)

        return "\n".join(sections)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["metrics"] = {
            "legacy_accuracy": self.legacy_accuracy,
            "exact_match": self.exact_match,
            "token_precision": self.token_precision,
            "answer_recall": self.answer_recall,
            "f1": self.f1,
        }
        return data


# ---------------------------------------------------------------------
# Loading evaluation data
# ---------------------------------------------------------------------

def _split_dir(split: Literal["test", "train"]) -> Path:
    path = DATA_ROOT / split

    if not path.is_dir():
        raise FileNotFoundError(
            f"Expected directory '{path}' does not exist. "
            "Check DATA_ROOT or create the directory."
        )

    return path


def _load_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as file:
        return [line.rstrip("\n") for line in file if line.strip()]


def _load_label_studio(path: Path) -> dict[int, dict]:
    """Return a dictionary keyed by the 1-based id from label_studio_import.json."""
    with path.open(encoding="utf-8") as file:
        items: list[dict] = json.load(file)

    return {item["id"]: item for item in items}


def _load_samples(split: Literal["test", "train"]) -> list[EvalSample]:
    """Load questions.txt and reference_answers.txt for the selected split."""
    directory = _split_dir(split)

    questions = _load_lines(directory / "questions.txt")
    references = _load_lines(directory / "reference_answers.txt")

    if len(questions) != len(references):
        raise ValueError(
            f"Line count mismatch: questions.txt has {len(questions)} lines "
            f"but reference_answers.txt has {len(references)} lines."
        )

    return [
        EvalSample(index=i + 1, question=question, reference_answer=reference)
        for i, (question, reference) in enumerate(zip(questions, references))
    ]


def annotation_add(
    samples: list[EvalSample],
    split: Literal["test", "train"],
) -> list[EvalSample]:
    """Attach Label Studio annotation data to each sample in-place."""
    directory = _split_dir(split)
    label_studio_path = directory / "label_studio_import.json"

    if not label_studio_path.exists():
        raise FileNotFoundError(f"label_studio_import.json not found in '{directory}'.")

    annotations = _load_label_studio(label_studio_path)

    for sample in samples:
        entry = annotations.get(sample.index)

        if entry is None:
            print(f"  [warn] No annotation for sample id={sample.index}, skipping.")
            continue

        data = entry.get("data", {})
        sample.annotated_document = data.get("document", "")
        sample.annotated_existing_answer = data.get("existing_answer", "")
        sample.split = data.get("split", split)

    return samples


# ---------------------------------------------------------------------
# SQuAD-style metrics
# ---------------------------------------------------------------------

def _normalize_legacy(text: str) -> str:
    """Older substring-matching normalization."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def _is_correct(predicted: str, reference: str) -> bool:
    """
    Legacy correctness check.

    Returns True if the normalized reference answer appears inside the
    normalized predicted answer.
    """
    return _normalize_legacy(reference) in _normalize_legacy(predicted)


def _normalize_squad(text: str) -> str:
    """
    SQuAD-style answer normalization:
    lowercase, remove punctuation, remove articles, and fix whitespace.
    """

    def lower(value: str) -> str:
        return value.lower()

    def remove_punctuation(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(char for char in value if char not in exclude)

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    return white_space_fix(remove_articles(remove_punctuation(lower(text))))


def _tokens(text: str) -> list[str]:
    normalized = _normalize_squad(text)
    return normalized.split() if normalized else []


def _reference_answers(reference: str) -> list[str]:
    """
    Support one or more acceptable reference answers.

    By default each line in reference_answers.txt is treated as one reference.
    If you want multiple acceptable answers for one question, separate them with:
    ||
    Example:
        positive || mostly positive || favorable
    """
    refs = [part.strip() for part in reference.split("||") if part.strip()]
    return refs or [""]


def exact_match_score(predicted: str, reference: str) -> float:
    """
    Exact Match: 1.0 if the normalized prediction exactly equals the normalized
    reference, otherwise 0.0. If multiple references are provided with ||,
    the best score is used.
    """
    prediction = _normalize_squad(predicted)

    return max(
        float(prediction == _normalize_squad(ref))
        for ref in _reference_answers(reference)
    )


def token_precision_recall_f1_single(predicted: str, reference: str) -> tuple[float, float, float]:
    pred_tokens = _tokens(predicted)
    ref_tokens = _tokens(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0, 1.0, 1.0

    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def token_precision_recall_f1(predicted: str, reference: str) -> tuple[float, float, float]:
    """
    Token Precision, Answer Recall, and F1.

    If multiple references are provided with ||, the reference with the highest
    F1 is used.
    """
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0

    for ref in _reference_answers(reference):
        precision, recall, f1 = token_precision_recall_f1_single(predicted, ref)

        if f1 > best_f1:
            best_precision = precision
            best_recall = recall
            best_f1 = f1

    return best_precision, best_recall, best_f1


# ---------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------

def _run_evaluation(
    samples: list[EvalSample],
    query_fn: Callable[[str], str],
    mode: str,
    split: str,
    use_annotated_context: bool = False,
    limit: int | None = None,
) -> EvalReport:
    """
    Iterate over samples, call query_fn, and calculate evaluation metrics.
    """
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be a positive integer.")
        samples = samples[:limit]

    results: list[EvalResult] = []
    correct = 0

    for sample_number, sample in enumerate(samples, 1):
        if use_annotated_context and sample.annotated_document:
            query = (
                "Use the following annotated context if it is relevant. "
                "Answer as concisely as possible.\n\n"
                f"Context:\n{sample.annotated_document}\n\n"
                f"Question: {sample.question}"
            )
        else:
            query = sample.question

        try:
            predicted = query_fn(query)
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] Query failed for sample {sample.index}: {exc}")
            predicted = ""

        correct_flag = _is_correct(predicted, sample.reference_answer)

        if correct_flag:
            correct += 1

        exact_match = exact_match_score(predicted, sample.reference_answer)
        token_precision, answer_recall, f1 = token_precision_recall_f1(
            predicted,
            sample.reference_answer,
        )

        results.append(
            EvalResult(
                index=sample.index,
                question=sample.question,
                reference_answer=sample.reference_answer,
                predicted_answer=predicted,
                is_correct=correct_flag,
                exact_match=exact_match,
                token_precision=token_precision,
                answer_recall=answer_recall,
                f1=f1,
                annotated_document=sample.annotated_document,
            )
        )

        status = "O" if correct_flag else "X"
        print(
            f"  [{status}] ({sample_number}/{len(samples)}) "
            f"EM={exact_match:.2f} Recall={answer_recall:.2f} F1={f1:.2f} "
            f"{sample.question[:60]}..."
        )

    return EvalReport(
        mode=mode,
        split=split,
        total=len(samples),
        correct=correct,
        results=results,
    )


def evaluate_basic(
    query_fn: Callable[[str], str],
    split: Literal["test", "train"] = "test",
    verbose: bool = True,
    limit: int | None = None,
) -> EvalReport:
    """
    Evaluate the RAG system using only questions.txt and reference_answers.txt.
    """
    print(f"\n[evaluate_basic] Loading '{split}' split ...")
    samples = _load_samples(split)

    if limit is not None:
        print(f"[evaluate_basic] Limiting run to first {limit} sample(s).")

    print(f"[evaluate_basic] {len(samples)} samples available. Running queries ...\n")

    report = _run_evaluation(
        samples=samples,
        query_fn=query_fn,
        mode="basic",
        split=split,
        use_annotated_context=False,
        limit=limit,
    )

    if verbose:
        print(report.summary())

    return report


def evaluate_annotated(
    query_fn: Callable[[str], str],
    split: Literal["test", "train"] = "test",
    verbose: bool = True,
    limit: int | None = None,
) -> EvalReport:
    """
    Evaluate the RAG system after prepending the annotated document/context
    to each question.
    """
    print(f"\n[evaluate_annotated] Loading '{split}' split ...")
    samples = _load_samples(split)
    samples = annotation_add(samples, split)

    if limit is not None:
        print(f"[evaluate_annotated] Limiting run to first {limit} sample(s).")

    print(f"[evaluate_annotated] {len(samples)} samples available. Running queries ...\n")

    report = _run_evaluation(
        samples=samples,
        query_fn=query_fn,
        mode="annotated",
        split=split,
        use_annotated_context=True,
        limit=limit,
    )

    if verbose:
        print(report.summary())

    return report


def compare_modes(
    query_fn: Callable[[str], str],
    split: Literal["test", "train"] = "test",
    limit: int | None = None,
) -> tuple[EvalReport, EvalReport]:
    """
    Run both basic and annotated evaluation on one split, then print a comparison.
    """
    basic = evaluate_basic(query_fn, split=split, verbose=False, limit=limit)
    annotated = evaluate_annotated(query_fn, split=split, verbose=False, limit=limit)

    print(f"\n{'=' * 72}")
    print(f"  Comparison — split: {split}")
    if limit is not None:
        print(f"  Limited to first {limit} sample(s)")
    print(f"{'=' * 72}")
    print("  Metric              Basic        Annotated    Delta")
    print("  --------------------------------------------------------")

    rows = [
        ("Legacy Accuracy", basic.legacy_accuracy, annotated.legacy_accuracy),
        ("Exact Match", basic.exact_match, annotated.exact_match),
        ("Token Precision", basic.token_precision, annotated.token_precision),
        ("Answer Recall", basic.answer_recall, annotated.answer_recall),
        ("F1", basic.f1, annotated.f1),
    ]

    for name, basic_value, annotated_value in rows:
        delta = annotated_value - basic_value
        arrow = "up" if delta > 0 else ("down" if delta < 0 else "--")
        print(
            f"  {name:<18} {basic_value:>8.2%}    "
            f"{annotated_value:>8.2%}    {arrow} {abs(delta):.2%}"
        )

    print(f"{'=' * 72}\n")

    return basic, annotated


if __name__ == "__main__":
    def _mock_query_fn(question: str) -> str:
        """Toy mock that echoes the first 80 characters of the question."""
        return question[:80]

    basic_report = evaluate_basic(_mock_query_fn, split="test", limit=3)
    annotated_report = evaluate_annotated(_mock_query_fn, split="test", limit=3)
    compare_modes(_mock_query_fn, split="test", limit=3)

    # Optional detailed output:
    # print(basic_report.detailed())
    # print(annotated_report.detailed())
