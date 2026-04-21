from __future__ import annotations
import json
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

DATA_ROOT = Path("data")          # root that contains test/ and training/
SIMILARITY_THRESHOLD = 0.0        # reserved for fuzzy matching


@dataclass
class EvalSample:
    """One question/answer pair, optionally enriched with annotation data."""
    index: int
    question: str
    reference_answer: str
    # Populated only in annotated mode
    annotated_document: str = ""
    annotated_existing_answer: str = ""
    split: str = ""


@dataclass
class EvalResult:
    """Result for a single sample."""
    index: int
    question: str
    reference_answer: str
    predicted_answer: str
    is_correct: bool
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
        return self.correct / self.total if self.total else 0.0

    def summary(self) -> str:
        lines = [
            f"\n{'=' * 60}",
            f"  Evaluation Mode : {self.mode}",
            f"  Split           : {self.split}",
            f"  Total Samples   : {self.total}",
            f"  Correct         : {self.correct}",
            f"  Precision       : {self.precision:.2%}",
            f"{'=' * 60}",
        ]
        return "\n".join(lines)

    def detailed(self, max_width: int = 80) -> str:
        """Full per-sample breakdown."""
        sections = [self.summary()]
        for r in self.results:
            mark = "O" if r.is_correct else "X"
            block = (
                f"\n[{mark}] Sample {r.index}\n"
                f"  Q : {r.question}\n"
                f"   Reference : {r.reference_answer}\n"
                f"   Predicted : {r.predicted_answer}\n"
            )
            if r.annotated_document:
                doc_preview = textwrap.shorten(r.annotated_document, width=120, placeholder="…")
                block += f"  Doc snippet: {doc_preview}\n"
            sections.append(block)
        return "\n".join(sections)

def _split_dir(split: Literal["test", "training"]) -> Path:
    p = DATA_ROOT / split
    if not p.is_dir():
        raise FileNotFoundError(
            f"Expected directory '{p}' does not exist. "
            "Check DATA_ROOT or create the directory."
        )
    return p


def _load_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]


def _load_label_studio(path: Path) -> dict[int, dict]:
    """Return a dict keyed by 1-based id from label_studio_import.json."""
    with path.open(encoding="utf-8") as fh:
        items: list[dict] = json.load(fh)
    return {item["id"]: item for item in items}


def _load_samples(split: Literal["test", "training"]) -> list[EvalSample]:
    """Load questions + reference answers (basic mode)."""
    d = _split_dir(split)
    questions = _load_lines(d / "questions.txt")
    references = _load_lines(d / "reference_answers.txt")

    if len(questions) != len(references):
        raise ValueError(
            f"Line count mismatch: questions.txt has {len(questions)} lines "
            f"but reference_answers.txt has {len(references)} lines."
        )

    return [
        EvalSample(index=i + 1, question=q, reference_answer=r)
        for i, (q, r) in enumerate(zip(questions, references))
    ]


def annotation_add(
    samples: list[EvalSample],
    split: Literal["test", "training"],
) -> list[EvalSample]:
    """Attach Label Studio annotation data to each sample (in-place)."""
    d = _split_dir(split)
    ls_path = d / "label_studio_import.json"
    if not ls_path.exists():
        raise FileNotFoundError(f"label_studio_import.json not found in '{d}'.")

    annotations = _load_label_studio(ls_path)

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


def _normalize(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def _is_correct(predicted: str, reference: str) -> bool:
    """
    Returns True if the reference answer appears (normalised) inside the
    predicted answer.  This handles cases where the RAG system returns a
    full sentence that *contains* the expected phrase.
    """
    return _normalize(reference) in _normalize(predicted)


def _run_evaluation(
    samples: list[EvalSample],
    query_fn: Callable[[str], str],
    mode: str,
    split: str,
    use_annotated_context: bool = False,
) -> EvalReport:
    """
    Iterate over samples, call query_fn, compare answers.

    Parameters
    ----------
    query_fn
        Callable that accepts a question string and returns the RAG answer.
        In annotated mode, the document context is prepended to the question
        so the retriever gets a richer signal.
    use_annotated_context
        When True, prepend the annotated document to the question before
        querying so the RAG system can leverage the richer context.
    """
    results: list[EvalResult] = []
    correct = 0

    for sample in samples:
        if use_annotated_context and sample.annotated_document:
            # Provide the annotated document as extra context in the query.
            query = (
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

        results.append(
            EvalResult(
                index=sample.index,
                question=sample.question,
                reference_answer=sample.reference_answer,
                predicted_answer=predicted,
                is_correct=correct_flag,
                annotated_document=sample.annotated_document,
            )
        )

        status = "O" if correct_flag else "X"
        print(f"  [{status}] ({sample.index}/{len(samples)}) {sample.question[:60]}…")

    return EvalReport(
        mode=mode,
        split=split,
        total=len(samples),
        correct=correct,
        results=results,
    )


def evaluate_basic(
    query_fn: Callable[[str], str],
    split: Literal["test", "training"] = "test",
    verbose: bool = True,
) -> EvalReport:
    """
    Evaluate the RAG system using questions.txt + reference_answers.txt only.

    Parameters
    ----------
    query_fn : Callable[[str], str]
        A function that takes a question and returns the RAG system's answer.
        Example wrapping a LangChain chain::

            def query_fn(q: str) -> str:
                return qa_chain.invoke({"query": q})["result"]

    split : "test" | "training"
        Which data split directory to read from.
    verbose : bool
        Print per-sample results and the summary.

    Returns
    -------
    EvalReport
    """
    print(f"\n[evaluate_basic] Loading '{split}' split …")
    samples = _load_samples(split)
    print(f"[evaluate_basic] {len(samples)} samples loaded. Running queries …\n")

    report = _run_evaluation(
        samples=samples,
        query_fn=query_fn,
        mode="basic",
        split=split,
        use_annotated_context=False,
    )

    if verbose:
        print(report.summary())

    return report


def evaluate_annotated(
    query_fn: Callable[[str], str],
    split: Literal["test", "training"] = "test",
    verbose: bool = True,
) -> EvalReport:
    print(f"\n[evaluate_annotated] Loading '{split}' split …")
    samples = _load_samples(split)
    samples = annotation_add(samples, split)
    print(f"[evaluate_annotated] {len(samples)} samples loaded. Running queries …\n")

    report = _run_evaluation(
        samples=samples,
        query_fn=query_fn,
        mode="annotated",
        split=split,
        use_annotated_context=True,
    )

    if verbose:
        print(report.summary())

    return report


def compare_modes(
    query_fn: Callable[[str], str],
    split: Literal["test", "training"] = "test",
) -> tuple[EvalReport, EvalReport]:

    basic = evaluate_basic(query_fn, split, verbose=False)
    annotated = evaluate_annotated(query_fn, split, verbose=False)

    print(f"\n{'=' * 60}")
    print(f"  Comparison — split: {split}")
    print(f"{'=' * 60}")
    print(f"  Basic      precision : {basic.precision:.2%}  ({basic.correct}/{basic.total})")
    print(f"  Annotated  precision : {annotated.precision:.2%}  ({annotated.correct}/{annotated.total})")
    delta = annotated.precision - basic.precision
    arrow = "up" if delta > 0 else ("down" if delta < 0 else "—")
    print(f"  Delta                : {arrow} {abs(delta):.2%}")
    print(f"{'=' * 60}\n")

    return basic, annotated


if __name__ == "__main__":
    def _mock_query_fn(question: str) -> str:
        """Toy mock that echoes the first 80 chars of the question as an answer."""
        return question[:80]

    # Basic evaluation
    basic_report = evaluate_basic(_mock_query_fn, split="test")

    # Annotated evaluation
    annotated_report = evaluate_annotated(_mock_query_fn, split="test")

    # Side-by-side comparison
    compare_modes(_mock_query_fn, split="test")

    # Full per-sample breakdown (optional)
    
    # print(basic_report.detailed())
    # print(annotated_report.detailed())