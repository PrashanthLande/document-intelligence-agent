import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalDocument:
    path: Path
    source_url: str

    @property
    def document_id(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    document_id: str
    pages: list[int]
    # Short phrases, at least one of which a relevant chunk must contain.
    evidence: list[str]


@dataclass(frozen=True)
class EvalDataset:
    documents: list[EvalDocument]
    questions: list[EvalQuestion]


def load_dataset(path: str | Path) -> EvalDataset:
    data = json.loads(Path(path).read_text())

    return EvalDataset(
        documents=[
            EvalDocument(path=Path(doc["path"]), source_url=doc["source_url"])
            for doc in data["documents"]
        ],
        questions=[
            EvalQuestion(
                id=q["id"],
                question=q["question"],
                document_id=q["document"],
                pages=q["pages"],
                evidence=q["evidence"],
            )
            for q in data["questions"]
        ],
    )


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace, so line breaks don't matter."""
    return " ".join(text.lower().split())


def is_relevant(
        question: EvalQuestion,
        document_id: str,
        page_numbers: list[int],
        text: str,
) -> bool:
    """A chunk is relevant if it is from the right document, on a gold
    page, and contains one of the evidence phrases."""
    if document_id != question.document_id:
        return False

    if not set(page_numbers) & set(question.pages):
        return False

    chunk_text = normalize(text)

    return any(normalize(phrase) in chunk_text for phrase in question.evidence)
