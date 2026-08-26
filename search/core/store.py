"""JSONL persistence. Every stage writes a plain, inspectable file."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")
            count += 1
    return count


def append_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")
            count += 1
    return count


def read_jsonl(path: Path, schema: type[T]) -> Iterator[T]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield schema.model_validate_json(line)
