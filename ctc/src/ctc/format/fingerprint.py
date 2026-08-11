"""
Format fingerprints: a hard train/eval compatibility check.

A checkpoint is only meaningful against the exact prompt format it was trained on. Nothing enforces
that today, so when the two drift the model still runs, still emits plausible text, and still gets a
number -- one that reads as a modelling result. Several of the worst bugs in this project were
exactly this, and each cost days:

* **Chunk-layout drift.** Training wrapped documents one way, eval another; the mask therefore
  isolated different spans than the model was trained under.
* **Doc-id digit range.** Training never showed an id above 697; eval showed ids up to 1423. The
  model had genuinely never seen a four-digit id, and the drop read as long-context collapse.
* **Gold index base.** ``contradiction`` counts gold documents from 1, ``outlier``/``rerank``/``nq``
  from 0. Grade one with the other's convention and every answer is off by one.
* **Marker token ids.** Shards built against one marker id set, a checkpoint repaired against
  another.

The fix is to write a fingerprint next to the data at tokenize time and next to the checkpoint at
train time, then **refuse to evaluate** on a mismatch. Refusing is the point: a warning in a log
scrolls past, and these failures are invisible in the output.

Two design choices worth knowing:

1. **The record is structured, not a single hash.** A bare hash mismatch says "something differs"
   and leaves you to bisect. :meth:`FormatFingerprint.compare` names the field.
2. **Fields carry their own comparison rule.** Most must match exactly, but the digit-range bug is
   not an equality failure -- it is a *containment* failure, where eval exceeds the range training
   covered. See :data:`_RULES`.
3. **Some fields are provenance and are never compared.** ``data_paths`` records which corpora fed
   a task, so a checkpoint can answer "what was I trained on"; comparing it would fail two runs
   over the same shards staged at different paths. See :data:`_PROVENANCE_FIELDS`.

Usage::

    # tokenize time, alongside the shards -- one task per shard directory
    spec.fingerprint(query_position=..., tokenizer=...).write(shard_dir)

    # train time, alongside every checkpoint -- a mix trains several tasks, so this is a SET.
    # ctc.train.FormatFingerprintCallback does this automatically, collecting from the shard dirs.
    FingerprintSet(formats).write(ckpt_dir)

    # eval time -- raises before a single token is generated
    check_or_explain_missing(spec.fingerprint(query_position=...), ckpt_dir)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "FormatFingerprint",
    "FingerprintSet",
    "Mismatch",
    "FormatMismatchError",
    "TaskNotTrainedError",
    "FINGERPRINT_FILENAME",
    "hash_prompt",
    "collect_fingerprints",
    "conflicting_formats",
    "chunk_layout_for",
]


def chunk_layout_for(emit: str, chunk_by: str, doc_markers: bool = True) -> str:
    """
    The canonical ``chunk_layout`` name for a set of converter options.

    One function so the converter that writes a shard's fingerprint and the evaluator that checks it
    cannot drift into two spellings of the same layout -- which would read as a format mismatch on
    data that matches perfectly.

    **This names the token stream, not the attention mask.** The box markers are present in the
    ``full`` arm too: "full" is a mask, not a token layout, and the full-vs-chunked comparison is
    run over identically tokenized shards precisely so the mask is the only difference. So a
    checkpoint trained on ``wrap_documents`` shards may legitimately be evaluated with ``--attn
    full``, and the fingerprint must not object. What it does object to is evaluating it against a
    prompt rendered *without* the markers, which is a different token stream and costs ~0.01 f1.

    :param emit: ``"dense"`` (wrapped tokens only) or ``"landmark"`` (packed into landmark windows).
    :param chunk_by: ``"document"`` (each ``documents[i]``) or ``"line"`` (each matching item line;
        OOLONG only).
    :param doc_markers: Whether the ``<|box_start|>``/``<|box_end|>`` boundary tokens are emitted.

    :returns: One of ``none``, ``wrap_documents``, ``wrap_lines``, ``landmark_documents``,
        ``landmark_lines``.

    :raises ValueError: On an unknown ``emit`` or ``chunk_by``.
    """
    if emit not in ("dense", "landmark"):
        raise ValueError(f"emit must be 'dense' or 'landmark', got {emit!r}")
    if chunk_by not in ("document", "line"):
        raise ValueError(f"chunk_by must be 'document' or 'line', got {chunk_by!r}")
    if not doc_markers:
        # No boundary tokens at all: the marker-free baseline. Whether the build *would* have
        # chunked by document or by line is not observable in the stream, so it is not recorded --
        # recording it would make two byte-identical shards compare as different formats.
        return "none"
    unit = "documents" if chunk_by == "document" else "lines"
    return f"wrap_{unit}" if emit == "dense" else f"landmark_{unit}"


#: Written into both the shard directory and the checkpoint directory.
FINGERPRINT_FILENAME = "ctc_format_fingerprint.json"

#: Bump when a change makes old fingerprints uncomparable rather than merely different. Comparing
#: across schema versions is refused outright -- silently treating an absent field as "matches" is
#: how a guard stops guarding.
SCHEMA_VERSION = 1


def hash_prompt(*parts: Optional[str]) -> str:
    """
    Hash the exact strings that will be shown to the model.

    :param parts: Instruction and template strings, in a stable order. ``None`` entries are
        recorded distinctly from ``""`` -- "no template" and "empty template" are different formats.

    :returns: A short hex digest.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(b"\x00" if p is None else b"\x01")
        if p is not None:
            h.update(p.encode("utf-8"))
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class Mismatch:
    """One field that failed its compatibility rule."""

    field: str
    train: Any
    eval: Any
    rule: str
    why: str

    def __str__(self) -> str:
        return f"  {self.field}: train={self.train!r} eval={self.eval!r}\n      {self.why}"


class FormatMismatchError(RuntimeError):
    """Raised when an eval format is incompatible with the format a checkpoint was trained on."""

    def __init__(self, mismatches: Sequence[Mismatch]):
        self.mismatches = list(mismatches)
        body = "\n".join(str(m) for m in self.mismatches)
        super().__init__(
            f"eval format is incompatible with the checkpoint's training format "
            f"({len(self.mismatches)} field(s)):\n{body}\n\n"
            "This is a hard error on purpose. Every one of these failures produces plausible "
            "output and a plausible-looking score, so continuing would generate a number that "
            "silently means nothing. Fix the eval config, or pass --ignore-format-fingerprint if "
            "you have established the difference is benign."
        )


def _exact(name: str, train: Any, eval_: Any) -> Optional[str]:
    if train == eval_:
        return None
    return f"must be identical; the checkpoint was trained under a different {name}"


def _within(name: str, train: Any, eval_: Any) -> Optional[str]:
    """Eval's numeric range must sit inside the range training covered."""
    if train is None or eval_ is None:
        return None
    (tlo, thi), (elo, ehi) = tuple(train), tuple(eval_)
    if elo >= tlo and ehi <= thi:
        return None
    return (
        f"eval {name} falls outside the range seen in training; the model has never been "
        f"shown values in this range, and the resulting drop reads as a capability limit"
    )


def _optional_exact(name: str, train: Any, eval_: Any) -> Optional[str]:
    """Skip when either side did not record the field, otherwise require equality."""
    if train is None or eval_ is None:
        return None
    return _exact(name, train, eval_)


#: field -> (rule name, comparison). A field absent here is metadata and is not compared.
_RULES = {
    "task": ("exact", _exact),
    "prompt_hash": ("exact", _exact),
    "serializer": ("exact", _exact),
    "item_separator": ("exact", _exact),
    "gold_index_base": ("exact", _exact),
    "prompt_shape": ("exact", _exact),
    "query_position": ("exact", _exact),
    "chunk_layout": ("exact", _exact),
    "marker_token_ids": ("exact", _optional_exact),
    "tokenizer": ("exact", _optional_exact),
    "doc_id_range": ("within", _within),
}

#: Fields recorded for provenance and deliberately **not** compared. Two runs over byte-identical
#: data staged at different paths -- weka, node-local ``/data``, an S3 mirror -- are compatible, and
#: comparing paths would fail them all. It would also make the guard fire constantly for a reason
#: nobody can act on, which is how a guard gets switched off. They are also what
#: :meth:`FormatFingerprint.merged_with` accumulates rather than deduplicates.
_PROVENANCE_FIELDS = ("data_paths", "notes")


@dataclass(frozen=True)
class FormatFingerprint:
    """
    Everything about a data format that a checkpoint is bound to.

    :param task: Task name.
    :param prompt_hash: Digest of the exact instruction and templates, from :func:`hash_prompt`.
    :param serializer: Which document serializer rendered the context block.
    :param item_separator: The delimiter between items -- the chunk boundary the masks split on.
    :param gold_index_base: ``0`` or ``1``; differs per task and is an easy off-by-one.
    :param prompt_shape: ``"unified"`` (generic alpaca header, task instruction positioned with the
        documents) or ``"classic"`` (task instruction in the header, per-example query positioned).
        Which one a task uses is a property of the task, not a knob -- see :class:`TaskSpec`.
    :param query_position: ``"before"``, ``"after"`` or ``"both"``. Recorded because it is a real
        knob that really varies (``both`` and ``before`` and ``after`` are all in use across runs)
        and it changes the token stream substantially -- ``both`` repeats the entire query block on
        the far side of the documents. Two checkpoints differing only here would otherwise share a
        fingerprint, and the guard would pass on a format that genuinely differs.
    :param chunk_layout: Chunk-wrapping scheme; see :func:`chunk_layout_for` for the vocabulary
        and for why it describes the TOKEN STREAM rather than the attention mask.
    :param doc_id_range: ``(min, max)`` document id actually present. Compared by containment.
    :param marker_token_ids: Reserved marker ids, when the format uses them.
    :param tokenizer: Tokenizer identifier.
    :param data_paths: Which data this format was built from -- source corpora, shard directories,
        or both. Several entries when a task is fed by a mix, which is why
        :func:`collect_fingerprints` unions them rather than deduplicating whole records. Recorded,
        never compared: the same shards on weka and on node-local ``/data`` are the same shards.
    :param notes: Free-form provenance. Recorded, never compared.
    """

    task: str
    prompt_hash: str
    serializer: str
    item_separator: str
    gold_index_base: int
    prompt_shape: str = "classic"
    query_position: str = "after"
    chunk_layout: str = "none"
    doc_id_range: Optional[Tuple[int, int]] = None
    marker_token_ids: Optional[Tuple[int, ...]] = None
    tokenizer: Optional[str] = None
    data_paths: Tuple[str, ...] = ()
    notes: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.gold_index_base not in (0, 1):
            raise ValueError(
                f"gold_index_base must be 0 or 1, got {self.gold_index_base!r}. Tasks disagree "
                "(contradiction is 1-based, outlier/rerank/nq are 0-based), which is exactly why "
                "it is recorded."
            )
        if self.doc_id_range is not None:
            lo, hi = self.doc_id_range
            if lo > hi:
                raise ValueError(f"doc_id_range {self.doc_id_range!r} is inverted")
        if self.prompt_shape not in ("unified", "classic"):
            raise ValueError(
                f"prompt_shape must be 'unified' or 'classic', got {self.prompt_shape!r}"
            )
        if self.query_position not in ("before", "after", "both"):
            raise ValueError(
                f"query_position must be 'before', 'after' or 'both', got {self.query_position!r}"
            )

    # ── comparison ──────────────────────────────────────────────────────────────────────────────

    def compare(self, trained: "FormatFingerprint") -> List[Mismatch]:
        """
        Compare this (eval-side) fingerprint against the one recorded at training time.

        :param trained: The fingerprint written next to the checkpoint or its shards.

        :returns: Every field that failed its rule; empty means compatible.

        :raises ValueError: If the two use different schema versions, which cannot be compared
            meaningfully.
        """
        if trained.schema_version != self.schema_version:
            raise ValueError(
                f"fingerprint schema {trained.schema_version} vs {self.schema_version}; "
                "regenerate the training-side fingerprint rather than comparing across versions"
            )
        out = []
        for name, (rule, fn) in _RULES.items():
            why = fn(name, getattr(trained, name), getattr(self, name))
            if why is not None:
                out.append(
                    Mismatch(
                        field=name,
                        train=getattr(trained, name),
                        eval=getattr(self, name),
                        rule=rule,
                        why=why,
                    )
                )
        return out

    def same_format_as(self, other: "FormatFingerprint") -> bool:
        """
        :param other: Another fingerprint.

        :returns: Whether the two describe the same format, ignoring provenance. Distinct from
            ``==``: two shard directories built the same way from different corpora produce equal
            formats and different :attr:`data_paths`, and collapsing them by ``==`` would lose one
            of the paths.
        """
        return not self.compare(other)

    def merged_with(self, other: "FormatFingerprint") -> "FormatFingerprint":
        """
        Fold another record of the same format into this one, accumulating provenance.

        :param other: A fingerprint describing the same format.

        :returns: This record with ``other``'s data paths appended (order preserved, duplicates
            dropped) and its notes merged in.

        :raises ValueError: If the two are not the same format. Merging different formats would
            manufacture a record matching neither.
        """
        if not self.same_format_as(other):
            raise ValueError(
                "refusing to merge two different formats into one record; keep them as separate "
                f"entries: {[m.field for m in self.compare(other)]}"
            )
        paths = list(self.data_paths) + [p for p in other.data_paths if p not in self.data_paths]
        return replace(self, data_paths=tuple(paths), notes={**other.notes, **self.notes})

    def with_data_paths(self, *paths: str) -> "FormatFingerprint":
        """
        :param paths: Paths to record, appended in order, duplicates dropped.

        :returns: A copy carrying them.
        """
        new = list(self.data_paths) + [p for p in paths if p not in self.data_paths]
        return replace(self, data_paths=tuple(new))

    def require_compatible_with(self, trained: "FormatFingerprint") -> None:
        """
        Raise unless this eval format is compatible with ``trained``.

        :param trained: The training-time fingerprint.

        :raises FormatMismatchError: On any incompatible field.
        """
        mismatches = self.compare(trained)
        if mismatches:
            raise FormatMismatchError(mismatches)

    # ── i/o ─────────────────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """:returns: A JSON-ready mapping."""
        d = asdict(self)
        for k in ("doc_id_range", "marker_token_ids", "data_paths"):
            if d[k] is not None:
                d[k] = list(d[k])
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FormatFingerprint":
        """
        :param d: A mapping produced by :meth:`to_dict`.

        :returns: The fingerprint.

        :raises KeyError: If a required field is absent -- a truncated record must not read as a
            match.
        """
        d = dict(d)
        if d.get("doc_id_range") is not None:
            d["doc_id_range"] = tuple(d["doc_id_range"])
        if d.get("marker_token_ids") is not None:
            d["marker_token_ids"] = tuple(d["marker_token_ids"])
        # Tuple, not list: the record is frozen and hashable, and a list here would break equality
        # against a freshly derived fingerprint.
        d["data_paths"] = tuple(d.get("data_paths") or ())
        known = {f for f in cls.__dataclass_fields__}
        extra = set(d) - known
        if extra:
            # Forward compatibility: an older reader must not silently drop a field a newer writer
            # considered load-bearing.
            raise ValueError(
                f"fingerprint has unknown field(s) {sorted(extra)}; this reader is too old to "
                "validate it. Upgrade ctc rather than ignoring them."
            )
        return cls(**d)

    def write(self, directory: Path) -> Path:
        """
        Write this single fingerprint into ``directory``, replacing anything already there.

        A shard directory holds one task, so this is its natural writer. A *checkpoint* directory
        usually holds several -- training mixes tasks -- so use :meth:`FingerprintSet.write` there,
        or this will drop the others.

        :param directory: Shard directory (at tokenize time).

        :returns: The path written.
        """
        return FingerprintSet([self]).write(directory)

    @classmethod
    def read(cls, directory: Path) -> Optional["FormatFingerprint"]:
        """
        Read a single fingerprint from ``directory``.

        :param directory: Shard directory.

        :returns: The fingerprint, or ``None`` if the directory predates fingerprinting.

        :raises ValueError: If the directory records more than one format. Picking one arbitrarily
            would answer a question the caller did not ask -- use :meth:`FingerprintSet.read` and
            :meth:`FingerprintSet.for_task`.
        """
        found = FingerprintSet.read(directory)
        if found is None:
            return None
        if len(found.formats) != 1:
            raise ValueError(
                f"{Path(directory) / FINGERPRINT_FILENAME} records {len(found.formats)} formats "
                f"({', '.join(found.tasks)}); read it as a FingerprintSet and select a task"
            )
        return found.formats[0]

    def evolve(self, **changes: Any) -> "FormatFingerprint":
        """
        :param changes: Field overrides.

        :returns: A copy with those fields replaced.
        """
        return replace(self, **changes)


@dataclass(frozen=True)
class FingerprintSet:
    """
    Every format a checkpoint was trained under.

    A checkpoint is rarely bound to one format. The canonical SFT mix trains five tasks at once,
    so its record has five entries, and eval asks a narrower question than "does this checkpoint
    match" -- it asks "was *this task* trained, and in the format I am about to use".

    Two entries may share a task name. That is not a mistake: a curriculum can legitimately train
    one task under two layouts, and such a checkpoint is compatible with both. :meth:`for_task`
    therefore returns a list, and :meth:`require_compatible` passes if *any* of them matches.

    :param formats: The recorded formats, in a stable order.
    """

    formats: Tuple[FormatFingerprint, ...]

    def __init__(self, formats: Sequence[FormatFingerprint]):
        object.__setattr__(self, "formats", tuple(formats))
        if not self.formats:
            raise ValueError(
                "a fingerprint set must record at least one format; writing an empty one would "
                "make an unfingerprinted checkpoint look fingerprinted"
            )

    @property
    def tasks(self) -> List[str]:
        """:returns: The task names recorded, deduplicated, in first-seen order."""
        seen: Dict[str, None] = {}
        for fp in self.formats:
            seen.setdefault(fp.task, None)
        return list(seen)

    def for_task(self, task: str) -> List[FormatFingerprint]:
        """
        :param task: Task name.

        :returns: Every recorded format for that task; empty if it was not trained.
        """
        return [fp for fp in self.formats if fp.task == task]

    def merge(self, other: "FingerprintSet") -> "FingerprintSet":
        """
        Combine two sets, folding same-format records together.

        Deduplication is on *format*, not on the whole record, and matching records have their
        provenance accumulated. That distinction is the mixed-corpus case: contradiction built from
        PubMed and from FEVER yields two identical formats with different
        :attr:`FormatFingerprint.data_paths`, and dropping the second as a duplicate would erase
        half of what the model was trained on.

        :param other: The set to fold in.

        :returns: The union.
        """
        return FingerprintSet(_fold(list(self.formats), other.formats))

    def require_compatible(self, eval_fp: FormatFingerprint) -> None:
        """
        Raise unless ``eval_fp`` matches one of the recorded formats for its task.

        :param eval_fp: The format eval is about to use.

        :raises FormatMismatchError: If the task was recorded but no entry matches. The reported
            mismatches come from the *closest* entry, so a curriculum's several formats do not
            produce several unrelated error lists.
        :raises TaskNotTrainedError: If the task was never trained. This is a distinct failure:
            nothing is mismatched, there is simply nothing to check against.
        """
        candidates = self.for_task(eval_fp.task)
        if not candidates:
            raise TaskNotTrainedError(eval_fp.task, self.tasks)
        attempts = [eval_fp.compare(c) for c in candidates]
        if any(not a for a in attempts):
            return
        raise FormatMismatchError(min(attempts, key=len))

    # ── i/o ─────────────────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """:returns: A JSON-ready mapping."""
        return {
            "schema_version": SCHEMA_VERSION,
            "formats": [fp.to_dict() for fp in self.formats],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FingerprintSet":
        """
        :param d: A mapping produced by :meth:`to_dict`.

        :returns: The set.

        :raises ValueError: If the record is not a fingerprint set.
        """
        if "formats" not in d:
            raise ValueError(
                "not a fingerprint set: no 'formats' key. A record written by ctc always has one, "
                "even for a single task."
            )
        return cls([FormatFingerprint.from_dict(f) for f in d["formats"]])

    def write(self, directory: Path) -> Path:
        """
        Write the set into ``directory``, replacing any existing record.

        :param directory: Shard directory (tokenize time) or checkpoint directory (train time).

        :returns: The path written.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / FINGERPRINT_FILENAME
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def read(cls, directory: Path) -> Optional["FingerprintSet"]:
        """
        :param directory: Shard or checkpoint directory.

        :returns: The recorded set, or ``None`` if the directory predates fingerprinting. Callers
            must decide what to do with ``None`` explicitly -- see :func:`check_or_explain_missing`.
        """
        path = Path(directory) / FINGERPRINT_FILENAME
        if not path.exists():
            return None
        return cls.from_dict(json.loads(path.read_text()))


def _fold(
    into: List[FormatFingerprint], incoming: Sequence[FormatFingerprint]
) -> List[FormatFingerprint]:
    """
    Add records to a list, merging any that describe the same format.

    :param into: Accumulator, mutated and returned.
    :param incoming: Records to add.

    :returns: ``into``.
    """
    for fp in incoming:
        for i, existing in enumerate(into):
            if existing.task == fp.task and existing.same_format_as(fp):
                into[i] = existing.merged_with(fp)
                break
        else:
            into.append(fp)
    return into


def collect_fingerprints(
    directories: Sequence[Path],
    *,
    extra: Sequence[FormatFingerprint] = (),
    allow_missing: bool = False,
    record_source_paths: bool = True,
) -> Tuple[FingerprintSet, List[str]]:
    """
    Gather the formats recorded across several shard directories.

    Used by both the training callback and ``ctc-fingerprint collect``, which is the point: a
    checkpoint stamped at train time and one stamped afterwards must record the same thing, and
    two implementations of "collect" would eventually not.

    **Mixing is the case this has to get right.** Different tasks fold in side by side, one entry
    each. The *same* task from several directories -- contradiction from PubMed and from FEVER, or
    one task spread over per-rung directories -- produces several records with an identical format,
    which are merged into one entry whose :attr:`FormatFingerprint.data_paths` lists every source.
    Deduplicating on the whole record instead would drop all but the first, leaving a checkpoint
    that names one corpus out of five.

    :param directories: Shard directories to read.
    :param extra: Fingerprints to include that no directory records.
    :param allow_missing: Tolerate a directory with no record. Off by default -- a partial record
        is worse than none, because the eval guard then reports a trained task as untrained.
    :param record_source_paths: Record each directory on the fingerprints read from it, so the
        checkpoint says what it was trained on even when the shard writer recorded no path itself.

    :returns: ``(set, skipped)`` where ``skipped`` names the directories that had no record. It is
        non-empty only when ``allow_missing`` is set, and the caller must disclose it.

    :raises FileNotFoundError: If a directory has no record and ``allow_missing`` is not set.
    :raises ValueError: If nothing at all was found.
    """
    collected: List[FormatFingerprint] = list(extra)
    missing: List[str] = []
    for d in directories:
        found = FingerprintSet.read(Path(d))
        if found is None:
            missing.append(str(d))
            continue
        source = str(Path(d).resolve())
        _fold(
            collected,
            [fp.with_data_paths(source) if record_source_paths else fp for fp in found.formats],
        )

    if missing and not allow_missing:
        raise FileNotFoundError(
            f"no {FINGERPRINT_FILENAME} in {len(missing)} director(ies):\n  "
            + "\n  ".join(missing)
            + "\n\nFingerprint them first (ctc-fingerprint write --dir <shards> --task <name> …), "
            "or allow them to be skipped. A partial record makes the eval guard report a trained "
            "task as out-of-distribution."
        )
    if not collected:
        raise ValueError(
            "no formats found. Point this at the shard directories the run reads, or supply them "
            "explicitly. Recording nothing while appearing to record something is the one outcome "
            "this guard must not have."
        )
    return FingerprintSet(collected), missing


def conflicting_formats(fingerprints: FingerprintSet) -> Dict[str, List[str]]:
    """
    Find tasks recorded under more than one format.

    Legitimate for a curriculum that varies the layout deliberately, and
    :meth:`FingerprintSet.require_compatible` accepts either. But it is also what accidental drift
    between two shard builds looks like, and that case is otherwise invisible.

    :param fingerprints: The set to inspect.

    :returns: task -> the field names that differ, for each task with several formats.
    """
    by_task: Dict[str, List[FormatFingerprint]] = {}
    for fp in fingerprints.formats:
        by_task.setdefault(fp.task, []).append(fp)
    return {
        task: sorted({m.field for m in fps[0].compare(fps[1])})
        for task, fps in by_task.items()
        if len(fps) > 1
    }


class TaskNotTrainedError(RuntimeError):
    """Raised when a checkpoint's fingerprint records no format for the task being graded."""

    def __init__(self, task: str, trained: Sequence[str]):
        self.task = task
        self.trained = list(trained)
        super().__init__(
            f"this checkpoint records no training format for task {task!r}; it was trained on "
            f"{', '.join(self.trained)}.\n"
            "Either the task name is wrong, or this is a deliberate out-of-distribution eval. "
            "The guard cannot verify an OOD eval -- there is no training format to compare "
            "against -- so pass --ignore-format-fingerprint and the result will record that "
            "compatibility was unverified."
        )


def check_or_explain_missing(
    eval_fp: FormatFingerprint, ckpt_dir: Path, *, strict: bool = True
) -> Optional[str]:
    """
    Enforce the fingerprint against a checkpoint directory.

    :param eval_fp: The format eval is about to use.
    :param ckpt_dir: Checkpoint directory to read the training-time fingerprint from.
    :param strict: When ``True`` an *absent* fingerprint is also an error. Checkpoints trained
        before fingerprinting existed have none, so batch-grading old runs needs ``strict=False``
        -- but then the guard is off, and the caller should say so in the results file.

    :returns: A warning string when the check could not be performed and ``strict`` is ``False``,
        else ``None``.

    :raises FormatMismatchError: On an incompatible fingerprint. Always raised, in both modes: an
        actual mismatch is never merely a warning.
    :raises FileNotFoundError: When the fingerprint is absent and ``strict`` is ``True``.
    :raises TaskNotTrainedError: When the task was not trained and ``strict`` is ``True``.
    """
    trained = FingerprintSet.read(ckpt_dir)
    if trained is None:
        msg = (
            f"no {FINGERPRINT_FILENAME} in {ckpt_dir}; train/eval format compatibility is "
            "UNVERIFIED for this run"
        )
        if strict:
            raise FileNotFoundError(
                msg + ". Pass strict=False (or --ignore-format-fingerprint) to grade a checkpoint "
                "that predates fingerprinting."
            )
        return msg
    try:
        trained.require_compatible(eval_fp)
    except TaskNotTrainedError as e:
        if strict:
            raise
        return (
            f"{e.task!r} is not among this checkpoint's trained tasks ({', '.join(e.trained)}); "
            "grading it as out-of-distribution, format compatibility UNVERIFIED"
        )
    return None
