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

Usage::

    # tokenize time, alongside the shards; and train time, alongside the checkpoint
    fp.write(shard_dir)

    # eval time -- raises before a single token is generated
    FormatFingerprint.of(task=...).require_compatible_with(FormatFingerprint.read(ckpt_dir))
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "FormatFingerprint",
    "Mismatch",
    "FormatMismatchError",
    "FINGERPRINT_FILENAME",
    "hash_prompt",
]

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
    :param chunk_layout: Chunk-wrapping scheme, e.g. ``"wrap_documents"`` or ``"none"``.
    :param doc_id_range: ``(min, max)`` document id actually present. Compared by containment.
    :param marker_token_ids: Reserved marker ids, when the format uses them.
    :param tokenizer: Tokenizer identifier.
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
            raise ValueError(f"prompt_shape must be 'unified' or 'classic', got {self.prompt_shape!r}")
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
                        field=name, train=getattr(trained, name), eval=getattr(self, name),
                        rule=rule, why=why,
                    )
                )
        return out

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
        for k in ("doc_id_range", "marker_token_ids"):
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
        Write the fingerprint into ``directory``.

        :param directory: Shard directory (at tokenize time) or checkpoint directory (at train time).

        :returns: The path written.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / FINGERPRINT_FILENAME
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def read(cls, directory: Path) -> Optional["FormatFingerprint"]:
        """
        Read a fingerprint from ``directory``.

        :param directory: Shard or checkpoint directory.

        :returns: The fingerprint, or ``None`` if the directory predates fingerprinting. Callers
            must decide what to do with ``None`` explicitly -- see
            :func:`check_or_explain_missing`.
        """
        path = Path(directory) / FINGERPRINT_FILENAME
        if not path.exists():
            return None
        return cls.from_dict(json.loads(path.read_text()))

    def evolve(self, **changes: Any) -> "FormatFingerprint":
        """
        :param changes: Field overrides.

        :returns: A copy with those fields replaced.
        """
        return replace(self, **changes)


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

    :returns: A warning string when the fingerprint is absent and ``strict`` is ``False``, else
        ``None``.

    :raises FormatMismatchError: On an incompatible fingerprint.
    :raises FileNotFoundError: When the fingerprint is absent and ``strict`` is ``True``.
    """
    trained = FormatFingerprint.read(ckpt_dir)
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
    eval_fp.require_compatible_with(trained)
    return None
