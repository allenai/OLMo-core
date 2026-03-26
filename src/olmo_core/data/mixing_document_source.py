# -*- coding: utf-8 -*-
"""
MixingDocumentSource
=====================

This module defines a simple wrapper around a collection of
``DocumentSource`` objects.  The training loop uses a mix of
documents from several underlying sources.  The mix itself does not
need to know the individual names of the documents it contains, but
callbacks such as ``DocumentLogger`` need to be able to query the
full list of document names that belong to the current mix.

The :class:`MixingDocumentSource` class holds an iterable of
``DocumentSource`` objects and exposes a read‑only :pyattr:`document_names`
property that returns the list of names of the underlying sources.
"""

from __future__ import annotations

from typing import Iterable, List, Protocol, runtime_checkable

@runtime_checkable
class DocumentSource(Protocol):
    """Protocol for a document source.

    Concrete document source implementations are expected to expose a
    ``name`` attribute that uniquely identifies the source.
    """
    name: str

__all__ = ["MixingDocumentSource"]


class MixingDocumentSource:
    """A read‑only wrapper around a collection of :class:`DocumentSource`.

    Parameters
    ----------
    sources:
        An iterable of :class:`DocumentSource` objects that together
        form the current mix.

    Attributes
    ----------
    document_names:
        A list of the names of all underlying document sources.  The
        property is read‑only; the underlying list of sources is
        stored privately and cannot be modified through the public
        API.
    """

    def __init__(self, sources: Iterable[DocumentSource]) -> None:
        self._sources: List[DocumentSource] = list(sources)

    @property
    def document_names(self) -> List[str]:
        """Return the list of names of the underlying document sources.

        The property is intentionally read‑only to prevent accidental
        mutation of the mix configuration during training.
        """
        return [source.name for source in self._sources]
    
    def __repr__(self) -> str:  # pragma: no cover - simple debugging helper
        return f"{self.__class__.__name__}({self.document_names!r})"
    
    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self._sources)
    
    def __iter__(self):  # pragma: no cover - convenience
        return iter(self._sources)
