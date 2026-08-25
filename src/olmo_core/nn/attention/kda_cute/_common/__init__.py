"""Machinery every op family in this package shares.

Deliberately small, and deliberately not a place for op logic: the call cache, the support
predicate, the fla compatibility probe, and once-per-process logging. Those four are what
the kda / gdn / gnorm chains all needed independently — and each had its own copy in the
research tree, which is how one leak fix ended up written for gnorm and missing from the
other two for two weeks.

Nothing here imports torch or cutlass at module scope beyond what it must, so importing
`kernel_fun` on a machine without a GPU stays cheap.
"""
