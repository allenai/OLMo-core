"""
The shared contract: what data generation and evaluation must agree on.

Prompt templates, document serialization, chunk layout, the task registry, answer parsers, and
metrics. Pure Python -- no torch, no transformers, no olmo_core -- so every other module can depend
on it freely. One definition each, read by both halves of the pipeline.
"""
