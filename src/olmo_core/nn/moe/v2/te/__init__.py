"""
Experimental CPU activation-offloading utilities for the fused MoE-v2 stack.

.. warning::
    This is an experimental prototype and is not currently wired into any model. CPU activation
    offloading trades GPU memory for host<->device copies; it is only useful when PCIe bandwidth is
    not the bottleneck. See :func:`get_cpu_offload_context`.
"""

from .cpu_offload import get_cpu_offload_context

__all__ = ["get_cpu_offload_context"]
