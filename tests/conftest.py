"""Shared pytest fixtures for the pga-toolbox test suite.

Device parameterisation
------------------------
Every test runs on each device in ``DEVICES`` via the autouse, parametrised
:func:`device` fixture below. The fixture flips PyTorch's *default device* with
``torch.set_default_device`` so that the unmodified tensor constructions in the
tests (``torch.zeros(...)``, ``torch.randn(...)``, ...) land on the device under
test without threading ``device=`` through every call. The library itself is
device-agnostic (it inherits the device of its input tensors), so this exercises
the exact same code paths on CPU and CUDA.

The ``cuda`` parametrisation is skipped automatically when no CUDA device is
available, so the suite still runs (CPU-only) on machines and CI runners without
a GPU.

Note on generators: a ``torch.Generator`` is device-specific and a CPU generator
cannot seed a CUDA tensor. Tests that need a seeded generator must therefore
build it on the active default device, e.g.::

    gen = torch.Generator(device=torch.get_default_device()).manual_seed(0)
"""

from __future__ import annotations

import pytest
import torch

DEVICES = ["cpu", "cuda"]


@pytest.fixture(autouse=True, params=DEVICES)
def device(request):
    """Run each test once per device, switching the torch default device.

    Yields the device string ("cpu" / "cuda") for tests that want it
    explicitly; most tests rely on the default-device switch alone.
    """
    dev = request.param
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.set_default_device(dev)
    try:
        yield dev
    finally:
        torch.set_default_device("cpu")
