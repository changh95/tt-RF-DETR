# SPDX-License-Identifier: Apache-2.0
"""Minimal pytest fixtures for tt-RF-DETR.

Provides a single-device ``device`` fixture so the rf_detr tests can run without
the tt-metal monorepo's own conftest.py. The device is selected with
``--device-id N`` on the pytest CLI or the ``RF_DETR_DEVICE`` env var (default 0).
Set ``TT_VISIBLE_DEVICES`` to constrain which PCIe chips ttnn enumerates at all.

The device is opened with the deployment params RF-DETR needs:
  * ``l1_small_size`` — scratch L1 the projector's ``ttnn.conv2d`` requires.
  * ``trace_region_size`` — DRAM reserved for the metal-trace of the
    projector+transformer tail (see ``TtRfDetr``).
  * ``num_command_queues`` — single CQ (2-CQ overlap was measured to regress).
"""

from __future__ import annotations

import gc
import os

import pytest

# Deployment params (not measurement logic) — see TtRfDetr / benchmark.py.
DEVICE_PARAMS = dict(
    l1_small_size=int(os.environ.get("RF_DETR_L1_SMALL", 32768)),
    trace_region_size=int(os.environ.get("RF_DETR_TRACE_REGION", 90_000_000)),
    num_command_queues=1,
)


def pytest_addoption(parser):
    parser.addoption(
        "--device-id",
        action="store",
        default=None,
        help="Blackhole chip id to open (overrides $RF_DETR_DEVICE; default 0).",
    )


@pytest.fixture(autouse=True)
def _gc_between_tests():
    gc.collect()


@pytest.fixture(scope="session")
def device_id(request):
    cli = request.config.getoption("--device-id")
    return int(cli) if cli is not None else int(os.environ.get("RF_DETR_DEVICE", "0"))


@pytest.fixture(scope="session")
def device(device_id):
    import ttnn

    dev = ttnn.open_device(device_id=device_id, **DEVICE_PARAMS)
    yield dev
    ttnn.close_device(dev)
