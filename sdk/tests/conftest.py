from __future__ import annotations

import atexit
import os
import shutil
import tempfile

import pytest

# Set before test modules import the server's stateful singletons. Running
# sdk/tests alone does not load tests/conftest.py, so it needs its own sandbox.
for _variable in ("MAKERMODSLAB_HOME", "MAKERMODSLAB_OUTPUT_ROOT", "HF_LEROBOT_HOME", "HF_HOME"):
    _root = tempfile.mkdtemp(prefix="makermodslab-sdk-tests-")
    os.environ[_variable] = _root
    atexit.register(shutil.rmtree, _root, ignore_errors=True)
os.environ["HF_HUB_OFFLINE"] = "1"


@pytest.fixture(scope="session")
def app():
    """The real FastAPI app. Importing makermodslab.server is side-effect-safe
    (the OpenAPI export script and the contract tests already rely on that)."""
    from makermodslab.server import app as fastapi_app

    return fastapi_app


@pytest.fixture()
def sdk_client(app):
    """The SDK wired to the real app in-process: fastapi.testclient.TestClient
    IS an httpx.Client, injected as the SDK's http_client — full-stack
    contract tests with zero network and zero server process."""
    from fastapi.testclient import TestClient
    from makermodslab_sdk import Client

    client = Client("http://testserver", http_client=TestClient(app), check_compatibility=False)
    yield client
    client.close()
