import os
import pytest

if os.getenv('_PYTEST_RAISE', "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value

# Ensure pytest uses repo dir if running from above (so it uses the correct paths)
@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    repo_path = os.path.dirname(request.fspath.dirname)
    monkeypatch.chdir(repo_path)
