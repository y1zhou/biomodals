"""HTTP abandonment cancels native preflight without retaining a validation."""

# ruff: noqa: D103

import asyncio
import hashlib
from types import SimpleNamespace
from urllib.parse import urlsplit

import orjson
import pytest
from test_api_contract import ORIGIN, _app, _humanization_session

from biomodals.app.fold.alphafold3.chemistry import ChemistryReceipt
from biomodals.app.fold.alphafold3.profiles import ALPHAFOLD3_COMMIT
from biomodals.service.alphafold3 import modal as af3_modal
from biomodals.service.alphafold3.validation import ValidatedInputStore


@pytest.mark.parametrize("during_publication", [False, True])
def test_http_disconnect_cleans_up_and_releases_upload_slots(
    tmp_path, monkeypatch, during_publication
):
    async def run():
        app = _app(tmp_path)
        _humanization_session(app)
        cancellations = []
        publish = ValidatedInputStore.publish
        loop = asyncio.get_running_loop()
        body = orjson.dumps({
            "name": "ptm",
            "modelSeeds": [1],
            "dialect": "alphafold3",
            "version": 3,
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": "AST",
                        "modifications": [{"ptmType": "SEP", "ptmPosition": 2}],
                    }
                }
            ],
        })
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "https",
            "path": "/api/v1/alphafold3/validations",
            "raw_path": b"/api/v1/alphafold3/validations",
            "root_path": "",
            "query_string": b"",
            "server": (urlsplit(ORIGIN).hostname, 443),
            "client": ("127.0.0.1", 1234),
            "headers": [
                (b"host", urlsplit(ORIGIN).netloc.encode()),
                (b"origin", ORIGIN.encode()),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }

        async def attempt(index):
            started = asyncio.Event()
            incoming = asyncio.Queue()
            responses = []
            await incoming.put({
                "type": "http.request",
                "body": body,
                "more_body": False,
            })

            async def hydrate():
                return None

            async def cancel():
                cancellations.append(index)

            async def spawn(content):
                async def get():
                    started.set()
                    if during_publication:
                        return {
                            "receipt": ChemistryReceipt(
                                input_sha256=hashlib.sha256(content).hexdigest(),
                                ccd_sha256="a" * 64,
                                upstream_commit=ALPHAFOLD3_COMMIT,
                            ).model_dump()
                        }
                    await asyncio.Event().wait()

                return SimpleNamespace(
                    get=SimpleNamespace(aio=get), cancel=SimpleNamespace(aio=cancel)
                )

            def disconnect_during_publish(self, *args, **kwargs):
                loop.call_soon_threadsafe(
                    incoming.put_nowait, {"type": "http.disconnect"}
                )
                return publish(self, *args, **kwargs)

            if during_publication:
                monkeypatch.setattr(
                    ValidatedInputStore, "publish", disconnect_during_publish
                )

            function = SimpleNamespace(
                hydrate=SimpleNamespace(aio=hydrate), spawn=SimpleNamespace(aio=spawn)
            )
            monkeypatch.setattr(
                af3_modal.modal.Function, "from_name", lambda *a, **kw: function
            )

            async def send(message):
                responses.append(message)

            task = asyncio.create_task(app(scope.copy(), incoming.get, send))
            await asyncio.wait_for(started.wait(), 2)
            if not during_publication:
                await incoming.put({"type": "http.disconnect"})
            await asyncio.wait_for(task, 2)
            assert [
                m["status"] for m in responses if m["type"] == "http.response.start"
            ] == [499]
            assert cancellations == (
                [] if during_publication else list(range(index + 1))
            )
            assert ValidatedInputStore(tmp_path / "validations").usage()[0] == 0
            assert not list(tmp_path.glob(".alphafold3-upload-*"))

        # Three requests exceed the two upload slots if either abort leaks one.
        for index in range(3):
            await attempt(index)

    asyncio.run(run())
