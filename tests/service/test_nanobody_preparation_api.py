"""Local preview is authenticated, bounded and never admits scientific work."""

import asyncio

from test_api_contract import ORIGIN, _app, _humanization_session, _request

from biomodals.service.nanobody_humanization.router import create_router

VHH = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYWMYWVRQAPGKGLEWVSEINTNGLITKYPDSVKGRFTISRDNAKNTLYLQMNSLRPEDTAVYYCARSPSGFNRGQGTLVTVSS"
ROOT = "/api/v1/nanobody-humanization"


def test_native_preview_rows_digest_and_authentication(tmp_path):
    """An invalid sibling keeps its row; only a wholly valid review can submit."""
    app = _app(tmp_path)
    app.include_router(create_router())
    try:
        assert _request(app, "GET", ROOT + "/options").status_code == 401
        assert (
            _request(
                app,
                "POST",
                ROOT + "/prepare",
                headers={"Origin": ORIGIN},
                json={"parents": [{"id": "one", "vhh": VHH}]},
            ).status_code
            == 401
        )
        _humanization_session(app)
        options = _request(app, "GET", ROOT + "/options").json()
        assert options["max_parents"] == 100
        assert options["max_input_length"] == 512
        assert options["max_csv_bytes"] == 10 * 1024 * 1024
        assert options["defaults"]["hudiff_nb_candidate_count"] == 10
        assert set(options["defaults"]) == set(options["settings_schema"]["properties"])
        originals = [
            {"id": "valid", "vhh": "HHHHHH" + VHH + "GGG"},
            {"id": "invalid", "vhh": "ACDE"},
            {"id": "truncated", "vhh": VHH[5:-3]},
        ]
        response = _request(app, "POST", ROOT + "/prepare", json={"parents": originals})
        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "private, no-store"
        result = response.json()
        assert result["rows"][0] == {"row_index": 0, "id": "valid", "vh": VHH}
        assert result["rows"][1] == {"row_index": 1, "id": "invalid", "vh": None}
        assert len(result["rows"][2]["vh"]) > len(originals[2]["vhh"])
        assert result["errors"][0]["row_index"] == 1
        assert result["preparation_digest"] is None
        digests = []
        for entries in ([originals[0]], [originals[0]], [{"id": "valid", "vhh": VHH}]):
            valid = _request(
                app, "POST", ROOT + "/prepare", json={"parents": entries}
            ).json()
            assert valid["errors"] == []
            assert len(valid["preparation_digest"]) == 64
            digests.append(valid["preparation_digest"])
        assert digests[0] == digests[1] != digests[2]
        assert _request(app, "GET", "/api/v1/jobs").json()["jobs"] == []
    finally:
        asyncio.run(app.state.antibody_analysis.shutdown())


def test_server_parent_bound_and_duplicate_id_preview(tmp_path):
    """Enforce the configured bound before numbering; retain both duplicate rows."""
    app = _app(tmp_path)
    app.include_router(create_router(max_parents=2))
    _humanization_session(app)
    try:
        response = _request(
            app,
            "POST",
            ROOT + "/prepare",
            json={"parents": [{"id": key, "vhh": VHH} for key in ("a", "b", "c")]},
        )
        assert response.status_code == 422
        assert response.json()["code"] == "batch_too_large"
        response = _request(
            app,
            "POST",
            ROOT + "/prepare",
            json={"parents": [{"id": key, "vhh": VHH} for key in ("a b", "a_b")]},
        ).json()
        assert [row["vh"] for row in response["rows"]] == [VHH, VHH]
        assert [error["row_index"] for error in response["errors"]] == [0, 1]
        assert response["preparation_digest"] is None
    finally:
        asyncio.run(app.state.antibody_analysis.shutdown())
