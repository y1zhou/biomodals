"""CSV-first review without scientific execution or persisted model state."""

import asyncio

from test_api_contract import ORIGIN, _app, _humanization_session, _request

ROOT = "/api/v1/protein-optimization"
CSV = "id,mutations,label\nparent,,0\none,A:A1C,1\ntwo,B:C1V,2\n"


def test_authenticated_discovery_preserves_bad_rows_then_validates_parent(tmp_path):
    """Required chain IDs come from the CSV, with labels left visible for repair."""
    app = _app(tmp_path)
    try:
        assert _request(app, "GET", ROOT + "/options").status_code == 401
        assert (
            _request(
                app,
                "POST",
                ROOT + "/review",
                headers={"Origin": ORIGIN},
                json={"measurements_csv": CSV},
            ).status_code
            == 401
        )
        _humanization_session(app)
        options = _request(app, "GET", ROOT + "/options").json()
        assert options["defaults"]["combination"]["candidate_budget"] == 1_000_000
        assert options["defaults"]["exploration"]["candidate_budget"] == 5000
        assert options["max_exploration_chain_length"] == 2046
        assert (
            options["settings_schema"]["$defs"]["PositionChoices"]["properties"][
                "amino_acids"
            ]["default"]
            == "ADEFGHIKLNPQRSTVWY"
        )
        response = _request(
            app,
            "POST",
            ROOT + "/review",
            json={"measurements_csv": CSV.replace(",2", ",>1000")},
        )
        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "private, no-store"
        result = response.json()
        assert result["required_chain_ids"] == ["A", "B"]
        assert result["rows"][2]["label"] == ">1000"
        assert result["errors"][0]["row_index"] == 2
        assert result["review_digest"] is None
        body = {"measurements_csv": CSV, "parental_fasta": ">A\nAA\n>B\nC\n"}
        good = _request(app, "POST", ROOT + "/review", json=body).json()
        assert good["errors"] == []
        assert good["evaluation_count"] == 1
        assert good["candidate_space_size"] == "1"
        assert good["unique_variant_count"] == 3
        assert good["replicate_rows"] == 0
        assert good["chains"] == [
            {"chain_id": "A", "sequence": "AA"},
            {"chain_id": "B", "sequence": "C"},
        ]
        assert len(good["review_digest"]) == 64
        assert (
            _request(app, "POST", ROOT + "/review", json=body).json()["review_digest"]
            == good["review_digest"]
        )
        body["settings"] = {"direction": "minimize"}
        assert (
            _request(app, "POST", ROOT + "/review", json=body).json()["review_digest"]
            != good["review_digest"]
        )
        assert _request(app, "GET", "/api/v1/jobs").json()["jobs"] == []
    finally:
        asyncio.run(app.state.antibody_analysis.shutdown())


def test_exploration_review_counts_novelty_masks_and_reports_mode_limit(tmp_path):
    """Exploration settings affect proposals, never silently alter training labels."""
    app = _app(tmp_path)
    try:
        _humanization_session(app)
        body = {
            "measurements_csv": CSV,
            "parental_fasta": ">A\nAA\n>B\nC\n",
            "settings": {
                "mode": "exploration",
                "candidate_budget": 3,
                "max_mutations": 2,
            },
        }
        result = _request(app, "POST", ROOT + "/review", json=body).json()
        assert result["errors"] == []
        assert result["evaluation_count"] == 3
        assert int(result["candidate_space_size"]) > 3
        assert [row["amino_acids"] for row in result["positions"]] == [
            "ADEFGHIKLNPQRSTVWY"
        ] * 2
        assert result["unique_variant_count"] == 3
        body["settings"]["candidate_budget"] = 1_000_000
        invalid = _request(app, "POST", ROOT + "/review", json=body).json()
        assert invalid["errors"][0]["code"] == "invalid_design_space"
        assert invalid["review_digest"] is None
        body["parental_fasta"] = ">A\nGA\n>B\nC\n"
        mismatch = _request(app, "POST", ROOT + "/review", json=body).json()
        assert any(issue["code"] == "original_mismatch" for issue in mismatch["errors"])
    finally:
        asyncio.run(app.state.antibody_analysis.shutdown())


def test_invalid_csv_is_known_rejection_and_parentless_only_rows_need_fasta(tmp_path):
    """Bad framing uses a coded error; discovery does not invent parental chains."""
    app = _app(tmp_path)
    try:
        _humanization_session(app)
        invalid = _request(
            app,
            "POST",
            ROOT + "/review",
            json={"measurements_csv": "wrong,columns\n1,2\n"},
        )
        assert invalid.status_code == 422
        assert invalid.json()["code"] == "invalid_measurements"
        result = _request(
            app,
            "POST",
            ROOT + "/review",
            json={"measurements_csv": "mutations,label\n,1\n"},
        ).json()
        assert result["required_chain_ids"] == []
        assert result["review_digest"] is None
        assert result["rows"][0]["canonical_mutations"] == ""
    finally:
        asyncio.run(app.state.antibody_analysis.shutdown())


def test_rejected_large_space_uses_exact_string_not_unsafe_json_integer(tmp_path):
    """A valid input can describe more combinations than a browser can count."""
    app = _app(tmp_path)
    try:
        _humanization_session(app)
        positions = 60
        measurements = "mutations,label\n" + "".join(
            f"A:A{position}V,{position}\n" for position in range(1, positions + 1)
        )
        result = _request(
            app,
            "POST",
            ROOT + "/review",
            json={
                "measurements_csv": measurements,
                "parental_fasta": ">A\n" + "A" * positions,
                "settings": {"max_mutations": positions},
            },
        ).json()
        assert result["candidate_space_size"] == str(2**positions - positions - 1)
        assert result["evaluation_count"] is None
        assert result["errors"][0]["code"] == "invalid_design_space"
        assert result["review_digest"] is None
    finally:
        asyncio.run(app.state.antibody_analysis.shutdown())
