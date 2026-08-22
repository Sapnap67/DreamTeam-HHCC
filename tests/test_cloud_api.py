from app import app


def test_health_endpoint_reports_model_and_upload_capabilities():
    response = app.test_client().get("/api/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["service"] == "BlindSpot Guardian"
    assert isinstance(payload["busy"], bool)
    assert payload["max_upload_bytes"] > 0

