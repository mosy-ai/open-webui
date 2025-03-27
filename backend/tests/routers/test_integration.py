import sys
import os
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI, HTTPException

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
backend_path = os.path.join(project_root, 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Now you can import modules from the backend package.
from open_webui.utils.auth import get_admin_user
from open_webui.models.social_integrations import Integrations
from open_webui.routers.v2.social_integrations import router

# Create a FastAPI app and include the router.
app = FastAPI()
app.include_router(router)

# Override the get_admin_user dependency to always return a dummy admin user.
class DummyUser:
    id = "dummy_admin_id"

def dummy_get_admin_user():
    return DummyUser()

app.dependency_overrides[get_admin_user] = dummy_get_admin_user

@pytest.fixture
def client():
    return TestClient(app)

# Test the POST /integrations/ endpoint for success.
def test_create_integration_success(monkeypatch, client):
    # Fake the insert_new_integration function to return a dummy integration.
    def fake_insert_new_integration(user_id, integration_form):
        # Assume integration_form has a 'config' attribute
        return {
            "id": "integration_1",
            "user_id": user_id,
            "config": getattr(integration_form, "config", {})
        }
    monkeypatch.setattr(Integrations, "insert_new_integration", fake_insert_new_integration)

    payload = {"config": {"key": "value"}}
    response = client.post("/integrations/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "integration_1"
    assert data["user_id"] == "dummy_admin_id"
    assert data["config"] == {"key": "value"}

# Test the POST /integrations/ endpoint when creation fails.
def test_create_integration_failure(monkeypatch, client):
    def fake_insert_new_integration(user_id, integration_form):
        return None  # Simulate failure
    monkeypatch.setattr(Integrations, "insert_new_integration", fake_insert_new_integration)

    payload = {"config": {"key": "value"}}
    response = client.post("/integrations/", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Failed to create integration"

# Test the GET /integrations/{integration_id} endpoint when the integration is found.
def test_read_integration_found(monkeypatch, client):
    def fake_get_integration_by_id(integration_id):
        return {"id": integration_id, "user_id": "dummy_admin_id", "config": {}}
    monkeypatch.setattr(Integrations, "get_integration_by_id", fake_get_integration_by_id)

    response = client.get("/integrations/integration_1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "integration_1"

# Test the GET endpoint when the integration is not found.
def test_read_integration_not_found(monkeypatch, client):
    def fake_get_integration_by_id(integration_id):
        return None  # Not found
    monkeypatch.setattr(Integrations, "get_integration_by_id", fake_get_integration_by_id)

    response = client.get("/integrations/nonexistent")
    assert response.status_code == 404
    data = response.json()
    assert data["detail"] == "Integration not found"

# Test the PATCH /integrations/{integration_id}/config endpoint for a successful config update.
def test_update_integration_config_success(monkeypatch, client):
    def fake_update_integration_config_by_id(integration_id, config):
        return {"id": integration_id, "user_id": "dummy_admin_id", "config": config}
    monkeypatch.setattr(Integrations, "update_integration_config_by_id", fake_update_integration_config_by_id)

    new_config = {"new_config": "updated_value"}
    response = client.patch("/integrations/integration_1/config", json=new_config)
    assert response.status_code == 200
    data = response.json()
    assert data["config"] == new_config

# Test the PATCH endpoint when the config update fails.
def test_update_integration_config_failure(monkeypatch, client):
    def fake_update_integration_config_by_id(integration_id, config):
        return None  # Simulate failure
    monkeypatch.setattr(Integrations, "update_integration_config_by_id", fake_update_integration_config_by_id)

    new_config = {"new_config": "updated_value"}
    response = client.patch("/integrations/integration_1/config", json=new_config)
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Failed to update configuration"

# Test the webhook endpoint for a successful update (both config and status).
def test_webhook_update_integration_status_success(monkeypatch, client):
    # Simulate that the integration exists.
    def fake_get_integration_by_id(integration_id):
        return {"id": integration_id, "user_id": "dummy_admin_id", "config": {}}
    monkeypatch.setattr(Integrations, "get_integration_by_id", fake_get_integration_by_id)

    def fake_update_integration_config_by_id(integration_id, config):
        return {"id": integration_id, "user_id": "dummy_admin_id", "config": config}
    monkeypatch.setattr(Integrations, "update_integration_config_by_id", fake_update_integration_config_by_id)

    def fake_update_integration_status_by_id(integration_id, status):
        return {"id": integration_id, "user_id": "dummy_admin_id", "status": status, "config": {}}
    monkeypatch.setattr(Integrations, "update_integration_status_by_id", fake_update_integration_status_by_id)

    payload = {"status": "success", "config": {"key": "value"}}
    response = client.post("/integrations/webhook/integration_1", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

# Test the webhook endpoint when the config update fails.
def test_webhook_update_integration_status_failure_config(monkeypatch, client):
    def fake_get_integration_by_id(integration_id):
        return {"id": integration_id, "user_id": "dummy_admin_id", "config": {}}
    monkeypatch.setattr(Integrations, "get_integration_by_id", fake_get_integration_by_id)

    def fake_update_integration_config_by_id(integration_id, config):
        return None  # Simulate failure on config update
    monkeypatch.setattr(Integrations, "update_integration_config_by_id", fake_update_integration_config_by_id)

    payload = {"status": "success", "config": {"key": "value"}}
    response = client.post("/integrations/webhook/integration_1", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Failed to update configuration"

# Test the webhook endpoint when the status update fails.
def test_webhook_update_integration_status_failure_status(monkeypatch, client):
    def fake_get_integration_by_id(integration_id):
        return {"id": integration_id, "user_id": "dummy_admin_id", "config": {}}
    monkeypatch.setattr(Integrations, "get_integration_by_id", fake_get_integration_by_id)

    def fake_update_integration_config_by_id(integration_id, config):
        return {"id": integration_id, "user_id": "dummy_admin_id", "config": config}
    monkeypatch.setattr(Integrations, "update_integration_config_by_id", fake_update_integration_config_by_id)

    def fake_update_integration_status_by_id(integration_id, status):
        return None  # Simulate failure on status update
    monkeypatch.setattr(Integrations, "update_integration_status_by_id", fake_update_integration_status_by_id)

    payload = {"status": "failed", "config": {"key": "value"}}
    response = client.post("/integrations/webhook/integration_1", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Failed to update integration status"

# Test the DELETE /integrations/{integration_id} endpoint for success.
def test_delete_integration_success(monkeypatch, client):
    def fake_delete_integration_by_id(integration_id):
        return True
    monkeypatch.setattr(Integrations, "delete_integration_by_id", fake_delete_integration_by_id)

    response = client.delete("/integrations/integration_1")
    assert response.status_code == 200
    data = response.json()
    assert data["detail"] == "Integration deleted successfully"

# Test the DELETE endpoint when deletion fails.
def test_delete_integration_failure(monkeypatch, client):
    def fake_delete_integration_by_id(integration_id):
        return False  # Simulate failure
    monkeypatch.setattr(Integrations, "delete_integration_by_id", fake_delete_integration_by_id)

    response = client.delete("/integrations/integration_1")
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Failed to delete integration"
