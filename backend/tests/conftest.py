import os
import pytest
from unittest.mock import MagicMock

# Set testing environment variable
os.environ["TESTING"] = "1"

@pytest.fixture(autouse=True)
def mock_db():
    """Mock database connections for all tests"""
    # Mock the database connection
    mock_db = MagicMock()
    mock_db.is_closed.return_value = True
    
    # Mock the database session
    mock_session = MagicMock()
    mock_session.close.return_value = None
    
    # Mock the database engine
    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_session
    
    # Mock the database models
    mock_base = MagicMock()
    mock_base.metadata = MagicMock()
    
    # Mock the JSONField
    mock_json_field = MagicMock()
    
    # Mock the get_db context manager
    mock_get_db = MagicMock()
    mock_get_db.return_value = mock_session
    
    # Apply mocks
    import open_webui.internal.db as db
    db.Base = mock_base
    db.JSONField = mock_json_field
    db.get_db = mock_get_db
    db.SessionLocal = MagicMock(return_value=mock_session)
    db.engine = mock_engine
    
    yield
    
    # Cleanup
    del os.environ["TESTING"]
