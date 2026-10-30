import pytest
from unittest.mock import patch
from ai_share_of_voice.mcp_sources import MCPSourcesServer

@pytest.fixture
def mcp_server():
    return MCPSourcesServer()

def test_check_url_respects_robots_txt(mcp_server):
    with patch("urllib.robotparser.RobotFileParser.can_fetch", return_value=False):
        assert mcp_server.check_url("http://example.com/private") is False

def test_read_source_strips_malicious_content(mcp_server):
    malicious_html = """
    <html>
        <body>
            <h1>Valid Content</h1>
            <script>alert('XSS Attack');</script>
            <iframe src="http://malicious.com"></iframe>
        </body>
    </html>
    """
    
    with patch("ai_share_of_voice.mcp_sources.MCPSourcesServer.check_url", return_value=True):
        with patch("requests.get") as mock_get:
            mock_get.return_value.text = malicious_html
            mock_get.return_value.raise_for_status = lambda: None
            
            result = mcp_server.read_source("http://example.com")
            
            assert "Valid Content" in result
            assert "<script>" not in result
            assert "XSS Attack" not in result
            assert "untrusted_data" in result
