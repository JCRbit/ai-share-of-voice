import urllib.robotparser
import urllib.parse
import requests
from bs4 import BeautifulSoup
from typing import List, Dict

class MCPSourcesServer:
    def __init__(self):
        self.user_agent = "CloudistaSoVBot/1.0"
        self._robot_parsers: Dict[str, urllib.robotparser.RobotFileParser] = {}

    def check_url(self, url: str) -> bool:
        """Parses robots.txt and verifies if the URL is allowed to be scraped."""
        parsed_url = urllib.parse.urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        if base_url not in self._robot_parsers:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{base_url}/robots.txt")
            try:
                rp.read()
            except Exception:
                pass # Default to allowed if robots.txt is unreachable
            self._robot_parsers[base_url] = rp
            
        return self._robot_parsers[base_url].can_fetch(self.user_agent, url)

    def read_source(self, url: str) -> str:
        """Fetches page content, validates robots.txt, and returns untrusted text."""
        if not self.check_url(url):
            raise PermissionError(f"Scraping denied by robots.txt for {url}")

        try:
            response = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=10)
            response.raise_for_status()
            
            # Extract text and strip potentially malicious HTML/scripts
            soup = BeautifulSoup(response.text, "html.parser")
            for script in soup(["script", "style", "iframe", "object", "embed"]):
                script.extract()
                
            raw_text = soup.get_text(separator=" ", strip=True)
            
            # Explicitly mark content as untrusted data for downstream LLM parsing
            return f"<untrusted_data source='{url}'>\n{raw_text}\n</untrusted_data>"
            
        except requests.RequestException as e:
            return f"<error>Failed to fetch source: {str(e)}</error>"

    def search_sources(self, query: str, limit: int = 5) -> List[str]:
        """Searches for relevant sources (mocked for initial implementation)."""
        # TODO: Integrate with actual search API (e.g., Tavily, DuckDuckGo, or Google Custom Search)
        return [
            f"https://example.com/search?q={urllib.parse.quote(query)}",
            f"https://news.example.com/article?q={urllib.parse.quote(query)}"
        ]
