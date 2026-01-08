import arxiv
import yaml
from datetime import datetime, timedelta, timezone

class ArxivSearchAgent:
    def __init__(self, config_path: str = "scholar_bridge/config/model_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["arxiv_settings"]

    def search_papers(self, query: str) -> list:
        """
        Searches ArXiv for papers based on the query.
        Filters for papers within the configured search_period_days.
        """
        print(f"Searching ArXiv for: {query}...")
        
        # Construct client
        client = arxiv.Client()
        
        # Build search
        search = arxiv.Search(
            query = query,
            max_results = self.config["max_results"] * 2, # Fetch extra to allow for date filtering
            sort_by = getattr(arxiv.SortCriterion, self.config["sort_by"][0].upper() + self.config["sort_by"][1:]),
            sort_order = getattr(arxiv.SortOrder, self.config["sort_order"].upper()[0] + self.config["sort_order"].lower()[1:])
        )

        results = []
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config["search_period_days"])
        
        try:
            for r in client.results(search):
                if r.published < cutoff_date:
                    continue
                    
                paper_data = {
                    "title": r.title,
                    "abstract": r.summary.replace("\n", " "),
                    "published": r.published.strftime("%Y-%m-%d"),
                    "url": r.pdf_url,
                    "authors": [a.name for a in r.authors]
                }
                results.append(paper_data)
                
                if len(results) >= self.config["max_results"]:
                    break
                    
            print(f"Found {len(results)} relevant papers.")
            return results
            
        except Exception as e:
            print(f"Error searching ArXiv: {e}")
            return []
