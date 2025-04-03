import requests
import logging

from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


class CohereReranker:
    """Wrapper class for Cohere reranking to match CrossEncoder interface"""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.url = "https://api.cohere.com/v2/rerank"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "accept": "application/json"
        }

    def predict(
        self, query: str, documents: list[str], top_n: int = None
    ) -> list[tuple[str, float]]:
        """
        Predict relevance scores for pairs of queries and documents.

        Args:
            query: Query string
            documents: List of document strings
        Returns:
            List of tuples (index, relevance_score)
            - index: index of the document in the original documents list
            - relevance_score: relevance score of the document
        """
        if not query or not documents:
            log.warning("No pairs provided to CohereReranker")
            return []

        try:
            data = {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": top_n if top_n else len(documents),
            }

            response = requests.post(self.url, headers=self.headers, json=data)
            response.raise_for_status()

            results = response.json()["results"]
            # Extract scores in same order as input documents
            return [
                (result["index"], result["relevance_score"]) 
                for result in results
            ]

        except Exception as e:
            log.error(f"Cohere reranking failed: {str(e)}")
            # Return neutral scores on error
            return [(i, 0.5) for i in range(len(documents))]
