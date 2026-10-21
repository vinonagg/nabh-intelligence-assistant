import os
from neo4j import GraphDatabase

class Neo4jGraphStore:
    """Optional graph retrieval layer. Requires Passage nodes in a populated Neo4j database."""
    def __init__(self, uri, username, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self.driver.verify_connectivity()

    @classmethod
    def from_env(cls):
        def setting(key, default=""):
            try:
                value = os.getenv(key, "")
                if not value:
                    import streamlit as st
                    value = st.secrets.get(key, "")
                return value or default
            except Exception:
                return os.getenv(key, default)
        uri = setting("NEO4J_URI")
        username = setting("NEO4J_USERNAME")
        password = setting("NEO4J_PASSWORD")
        database = setting("NEO4J_DATABASE", "neo4j")
        if not all([uri, username, password]):
            raise ValueError("Configure NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in .env or Streamlit Secrets.")
        return cls(uri, username, password, database)

    def search(self, question, limit=4):
        query = """
        MATCH (p:Passage)
        WHERE toLower(p.text) CONTAINS toLower($q)
           OR any(term IN $terms WHERE toLower(p.text) CONTAINS term)
        RETURN p.text AS text, p.page_start AS page_start, p.page_end AS page_end, p.chunk_id AS chunk_id
        LIMIT $limit
        """
        terms = [term.lower() for term in question.split() if len(term) > 3][:8]
        with self.driver.session(database=self.database) as session:
            rows = session.run(query, q=question, terms=terms, limit=limit)
            return [{
                "text": row["text"], "page_start": row["page_start"] or 0,
                "page_end": row["page_end"] or row["page_start"] or 0,
                "chunk_id": row["chunk_id"] or "graph",
                "label": f"Knowledge graph passage page {row['page_start']}", "score": 0.0,
            } for row in rows]

    def close(self):
        self.driver.close()
