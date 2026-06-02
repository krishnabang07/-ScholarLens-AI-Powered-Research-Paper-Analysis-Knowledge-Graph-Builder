"""
graph/builder.py — builds and maintains the knowledge graph using NetworkX.

Graph model:
  Nodes: concepts (methods, datasets, metrics, tasks), papers, authors
  Edges: co-occurrence (concept-concept), authorship, citation

Persistence: saved as JSON (nodes + edge list) in data/graph.json
"""

import json
import math
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import networkx as nx
from community import community_louvain  # python-louvain

from src.pipeline.entity_extractor import ExtractionResult
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

GRAPH_PATH = Path("data/graph.json")


class KnowledgeGraphBuilder:
    """
    Incremental knowledge graph: call `add_paper()` each time a new
    paper is ingested; the graph is updated in-place and saved.
    """

    def __init__(self):
        self.G = nx.Graph()
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        if GRAPH_PATH.exists():
            data = json.loads(GRAPH_PATH.read_text())
            self.G = nx.node_link_graph(data)
            logger.info("Loaded graph: %d nodes, %d edges", self.G.number_of_nodes(), self.G.number_of_edges())
        else:
            logger.info("Starting fresh knowledge graph")

    def save(self):
        GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.G)
        GRAPH_PATH.write_text(json.dumps(data))

    # ── Graph construction ────────────────────────────────────────────────────

    def add_paper(
        self,
        paper_id: str,
        title: str,
        authors: list[str],
        year: int,
        extraction: ExtractionResult,
    ) -> int:
        """
        Add a paper and its extracted entities to the graph.
        Returns number of NEW nodes added.
        """
        nodes_before = self.G.number_of_nodes()

        # Paper node
        self.G.add_node(
            paper_id,
            label=title[:80],
            type="paper",
            year=year,
            weight=1.0,
        )

        # Author nodes + paper-author edges
        for author in authors:
            auth_id = f"author::{author.lower().replace(' ', '_')}"
            if not self.G.has_node(auth_id):
                self.G.add_node(auth_id, label=author, type="author", weight=1.0, papers=[])
            self.G.nodes[auth_id]["papers"] = self.G.nodes[auth_id].get("papers", []) + [paper_id]
            self.G.nodes[auth_id]["weight"] = len(self.G.nodes[auth_id]["papers"])
            self.G.add_edge(paper_id, auth_id, edge_type="authored_by", weight=1.0)

        # Concept nodes + paper-concept edges
        all_concepts = []
        for entity in extraction.entities:
            concept_id = f"{entity.label.lower()}::{entity.normalized}"
            if not self.G.has_node(concept_id):
                self.G.add_node(
                    concept_id,
                    label=entity.text,
                    type=entity.label.lower(),
                    weight=0,
                    papers=[],
                )
            node = self.G.nodes[concept_id]
            node["papers"] = list(set(node.get("papers", []) + [paper_id]))
            node["weight"] = len(node["papers"])
            self.G.add_edge(paper_id, concept_id, edge_type="mentions", weight=entity.confidence)
            all_concepts.append(concept_id)

        # Concept-concept co-occurrence edges (within same paper)
        for c1, c2 in combinations(all_concepts, 2):
            if self.G.has_edge(c1, c2):
                self.G[c1][c2]["weight"] += 1
            else:
                self.G.add_edge(c1, c2, edge_type="co-occurs", weight=1)

        nodes_added = self.G.number_of_nodes() - nodes_before
        self.save()
        logger.info("Added paper '%s': +%d nodes", title[:40], nodes_added)
        return nodes_added

    # ── Querying ──────────────────────────────────────────────────────────────

    def subgraph_around(self, concept_query: str, depth: int = 2) -> dict:
        """
        Return a subgraph centered on nodes matching concept_query.
        BFS up to `depth` hops.
        """
        # Find seed nodes
        q = concept_query.lower()
        seeds = [
            n for n in self.G.nodes
            if q in self.G.nodes[n].get("label", "").lower()
            or q in n.lower()
        ]
        if not seeds:
            return {"nodes": [], "edges": [], "metadata": {"found": False}}

        # BFS
        visited = set(seeds)
        frontier = set(seeds)
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.G.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier

        sub = self.G.subgraph(visited)
        return self._serialize_graph(sub)

    def top_concepts(self, n: int = 50, concept_type: str = None) -> list[dict]:
        """Return top-N nodes by weight (frequency)."""
        nodes = [
            (nid, data) for nid, data in self.G.nodes(data=True)
            if data.get("type") != "paper"
            and (concept_type is None or data.get("type") == concept_type)
        ]
        nodes.sort(key=lambda x: x[1].get("weight", 0), reverse=True)
        return [
            {"id": nid, **data}
            for nid, data in nodes[:n]
        ]

    def detect_communities(self) -> dict[str, int]:
        """Run Louvain community detection. Returns node → community_id."""
        # Only run on concept subgraph (exclude paper/author nodes)
        concept_nodes = [
            n for n, d in self.G.nodes(data=True)
            if d.get("type") not in ("paper", "author")
        ]
        sub = self.G.subgraph(concept_nodes)
        partition = community_louvain.best_partition(sub)
        return partition

    # ── Serialization ─────────────────────────────────────────────────────────

    def _serialize_graph(self, G: nx.Graph) -> dict:
        nodes = []
        for nid, data in G.nodes(data=True):
            nodes.append({
                "id": nid,
                "label": data.get("label", nid),
                "type": data.get("type", "concept"),
                "weight": data.get("weight", 1),
                "papers": data.get("papers", []),
            })

        edges = []
        for u, v, data in G.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "weight": data.get("weight", 1),
                "edge_type": data.get("edge_type", "co-occurs"),
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
            },
        }

    def export_full(self) -> dict:
        return self._serialize_graph(self.G)
