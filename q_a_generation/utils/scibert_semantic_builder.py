"""
SciBERT-based Semantic Relationship Builder

Uses SciBERT embeddings to capture semantic relationships between full abstracts.
Complements entity-based relationships (Jaccard) by finding conceptually related
papers even when they use different terminology.
"""

from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ragas.testset.transforms.base import RelationshipBuilder, KnowledgeGraph, Relationship


@dataclass
class SciBERTSemanticBuilder(RelationshipBuilder):
    """
    Build semantic relationships using SciBERT embeddings on full abstracts.

    Attributes
    ----------
    property_name : str
        The property containing text to embed (default: "page_content")
    new_property_name : str
        Property name for similarity score (default: "semantic_similarity")
    threshold : float
        Minimum cosine similarity to create edge (default: 0.7)
        - 0.9+ : Very similar papers (almost duplicates)
        - 0.7-0.9 : Semantically related (same topic/methods)
        - 0.5-0.7 : Loosely related
    model_name : str
        SciBERT model variant (default: allenai/scibert_scivocab_uncased)
    batch_size : int
        Batch size for encoding (default: 32)
    """

    property_name: str = "page_content"
    new_property_name: str = "semantic_similarity"
    threshold: float = 0.7
    model_name: str = "allenai/scibert_scivocab_uncased"
    batch_size: int = 32

    def __post_init__(self):
        """Lazy load the model only when needed."""
        self._model = None

    @property
    def model(self):
        """Lazy load SentenceTransformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[INFO] Loading SciBERT model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                print(f"[INFO] SciBERT model loaded successfully")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for SciBERT embeddings. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    async def transform(self, kg: KnowledgeGraph):
        """
        Generate semantic relationships based on SciBERT embeddings.

        Process:
        1. Extract full text from all nodes
        2. Generate SciBERT embeddings in batches
        3. Compute pairwise cosine similarity
        4. Create relationships for pairs above threshold

        Parameters
        ----------
        kg : KnowledgeGraph
            Knowledge graph with nodes containing text

        Returns
        -------
        List[Relationship]
            Semantic similarity relationships
        """
        # Extract text from all nodes
        texts = []
        valid_nodes = []

        for node in kg.nodes:
            text = node.get_property(self.property_name)
            if text and isinstance(text, str) and len(text.strip()) > 0:
                texts.append(text.strip())
                valid_nodes.append(node)
            else:
                print(f"[WARN] Node {node.id} has no valid text content, skipping")

        if len(valid_nodes) < 2:
            print(f"[WARN] Only {len(valid_nodes)} valid nodes found, cannot create relationships")
            return []

        print(f"[INFO] Generating SciBERT embeddings for {len(texts)} documents...")
        print(f"[INFO] Using model: {self.model_name}, batch_size: {self.batch_size}")

        # Generate embeddings in batches for efficiency
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print(f"[INFO] Generated embeddings with shape: {embeddings.shape}")

        # Compute pairwise cosine similarity
        print(f"[INFO] Computing pairwise cosine similarity...")
        similarity_matrix = cosine_similarity(embeddings)

        # Create relationships for pairs above threshold
        relationships = []
        edge_count = 0

        for i in range(len(valid_nodes)):
            for j in range(i + 1, len(valid_nodes)):
                sim_score = float(similarity_matrix[i, j])

                if sim_score >= self.threshold:
                    # Check if nodes are from different documents (cross-doc preferred)
                    src_doc = valid_nodes[i].properties.get("doc_source", "unknown")
                    tgt_doc = valid_nodes[j].properties.get("doc_source", "unknown")
                    is_cross_doc = src_doc != tgt_doc

                    relationships.append(
                        Relationship(
                            source=valid_nodes[i],
                            target=valid_nodes[j],
                            type="semantic_similarity",
                            properties={
                                self.new_property_name: sim_score,
                                "cross_doc": is_cross_doc,
                                "relationship_type": "scibert_semantic"
                            }
                        )
                    )
                    edge_count += 1

        # Statistics
        cross_doc_edges = sum(1 for r in relationships if r.properties.get("cross_doc", False))
        print(f"[INFO] Created {edge_count} semantic relationships (threshold={self.threshold})")
        print(f"[INFO] Cross-document edges: {cross_doc_edges}/{edge_count}")

        if edge_count > 0:
            avg_sim = np.mean([r.properties[self.new_property_name] for r in relationships])
            max_sim = max([r.properties[self.new_property_name] for r in relationships])
            min_sim = min([r.properties[self.new_property_name] for r in relationships])
            print(f"[INFO] Similarity stats - avg: {avg_sim:.3f}, min: {min_sim:.3f}, max: {max_sim:.3f}")

        return relationships
