"""
Single-Hop RAGAS KG Testset Generator (2025 API)
"""

from __future__ import annotations
import argparse, asyncio, random
from pathlib import Path
from dataclasses import dataclass
import pandas as pd

from ragas.testset.graph import Node, KnowledgeGraph, NodeType
from ragas.testset.transforms import apply_transforms, Parallel
from ragas.testset.synthesizers.single_hop.base import (
    SingleHopQuerySynthesizer,
    SingleHopScenario,
)
from ragas.testset.synthesizers.base import QueryStyle, QueryLength

from ragas.testset import TestsetGenerator
from ragas.llms.base import llm_factory, LangchainLLMWrapper
from ragas.embeddings import embedding_factory, LangchainEmbeddingsWrapper
from ragas.testset.persona import Persona
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# from ragas_benchmark.utils.scispacyNER import SciSpacyNERExtractor
from utils.scispacyNER import SciSpacyNERExtractor

from utils.jaccardTFIDF import WeightedJaccardBuilder
from utils.centroid_cluster_builder import ClusterCentroidBuilder
from ragas.testset.transforms.relationship_builders.traditional import (
    JaccardSimilarityBuilder,
)

DEFAULT_PERSONA = Persona(
    name="Biomedical Reader",
    role_description="Reads and reasons over a single biomedical study abstract."
)


# LOAD INPUT STUDIES
def load_rows(path):
    import json
    path_str = str(path)
    if path_str.endswith(".csv"):
        df = pd.read_csv(
            path,
            engine="python",      # <-- handles multiline descriptions
            on_bad_lines="skip",  # <-- avoids crashes
        )
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "doc_id": r.get("Accession") or r.get("StudyId"),
                "title": r.get("Study Name") or r.get("StudyName") or "",
                "abstract": r.get("Description") or "",
                "permalink": r.get("Permalink") or "",
            })
        return rows
    elif path_str.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for r in data:
            rows.append({
                "doc_id": r.get("Accession") or r.get("StudyId"),
                "title": r.get("Study Name") or r.get("StudyName") or "",
                "abstract": r.get("Description") or "",
                "permalink": r.get("Permalink") or "",
            })
        return rows
    else:
        df = pd.read_excel(path)
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "doc_id": r.get("Accession") or r.get("StudyId"),
                "title": r.get("Study Name") or r.get("StudyName") or "",
                "abstract": r.get("Description") or "",
                "permalink": r.get("Permalink") or "",
            })
        return rows



def build_nodes(rows):
    nodes = []
    for r in rows:
        txt = f"{r['title']}. {r['abstract']}".strip()
        if not txt:
            continue
        nodes.append(
            Node(
                properties={
                    "page_content": txt,
                    "doc_id": r["doc_id"],
                    "title": r["title"],
                    "permalink": r["permalink"],
                },
                type=NodeType.DOCUMENT,
            )
        )
    return nodes

def entity_pairs(entities, max_pairs=3):
    if len(entities) < 2:
        return []
    pairs = list(zip(entities[:-1], entities[1:]))
    return pairs[:max_pairs]

# APPLY TRANSFORMS
async def enrich_graph(nodes, outdir):
    kg = KnowledgeGraph(nodes=nodes)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- NER ---
    ner = SciSpacyNERExtractor()
    maybe = apply_transforms(kg, [Parallel(ner)])
    if asyncio.iscoroutine(maybe):
        await maybe

    STOP = {"the", "and", "in", "of", "to"}
    for n in kg.nodes:
        ents = [
            e.lower().strip()
            for e in n.properties.get("entities", [])
            if e.lower() not in STOP
        ]
        n.properties["entities"] = ents

    transforms = [
        WeightedJaccardBuilder(property_name="entities", new_property_name="weighted_jaccard_similarity", threshold=0.5),
        JaccardSimilarityBuilder(property_name="entities", new_property_name="entity_jaccard_similarity", threshold=0.5),
        ClusterCentroidBuilder(property_name="entities", new_property_name="centroid_similarity"),
    ]

    maybe = apply_transforms(kg, transforms)
    if asyncio.iscoroutine(maybe):
        await maybe

    return kg

class MySingleHopQuery(SingleHopQuerySynthesizer):

    def __init__(self, kg, llm, is_async=True):
        super().__init__(llm=llm)
        self.kg = kg
        self.is_async = is_async

    async def _generate_scenarios(self, *args, n_samples=None, persona=None, **kwargs):

        if isinstance(persona, list):
            persona = persona[0]

        if persona is None:
            persona = Persona(
                name="Default",
                role_description="General biomedical reasoning agent."
            )

        if n_samples is None:
            n_samples = 20

        scenarios = []

        docs = [n for n in self.kg.nodes if n.type == NodeType.DOCUMENT]
        random.shuffle(docs)

        for doc in docs:
            ents = doc.properties.get("entities", [])
            pairs = entity_pairs(ents)

            for e1, e2 in pairs:
                scenario = SingleHopScenario(
                    term=f"{e1} → {e2}",
                    nodes=[doc],
                    persona=DEFAULT_PERSONA,
                    style=QueryStyle.PERFECT_GRAMMAR,
                    length=QueryLength.SHORT,
                )

                scenarios.append(scenario) 

                if len(scenarios) >= n_samples:
                    return scenarios

        return scenarios


async def _amain(args):
    rows = load_rows(args.input)
    nodes = build_nodes(rows)
    kg = await enrich_graph(nodes, Path(args.outdir))

    # Use local vLLM server
    llm_url = os.environ.get("LLM_URL", "http://localhost:9091/v1")
    llm_model = os.environ.get("LLM_MODEL", "google/gemma-3-12b-it")

    chat_llm = ChatOpenAI(
        model=llm_model,
        base_url=llm_url,
        api_key="not-needed",  # vLLM doesn't require API key
    )
    llm = LangchainLLMWrapper(chat_llm)

    # Use local HuggingFace embeddings
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    personas = [
        Persona(name="Clinician", role_description="Interprets biomedical results."),
        Persona(name="Researcher", role_description="Understands scientific logic."),
        Persona(name="Graduate Student", role_description="Learning biomedical reasoning."),
        Persona(name="Biostatistician", role_description="Looking for data for secondary analysis"),
    ]

    generator = TestsetGenerator(
        llm=llm,
        knowledge_graph=kg,
        embedding_model=embeddings,
        persona_list=[],         
    )


    print("[INFO] Generating SINGLE-HOP testset...")

    testset = generator.generate(
        args.n_samples,
        query_distribution=[
            (MySingleHopQuery(kg=kg, llm=llm, is_async=True), 1.0)
        ],
    )


    df = testset.to_pandas()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_json(out / "singlehop_testset.jsonl", lines=True, orient="records")

    print("[DONE] Wrote:", out / "singlehop_testset.jsonl")



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--n-samples", type=int, default=20)
    args = p.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
