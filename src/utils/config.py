import yaml
from dataclasses import dataclass, field

"""
YAML CHANGES MUST BE INCLUDED IN CONFIG.PY FILE AND VICE-VERSAA

ANY CHANGES TO YAML MUST FIRST OCCUR HERE!
"""

# Default intent list used when intent_classifier section is absent from YAML.
# Each entry: {name, description, enabled}.  Order matters — shown to the LLM verbatim.
_DEFAULT_INTENTS = [
    {"name": "kiwix",    "description": "Factual question or topic to look up in the knowledge base",                          "enabled": True},
    {"name": "summarize","description": "User wants to summarize content they provided — pasted text, a URL, or a PDF",        "enabled": True},
    {"name": "wiki_url", "description": "User provided a local Wikipedia/Kiwix article URL and wants to ask about it",         "enabled": True},
    {"name": "chat",     "description": "Greeting, casual conversation, or non-retrieval question that needs no knowledge lookup", "enabled": True},
]

#LOAD YAML
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

@dataclass
class Config:
    # VALUES OVERWRITTEN VIA YAML.

    # Experiment
    experiment_name: str = ""
    experiment_alias: str = ""

    # Kiwix
    kiwix_endpoint: str = "http://127.0.0.1/kiwix/search"
    kiwix_result_limit: int = 10
    zim_search_term: str = "Albert"
    zim_content_id: str = ""

    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 0  # 0 = auto-detect from model on first embed

    # LLM — generation model (Stage 11, final answer)
    llm_model: str = "qwen2.5:1.5b-instruct-q4_K_M"
    llm_temperature: float = 0.2
    llm_backend: str = "ollama"
    llm_max_tokens: int = 256

    # LLM — utility model (Stage 3 query rewrite, Stage 8.5 relation extraction — fast/small)
    llm_utility_model: str = "qwen2.5:0.5b"


    # Chunking
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 96
    max_articles_to_chunk: int = 3
    max_chunks_per_article: int = 15       # hard cap on chunks per article (top-scoring article)
    title_score_threshold: float = 0.3    # articles below this title-rank score are dropped
    max_chunk_chars: int = 800             # merge small paragraphs up to this char count
    article_cleaning_enabled: bool = True  # strip junk sections before chunking
    min_paragraph_chars: int = 50

    # Chunk ranking
    topk_chunks: int = 8

    # Content graph
    content_graph_enabled: bool = True
    content_graph_max_hops: int = 2
    content_graph_max_neighbors: int = 15

    # Query memory
    query_memory_enabled: bool = True
    qm_high_sim: float = 0.92
    qm_medium_sim: float = 0.85
    qm_topk: int = 5

    # Embedder keep-alive — how long Ollama keeps the embedding model in RAM after last use
    embedding_keep_alive: str = "10m"

    # Query rewrite (Stage 3 — LLM expands abbreviations, synonyms, resolves ambiguity)
    # Deprecated: rewrite is now folded into intent classification (intent_classifier_enabled).
    query_rewrite_enabled: bool = False

    # Intent classifier (Stage 2 — single LLM call: classify intent + rewrite query)
    intent_classifier_enabled: bool = True
    intent_routing_enabled: bool = False   # False = always kiwix; True = route chat/summarize/wiki_url
    intent_min_text_chars: int = 300   # heuristic threshold for summarize:text pre-check
    intent_classifier_intents: list = field(default_factory=lambda: list(_DEFAULT_INTENTS))
    kiwix_max_queries: int = 3         # max search terms the LLM produces for kiwix mode
    kiwix_max_entities: int = 3        # max named entities the LLM extracts for exact-title lookup

    # LLM relation extraction (entity→relation→entity triples stored in graph)
    entity_llm_enabled: bool = False
    entity_llm_mode: str = "server"     # "server" (Ollama) or "client" (WebLLM, POSTs to /relations)

    # Inference mode
    inference_default_mode: str = "server"   # "server" or "client"
    inference_allow_user_toggle: bool = True

    # Server
    server_port: int = 5050
    server_host: str = "0.0.0.0"

    # Storage — defaults to data/{experiment_name}/ 
    #  Check YAML storage.data_dir
    data_dir: str = "./data/default/"

def config_print(d: dict):
    # Print Startup Config

    return

def config_from_dict(d: dict) -> Config:
    # lOAD YAML SECTIONS
    experiment = d.get("experiment", {})
    kiwix = d.get("kiwix", {})

    dataset = d.get("dataset", {})

    qe = d.get("query_embedding", {})

    llm = d.get("llm", {})

    extraction = d.get("article_extraction", {})
    chunking = d.get("chunking", {})
    chunk_ranking = d.get("chunk_ranking", {})

    cg = d.get("content_graph", {})
    cg_traversal = cg.get("traversal", {})

    qmg = d.get("query_memory_graph", {})
    qmg_sim = qmg.get("similarity_search", {})
    qmg_thresh = qmg_sim.get("thresholds", {})

    server = d.get("server", {})
    ner = d.get("ner", {})
    query_rewrite = d.get("query_rewrite", {})
    inference = d.get("inference", {})
    storage = d.get("storage") or {}
    ic = d.get("intent_classifier", {})

    exp_name = experiment.get("name", "default")
    default_data_dir = f"./data/{exp_name}/"

    return Config(
        experiment_name=exp_name,
        experiment_alias=experiment.get("alias", ""),
        kiwix_endpoint=kiwix.get("endpoint", "http://127.0.0.1/kiwix/search"),
        kiwix_result_limit=kiwix.get("result_limit", 10),
        zim_search_term=dataset.get("zim_search_term", "Albert"),
        zim_content_id=dataset.get("zim_content_id", dataset.get("zim_path", "")),
        embedding_model=qe.get("model", "BAAI/bge-small-en-v1.5"),
        embedding_dim=0 if qe.get("dimension") in (None, "auto") else int(qe.get("dimension", 384)),
        llm_model=llm.get("model", "qwen2.5:1.5b-instruct-q4_K_M"),
        llm_utility_model=llm.get("utility_model", "qwen2.5:0.5b"),
        llm_temperature=llm.get("temperature", 0.2),
        llm_backend=llm.get("backend", "ollama"),
        llm_max_tokens=llm.get("max_new_tokens", 256),
        chunk_size_tokens=chunking.get("chunk_tokens", 512),
        chunk_overlap_tokens=chunking.get("overlap_tokens", 0),
        max_articles_to_chunk=chunking.get("max_articles_to_chunk", 3),
        max_chunks_per_article=chunking.get("max_chunks_per_article", 15),
        title_score_threshold=chunking.get("title_score_threshold", 0.3),
        max_chunk_chars=chunking.get("max_chunk_chars", 800),
        article_cleaning_enabled=chunking.get("article_cleaning", True),
        min_paragraph_chars=chunking.get("min_paragraph_chars", 50),
        topk_chunks=chunk_ranking.get("topk_chunks", 8),
        content_graph_enabled=cg.get("enabled", True),
        content_graph_max_hops=cg_traversal.get("max_hops", 2),
        content_graph_max_neighbors=cg_traversal.get("max_neighbors", 15),
        query_memory_enabled=qmg.get("enabled", True),
        qm_high_sim=qmg_thresh.get("high_similarity", 0.92),
        qm_medium_sim=qmg_thresh.get("medium_similarity", 0.85),
        qm_topk=qmg_sim.get("topk", 5),
        embedding_keep_alive=qe.get("keep_alive", "10m"),
        query_rewrite_enabled=query_rewrite.get("enabled", False),
        intent_classifier_enabled=ic.get("enabled", True),
        intent_routing_enabled=ic.get("routing_enabled", False),
        intent_min_text_chars=ic.get("min_text_chars", 300),
        intent_classifier_intents=ic.get("intents", list(_DEFAULT_INTENTS)),
        kiwix_max_queries=ic.get("max_queries", 3),
        kiwix_max_entities=ic.get("max_entities", 3),
        entity_llm_enabled=ner.get("llm_extraction_enabled", False),
        entity_llm_mode=ner.get("llm_extraction_mode", "server"),
        inference_default_mode=inference.get("default_mode", "server"),
        inference_allow_user_toggle=inference.get("allow_user_toggle", True),
        server_port=server.get("port", 5050),
        server_host=server.get("host", "0.0.0.0"),
        data_dir=storage.get("data_dir", default_data_dir),
    )
