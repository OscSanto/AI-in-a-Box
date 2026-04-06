"""
Central configuration — loaded once at import time.
All other modules import from here instead of re-reading config.yaml.
"""
import os
import re
import yaml
import ollama

_CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def _load_cfg() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)


def _load_prompt(rel_path: str) -> str:
    """Load a system prompt from a .md file path relative to the config directory."""
    full_path = os.path.join(os.path.dirname(_CFG_PATH), rel_path)
    with open(full_path) as f:
        return f.read().strip()


CFG = _load_cfg()

BASE_URL  = CFG.get("base_url", "http://box")
NAV_RE    = re.compile(r"^(List(s)? of|Index of|Outline of|Portal:|Category:)|\(disambiguation\)$", re.I)

EMBED_HF_MODEL = CFG.get("embed_hf_model", "Snowflake/snowflake-arctic-embed-xs")
LLM_MODEL      = CFG.get("llm_model", "smollm2:360m")
LLM_OPTIONS    = CFG.get("llm_options", {})
ZIMS           = CFG.get("zims", [])
RAG            = CFG.get("rag", {})

_SEARCH_CFG        = CFG.get("search", {})
SEARCH_MAX_RESULTS = _SEARCH_CFG.get("max_results", 25)
TITLE_WEIGHT_FLOOR = _SEARCH_CFG.get("title_weight_floor", 0.2)

AI_PROMPT        = _load_prompt(RAG.get("system_prompt", "prompts/ai.md"))
AI_MODE_CFG      = CFG.get("ai_mode", {})
AI_MODE_PROMPT   = _load_prompt(AI_MODE_CFG.get("system_prompt", "prompts/ai_mode.md"))

AI_MODE_LLM_MODEL   = AI_MODE_CFG.get("llm_model", "smollm2:360m")
AI_MODE_LLM_OPTIONS = AI_MODE_CFG.get("llm_options", {})
AI_MODE_SIM_THRESH  = AI_MODE_CFG.get("semantic_cache", {}).get("similarity_threshold", 0.92)

OLLAMA_TIMEOUT = CFG.get("ollama_timeout", 300)
ollama_client  = ollama.Client(timeout=OLLAMA_TIMEOUT)
