"""
Central configuration — loaded once at import time.
All other modules import from here instead of re-reading config.yaml.
"""
import os
import yaml

_CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def _load_cfg() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)


CFG = _load_cfg()
EMBED_MODEL          = CFG.get("embed_model", "Snowflake/snowflake-arctic-embed-xs")
AI_MODE_CFG          = CFG.get("ai_mode", {})
AI_MODE_LLM_OPTIONS  = AI_MODE_CFG.get("llm_options", {})
AI_MODE_SIM_THRESH   = AI_MODE_CFG.get("semantic_cache", {}).get("similarity_threshold", 0.92)

LLM_TIMEOUT = CFG.get("llm_timeout", 300)
LLAMACPP_CFG   = CFG.get("llamacpp", {})
