"""
llama-server process manager.

Starts, stops, and restarts the llama-server subprocess when the active model
changes. Only used when backend = "llamacpp" in config.yaml.
"""
import shlex
import subprocess
import threading
import time
from pathlib import Path

from SearchEngine.config import LLAMACPP_CFG

_proc: subprocess.Popen | None = None
_proc_lock = threading.Lock()
_active_model: str = LLAMACPP_CFG.get("model", "")


def _server_url() -> str:
    return LLAMACPP_CFG.get("url", "http://localhost:8080")


def _model_dir() -> Path:
    return Path(LLAMACPP_CFG.get("model_dir", "/library/models"))


def list_gguf_models() -> list[str]:
    """Return filenames of all .gguf files found in model_dir."""
    d = _model_dir()
    if not d.exists():
        return []
    return sorted(p.name for p in d.glob("*.gguf"))


def active_model() -> str:
    return _active_model


def start(model_filename: str) -> None:
    """Start llama-server with the given model file. Stops any running instance first."""
    global _proc, _active_model

    model_path = _model_dir() / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    url = _server_url()
    host, _, port_str = url.replace("http://", "").partition(":")
    port = int(port_str) if port_str else 8080

    threads    = int(LLAMACPP_CFG.get("threads", 4))
    ctx_size   = int(LLAMACPP_CFG.get("ctx_size", 2048))
    batch_size = int(LLAMACPP_CFG.get("batch_size", 512))
    ubatch     = int(LLAMACPP_CFG.get("ubatch_size", 128))
    mlock      = bool(LLAMACPP_CFG.get("mlock", False))
    flash_attn = bool(LLAMACPP_CFG.get("flash_attn", False))
    extra      = LLAMACPP_CFG.get("extra_args", "").strip()

    cmd = (
        f"llama-server --model {model_path} "
        f"--host {host} --port {port} "
        f"--threads {threads} --ctx-size {ctx_size} "
        f"--batch-size {batch_size} --ubatch-size {ubatch} "
        f"--no-warmup"
    )
    if mlock:
        cmd += " --mlock"
    if flash_attn:
        cmd += " --flash-attn on"
    if extra:
        cmd += f" {extra}"

    with _proc_lock:
        _stop_locked()
        _proc = subprocess.Popen(
            shlex.split(cmd),
            stdout=subprocess.DEVNULL,
            stderr=None,  # inherit stderr → llama-server errors appear in app log
        )
        _active_model = model_filename

    # Give the server a moment to bind
    _wait_ready(url, timeout=60)


def stop() -> None:
    with _proc_lock:
        _stop_locked()


def _stop_locked() -> None:
    global _proc
    if _proc and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None


def _wait_ready(url: str, timeout: int = 120) -> None:
    import httpx
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(f"llama-server did not become ready within {timeout}s")


def is_running() -> bool:
    with _proc_lock:
        return _proc is not None and _proc.poll() is None
