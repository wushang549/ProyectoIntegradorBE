from pathlib import Path
from uuid import uuid4

UPLOADS_DIR = Path(__file__).resolve().parents[1] / "uploads"

def new_project_id() -> str:
    return uuid4().hex

def ensure_project_dir(project_id: str) -> Path:
    d = UPLOADS_DIR / project_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_upload(project_id: str, filename: str, content: bytes) -> Path:
    project_dir = ensure_project_dir(project_id)
    safe_name = filename.replace("..", "").replace("/", "_").replace("\\", "_")
    out_path = project_dir / safe_name
    out_path.write_bytes(content)
    return out_path
