from pydantic import BaseModel
from typing import Optional, Literal

ProjectStatus = Literal["queued", "processing", "completed", "failed"]

class CreateProjectResponse(BaseModel):
    project_id: str
    status: ProjectStatus

class ProjectResponse(BaseModel):
    project_id: str
    name: str
    status: ProjectStatus
    created_at: str
    updated_at: str
    input_path: str
    error_message: Optional[str] = None
