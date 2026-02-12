from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db import init_db

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# routers
from api.granulate import router as granulate_router
from api.projects import router as projects_router

app.include_router(granulate_router)
app.include_router(projects_router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
