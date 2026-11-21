"""FastAPI application for Q-Agri Lab."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.agent import router as agent_router
from app.api.routes.experiments import router as experiments_router
from app.api.routes.quantum import router as quantum_router
from app.api.routes.results import router as results_router

app = FastAPI(title="Q-Agri Lab API", version="0.1.0", description="Research experiments for Moroccan crop classification.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(agent_router, prefix="/api")
app.include_router(experiments_router, prefix="/api")
app.include_router(results_router, prefix="/api")
app.include_router(quantum_router, prefix="/api")


@app.get("/")
async def root():
    return {"name": "Q-Agri Lab API", "status": "ok", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
