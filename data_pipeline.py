"""
ForgeGuard — FastAPI Inference Server
Full REST API with prediction, metrics, feedback, and health endpoints.

Run: uvicorn src.api.server:app --host 0.0.0.0 --port 8000
"""

import os
import json
import time
import torch
from datetime import datetime
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn")


# ── Request / Response Models ─────────────────────────────────────────────────
if FASTAPI_AVAILABLE:

    class PredictRequest(BaseModel):
        question: str
        temperature: Optional[float] = 0.1
        max_new_tokens: Optional[int] = 150

    class PredictResponse(BaseModel):
        question: str
        answer: str
        confidence: float
        is_refusal: bool
        perplexity: Optional[float] = None
        latency_ms: float
        model_version: str
        timestamp: str

    class FeedbackRequest(BaseModel):
        question: str
        predicted_answer: str
        correct_answer: str
        confidence: float
        was_hallucination: bool

    class FeedbackResponse(BaseModel):
        message: str
        feedback_id: int
        total_feedback_collected: int

    class MetricsResponse(BaseModel):
        accuracy: float
        hallucination_rate: float
        ece: float
        avg_perplexity: float
        valid_refusal_rate: float
        total_queries: int
        model_version: str

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        device: str
        vram_used_gb: Optional[float]
        uptime_seconds: float
        version: str


# ── App State ─────────────────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.engine       = None
        self.config       = None
        self.start_time   = time.time()
        self.total_queries= 0
        self.feedback_log = []
        self.metrics_cache= {}
        self.model_version= "phase2"


state = AppState()


def load_model_on_startup():
    """Load model when server starts."""
    try:
        from src.training.model_loader import load_config
        from src.inference.inference_engine import load_inference_engine

        config_path = "configs/training_config.yaml"
        state.config = load_config(config_path)

        # Try adapters in order of preference
        adapter_paths = [
            "models/adapters/self_improved/final_adapter",
            "models/adapters/phase2",
            "models/adapters/phase1",
        ]
        for path in adapter_paths:
            if os.path.exists(path):
                state.engine = load_inference_engine(path, config_path)
                state.model_version = os.path.basename(path)
                print(f"✅ API: Model loaded from {path}")
                break

        if state.engine is None:
            print("⚠️  API: No trained adapter found — using demo mode")

    except Exception as e:
        print(f"⚠️  API: Could not load model: {e}")


# ── Create App ────────────────────────────────────────────────────────────────
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="ForgeGuard API",
        description="Self-Improving Hallucination-Calibrated LLM Inference API",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        load_model_on_startup()

    # ── Endpoints ─────────────────────────────────────────────────────────────

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": "ForgeGuard API",
            "version": "1.0.0",
            "description": "Self-Improving Hallucination-Calibrated LLM",
            "endpoints": ["/predict", "/metrics", "/health", "/feedback", "/docs"]
        }

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health():
        """Check if the model is loaded and ready."""
        vram = None
        device = "cpu"
        if torch.cuda.is_available():
            vram   = round(torch.cuda.memory_allocated() / 1024**3, 2)
            device = torch.cuda.get_device_name(0)

        return HealthResponse(
            status      = "healthy" if state.engine else "degraded",
            model_loaded= state.engine is not None,
            device      = device,
            vram_used_gb= vram,
            uptime_seconds= round(time.time() - state.start_time, 1),
            version     = state.model_version,
        )

    @app.post("/predict", response_model=PredictResponse, tags=["Inference"])
    async def predict(request: PredictRequest):
        """
        Generate an answer with confidence score.

        - **question**: The question to answer
        - **temperature**: Sampling temperature (0.0 = deterministic)
        - **max_new_tokens**: Maximum tokens to generate
        """
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        start = time.time()
        state.total_queries += 1

        if state.engine is None:
            # Demo mode
            return PredictResponse(
                question    = request.question,
                answer      = "Model not loaded. Please train the model first.",
                confidence  = 0.0,
                is_refusal  = True,
                latency_ms  = 0.0,
                model_version=state.model_version,
                timestamp   = datetime.now().isoformat(),
            )

        try:
            result = state.engine.generate(
                request.question,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

        latency_ms = round((time.time() - start) * 1000, 1)

        return PredictResponse(
            question     = request.question,
            answer       = result.get("answer", ""),
            confidence   = result.get("confidence", 0.5),
            is_refusal   = result.get("is_refusal", False),
            perplexity   = result.get("perplexity"),
            latency_ms   = latency_ms,
            model_version= state.model_version,
            timestamp    = datetime.now().isoformat(),
        )

    @app.post("/feedback", response_model=FeedbackResponse, tags=["Improvement"])
    async def feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
        """
        Submit feedback to improve the model.
        Hallucination corrections are queued for retraining.
        """
        feedback_entry = {
            "id":               len(state.feedback_log) + 1,
            "question":         request.question,
            "predicted_answer": request.predicted_answer,
            "correct_answer":   request.correct_answer,
            "confidence":       request.confidence,
            "was_hallucination":request.was_hallucination,
            "timestamp":        datetime.now().isoformat(),
        }
        state.feedback_log.append(feedback_entry)

        # Save feedback to file
        feedback_path = "data/synthetic/api_feedback.jsonl"
        os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
        with open(feedback_path, "a") as f:
            f.write(json.dumps(feedback_entry) + "\n")

        # If hallucination, queue for retraining
        if request.was_hallucination:
            background_tasks.add_task(
                save_corrective_sample,
                request.question,
                request.correct_answer
            )

        return FeedbackResponse(
            message=f"Feedback received. {'Queued for retraining.' if request.was_hallucination else 'Thank you!'}",
            feedback_id=feedback_entry["id"],
            total_feedback_collected=len(state.feedback_log),
        )

    @app.get("/metrics", response_model=MetricsResponse, tags=["Evaluation"])
    async def get_metrics():
        """Get current model evaluation metrics."""
        metrics_path = "outputs/metrics/evaluation_results.json"
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                data = json.load(f)
            m = data.get("metrics", {})
            return MetricsResponse(
                accuracy          = m.get("accuracy", 0.0),
                hallucination_rate= m.get("hallucination_rate", 0.0),
                ece               = m.get("ece", 0.0),
                avg_perplexity    = m.get("avg_perplexity", 0.0),
                valid_refusal_rate= m.get("valid_refusal_rate", 0.0),
                total_queries     = state.total_queries,
                model_version     = state.model_version,
            )
        raise HTTPException(status_code=404, detail="No evaluation results found. Run Phase 3 first.")

    @app.get("/feedback/log", tags=["Improvement"])
    async def get_feedback_log():
        """Get all collected feedback entries."""
        return {
            "total": len(state.feedback_log),
            "hallucinations_reported": sum(1 for f in state.feedback_log if f["was_hallucination"]),
            "entries": state.feedback_log[-20:],  # last 20
        }

    @app.post("/retrain", tags=["Improvement"])
    async def trigger_retrain(background_tasks: BackgroundTasks):
        """Trigger self-improvement retraining on collected feedback."""
        hallucinations = [f for f in state.feedback_log if f["was_hallucination"]]
        if len(hallucinations) < 3:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 3 hallucination reports to retrain. Have {len(hallucinations)}."
            )
        background_tasks.add_task(run_retraining_job)
        return {"message": f"Retraining started with {len(hallucinations)} hallucination corrections."}

    # ── Background Tasks ──────────────────────────────────────────────────────

    async def save_corrective_sample(question: str, correct_answer: str):
        """Save a corrective sample for future retraining."""
        sample = {
            "instruction": question,
            "response": f"Answer: {correct_answer}\nConfidence: 0.85",
            "label": "api_correction",
            "source": "user_feedback",
        }
        path = "data/synthetic/api_corrections.jsonl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(sample) + "\n")

    async def run_retraining_job():
        """Run self-improvement loop in background."""
        try:
            from src.self_improvement.self_improvement_loop import run_self_improvement_loop
            run_self_improvement_loop(max_iterations=1)
            print("✅ Background retraining complete")
        except Exception as e:
            print(f"❌ Background retraining failed: {e}")


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI server."""
    try:
        import uvicorn
        print(f"\n🚀 ForgeGuard API starting at http://{host}:{port}")
        print(f"   Docs: http://{host}:{port}/docs")
        print(f"   Health: http://{host}:{port}/health")
        uvicorn.run("src.api.server:app", host=host, port=port, reload=reload)
    except ImportError:
        print("Install uvicorn: pip install uvicorn")


if __name__ == "__main__":
    start_server()
