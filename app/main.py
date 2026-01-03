"""
CodeReview Agent Service

A FastAPI service that acts as an 'Analyst Agent'. It receives git details from TaskSync,
loads a fine-tuned local LLM (Qwen2.5-Coder-7B via Unsloth), analyzes the code,
and reports back to the TMS.
"""

import os
import re
import json
import shutil
import tempfile
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import httpx
from git import Repo
from git.exc import GitCommandError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("codereview-agent")


# ============================================================================
# Configuration
# ============================================================================

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    tms_callback_url: str = Field(default="http://localhost:8000/api/callback")
    port: int = Field(default=5002)
    model_path: str = Field(default="models/github_agent_v1")
    mock_inference: bool = Field(default=False)
    max_seq_length: int = Field(default=2048)
    max_new_tokens: int = Field(default=512)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


# ============================================================================
# Pydantic Schemas
# ============================================================================

class AnalyzeRequest(BaseModel):
    """Request schema for the /analyze endpoint."""
    repo_url: str = Field(..., description="Git repository URL to clone")
    commit_sha: str = Field(..., description="Commit SHA to analyze")
    github_token: str = Field(..., description="GitHub token for authentication")
    tms_project_id: str = Field(..., description="TMS project ID for callback")


class CodeAnalysisResult(BaseModel):
    """Schema for the code analysis result from the model."""
    code_quality_score: int = Field(default=0, ge=0, le=100)
    critical_issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    fixed_code: str = Field(default="")


class AnalyzeResponse(BaseModel):
    """Response schema for the /analyze endpoint."""
    status: str = Field(..., description="Status of the analysis request")
    message: str = Field(..., description="Status message")
    tms_project_id: str = Field(..., description="TMS project ID")


class CallbackPayload(BaseModel):
    """Payload sent to TMS callback URL."""
    tms_project_id: str
    status: str
    analysis_result: Optional[CodeAnalysisResult] = None
    error: Optional[str] = None
    diff_preview: Optional[str] = None


# ============================================================================
# Global State for Model
# ============================================================================

class ModelState:
    """Global state for holding the loaded model and tokenizer."""
    model = None
    tokenizer = None
    is_loaded: bool = False


model_state = ModelState()


# ============================================================================
# Lifespan Context Manager
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Loads the model once at startup and cleans up at shutdown.
    """
    logger.info("Starting CodeReview Agent Service...")

    if settings.mock_inference:
        logger.info("MOCK_INFERENCE=True - Skipping model loading")
        model_state.is_loaded = False
    else:
        logger.info(f"Loading model from {settings.model_path}...")
        try:
            # Import unsloth here to avoid import errors in mock mode
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=settings.model_path,
                max_seq_length=settings.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)

            model_state.model = model
            model_state.tokenizer = tokenizer
            model_state.is_loaded = True

            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down CodeReview Agent Service...")
    if model_state.is_loaded:
        # Clear model from GPU memory
        model_state.model = None
        model_state.tokenizer = None
        model_state.is_loaded = False

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded and GPU memory cleared")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="CodeReview Agent",
    description="Analyst Agent for code review using fine-tuned Qwen2.5-Coder-7B",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# Git Operations
# ============================================================================

def clone_repo_and_get_diff(repo_url: str, github_token: str, commit_sha: str) -> str:
    """
    Clone a repository using the GitHub token and extract the diff.

    Args:
        repo_url: The repository URL (https://github.com/owner/repo)
        github_token: GitHub personal access token
        commit_sha: The commit SHA to analyze

    Returns:
        The git diff output as a string
    """
    # Create authenticated URL
    if "github.com" in repo_url:
        # Convert https://github.com/owner/repo to https://token@github.com/owner/repo
        auth_url = repo_url.replace("https://", f"https://{github_token}@")
    else:
        auth_url = repo_url

    temp_dir = tempfile.mkdtemp(prefix="codereview_")

    try:
        logger.info(f"Cloning repository to {temp_dir}...")

        # Clone the repository
        repo = Repo.clone_from(auth_url, temp_dir)

        # Checkout the specific commit
        repo.git.checkout(commit_sha)

        # Get the diff between this commit and its parent
        try:
            diff_output = repo.git.diff("HEAD~1", "HEAD")
        except GitCommandError:
            # If HEAD~1 doesn't exist (initial commit), show all changes
            diff_output = repo.git.show("--format=", "--name-only", "HEAD")
            diff_output += "\n" + repo.git.diff("--root", "HEAD")

        logger.info(f"Got diff of {len(diff_output)} characters")
        return diff_output

    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"Cleaned up temp directory: {temp_dir}")


# ============================================================================
# Prompt Formatting
# ============================================================================

def format_alpaca_prompt(code_diff: str) -> str:
    """
    Format the code diff into the Alpaca prompt format used during training.

    Args:
        code_diff: The git diff to analyze

    Returns:
        Formatted prompt string
    """
    # Truncate diff if too long to fit in context
    max_diff_length = 6000  # Leave room for prompt template and response
    if len(code_diff) > max_diff_length:
        code_diff = code_diff[:max_diff_length] + "\n... [truncated]"

    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Analyze this code for defects, security vulnerabilities, and quality issues. Return ONLY a JSON object.

### Input:
{code_diff}

### Response:
"""
    return prompt


# ============================================================================
# Inference
# ============================================================================

def run_inference(prompt: str) -> str:
    """
    Run model inference on the given prompt.
    This function runs synchronously and should be called via run_in_threadpool.

    Args:
        prompt: The formatted Alpaca prompt

    Returns:
        The model's generated text response
    """
    import torch

    if not model_state.is_loaded:
        raise RuntimeError("Model is not loaded")

    inputs = model_state.tokenizer(
        [prompt],
        return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    outputs = model_state.model.generate(
        **inputs,
        max_new_tokens=settings.max_new_tokens,
        use_cache=True
    )

    response_text = model_state.tokenizer.batch_decode(outputs)[0]
    return response_text


def get_mock_response() -> CodeAnalysisResult:
    """Return a mock response for development/testing."""
    return CodeAnalysisResult(
        code_quality_score=65,
        critical_issues=[
            "Potential SQL injection vulnerability detected",
            "Hardcoded credentials found in configuration"
        ],
        suggestions=[
            "Use parameterized queries for database operations",
            "Move credentials to environment variables",
            "Add input validation for user-provided data"
        ],
        fixed_code="# Mock fixed code - actual implementation would show corrected code"
    )


def parse_model_response(response_text: str) -> CodeAnalysisResult:
    """
    Parse the model's text response to extract the JSON analysis result.

    Args:
        response_text: The raw model output

    Returns:
        Parsed CodeAnalysisResult
    """
    # Extract the response part after "### Response:"
    if "### Response:" in response_text:
        json_part = response_text.split("### Response:")[-1]
    else:
        json_part = response_text

    # Clean up the response
    json_part = json_part.replace("<|endoftext|>", "").strip()
    json_part = json_part.replace("<|im_end|>", "").strip()

    # Try to find JSON object in the response
    # Look for content between { and }
    json_match = re.search(r'\{[\s\S]*\}', json_part)

    if json_match:
        try:
            json_str = json_match.group(0)
            data = json.loads(json_str)

            return CodeAnalysisResult(
                code_quality_score=data.get("code_quality_score", 0),
                critical_issues=data.get("critical_issues", []),
                suggestions=data.get("suggestions", []),
                fixed_code=data.get("fixed_code", "")
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from model response: {e}")
            logger.debug(f"Raw JSON part: {json_str}")

    # If JSON parsing fails, return a default result with the raw response
    logger.warning("Could not parse JSON from model response, returning raw text")
    return CodeAnalysisResult(
        code_quality_score=0,
        critical_issues=["Failed to parse model response"],
        suggestions=[f"Raw response: {json_part[:500]}..."],
        fixed_code=""
    )


# ============================================================================
# TMS Callback
# ============================================================================

async def send_callback_to_tms(payload: CallbackPayload):
    """
    Send the analysis result back to the TMS callback URL.

    Args:
        payload: The callback payload to send
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                settings.tms_callback_url,
                json=payload.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            logger.info(f"Successfully sent callback to TMS for project {payload.tms_project_id}")
    except httpx.HTTPError as e:
        logger.error(f"Failed to send callback to TMS: {e}")
    except Exception as e:
        logger.error(f"Unexpected error sending callback: {e}")


# ============================================================================
# Background Task for Analysis
# ============================================================================

async def perform_analysis(request: AnalyzeRequest):
    """
    Perform the full analysis pipeline as a background task.

    Steps:
    1. Clone repo and get diff
    2. Format prompt
    3. Run inference (or mock)
    4. Parse response
    5. Send callback to TMS
    """
    try:
        # Step 1: Clone repo and get diff
        logger.info(f"Starting analysis for project {request.tms_project_id}")

        diff = await run_in_threadpool(
            clone_repo_and_get_diff,
            request.repo_url,
            request.github_token,
            request.commit_sha
        )

        if not diff.strip():
            logger.warning("Empty diff received")
            diff = "No changes detected in this commit."

        diff_preview = diff[:500] + "..." if len(diff) > 500 else diff

        # Step 2 & 3: Format prompt and run inference
        if settings.mock_inference:
            logger.info("Using mock inference")
            result = get_mock_response()
        else:
            prompt = format_alpaca_prompt(diff)
            logger.info("Running model inference...")

            response_text = await run_in_threadpool(run_inference, prompt)

            # Step 4: Parse response
            result = parse_model_response(response_text)

        logger.info(f"Analysis complete. Score: {result.code_quality_score}")

        # Step 5: Send callback
        callback = CallbackPayload(
            tms_project_id=request.tms_project_id,
            status="completed",
            analysis_result=result,
            diff_preview=diff_preview
        )
        await send_callback_to_tms(callback)

    except GitCommandError as e:
        logger.error(f"Git error: {e}")
        callback = CallbackPayload(
            tms_project_id=request.tms_project_id,
            status="failed",
            error=f"Git operation failed: {str(e)}"
        )
        await send_callback_to_tms(callback)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        callback = CallbackPayload(
            tms_project_id=request.tms_project_id,
            status="failed",
            error=str(e)
        )
        await send_callback_to_tms(callback)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "CodeReview Agent",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model_state.is_loaded,
        "mock_mode": settings.mock_inference
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_state.is_loaded,
        "mock_mode": settings.mock_inference
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Analyze code changes from a git repository.

    This endpoint accepts the analysis request and immediately returns.
    The actual analysis is performed in the background, and results
    are sent to the TMS callback URL when complete.
    """
    # Validate that we can process requests
    if not settings.mock_inference and not model_state.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is not ready to process requests."
        )

    logger.info(f"Received analysis request for project {request.tms_project_id}")
    logger.info(f"Repository: {request.repo_url}, Commit: {request.commit_sha}")

    # Queue the analysis as a background task
    background_tasks.add_task(perform_analysis, request)

    return AnalyzeResponse(
        status="accepted",
        message="Analysis request queued. Results will be sent to TMS callback URL.",
        tms_project_id=request.tms_project_id
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=False,
        log_level="info"
    )
