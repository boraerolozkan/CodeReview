"""
Comprehensive tests for app/main.py - CodeReview Agent Service

This test module achieves >90% code coverage by testing:
- All API endpoints (/, /health, /analyze)
- Git operations (clone_repo_and_get_diff)
- Prompt formatting (format_alpaca_prompt)
- Model inference (run_inference, parse_model_response)
- TMS callbacks (send_callback_to_tms)
- Lifespan/startup logic
- Error handling paths
"""

import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

# ==============================================================================
# CRITICAL: Mock GPU-dependent modules BEFORE any imports from app.main
# ==============================================================================

# Create comprehensive mocks for unsloth and torch
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.cuda.empty_cache = MagicMock()

mock_unsloth = MagicMock()
mock_fast_language_model = MagicMock()
mock_fast_language_model.from_pretrained.return_value = (MagicMock(), MagicMock())
mock_fast_language_model.for_inference = MagicMock()
mock_unsloth.FastLanguageModel = mock_fast_language_model

# Inject mocks into sys.modules BEFORE importing app.main
sys.modules["torch"] = mock_torch
sys.modules["unsloth"] = mock_unsloth

# Set environment variables BEFORE importing app.main
os.environ["MOCK_INFERENCE"] = "True"
os.environ["TMS_CALLBACK_URL"] = "http://test-tms/callback"
os.environ["MODEL_PATH"] = "test/model/path"

# ==============================================================================
# NOW import app.main (after mocks are in place)
# ==============================================================================

import app.main as app_module
from app.main import (
    app,
    model_state,
    format_alpaca_prompt,
    parse_model_response,
    get_mock_response,
    clone_repo_and_get_diff,
    run_inference,
    send_callback_to_tms,
    perform_analysis,
    lifespan,
    settings,
    CodeAnalysisResult,
    CallbackPayload,
    AnalyzeRequest,
    Settings,
    ModelState,
)
from fastapi.testclient import TestClient
from git.exc import GitCommandError


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def client():
    """Create a test client with model state properly set."""
    # Save original state
    original_model = model_state.model
    original_tokenizer = model_state.tokenizer
    original_is_loaded = model_state.is_loaded

    with TestClient(app) as test_client:
        yield test_client

    # Restore original state
    model_state.model = original_model
    model_state.tokenizer = original_tokenizer
    model_state.is_loaded = original_is_loaded


@pytest.fixture
def mock_model_loaded():
    """Set up mock model and tokenizer as if they're loaded."""
    fake_model = MagicMock()
    fake_tokenizer = MagicMock()

    # Mock tokenizer methods
    fake_tokenizer.return_value = MagicMock()
    fake_tokenizer.return_value.to.return_value = {"input_ids": MagicMock()}
    fake_tokenizer.batch_decode.return_value = ['{"code_quality_score": 85}']

    # Mock model generate
    fake_model.generate.return_value = MagicMock()

    # Inject into model_state
    model_state.model = fake_model
    model_state.tokenizer = fake_tokenizer
    model_state.is_loaded = True

    yield fake_model, fake_tokenizer

    # Cleanup
    model_state.model = None
    model_state.tokenizer = None
    model_state.is_loaded = False


# ==============================================================================
# Test API Endpoints
# ==============================================================================

class TestRootEndpoint:
    """Tests for the root (/) endpoint."""

    def test_root_returns_service_info(self, client):
        """Test that root endpoint returns service information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "CodeReview Agent"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert "model_loaded" in data
        assert "mock_mode" in data


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_healthy(self, client):
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "mock_mode" in data


class TestAnalyzeEndpoint:
    """Tests for the /analyze endpoint."""

    def test_analyze_returns_503_when_model_not_loaded(self, client):
        """Test that /analyze returns 503 when model is not loaded and not in mock mode."""
        # Temporarily disable mock mode and ensure model is not loaded
        with patch.object(settings, 'mock_inference', False):
            model_state.is_loaded = False
            model_state.model = None

            payload = {
                "repo_url": "https://github.com/test/repo",
                "commit_sha": "abc123",
                "github_token": "test_token",
                "tms_project_id": "project_1"
            }

            response = client.post("/analyze", json=payload)
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]

    def test_analyze_success_mock_mode(self, client):
        """Test successful /analyze request in mock mode."""
        # Ensure mock mode is enabled (default in our test setup)
        payload = {
            "repo_url": "https://github.com/test/repo",
            "commit_sha": "abc123",
            "github_token": "test_token",
            "tms_project_id": "project_1"
        }

        response = client.post("/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert data["tms_project_id"] == "project_1"
        assert "queued" in data["message"].lower() or "Analysis request" in data["message"]

    def test_analyze_success_with_model_loaded(self, client, mock_model_loaded):
        """Test successful /analyze request when model is loaded."""
        with patch.object(settings, 'mock_inference', False):
            payload = {
                "repo_url": "https://github.com/test/repo",
                "commit_sha": "abc123",
                "github_token": "test_token",
                "tms_project_id": "project_1"
            }

            response = client.post("/analyze", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "accepted"

    def test_analyze_invalid_payload(self, client):
        """Test /analyze with invalid payload."""
        response = client.post("/analyze", json={})
        assert response.status_code == 422  # Validation error


# ==============================================================================
# Test Prompt Formatting
# ==============================================================================

class TestFormatAlpacaPrompt:
    """Tests for format_alpaca_prompt function."""

    def test_format_prompt_basic(self):
        """Test basic prompt formatting."""
        diff = "def hello(): pass"
        prompt = format_alpaca_prompt(diff)

        assert "### Instruction:" in prompt
        assert "### Input:" in prompt
        assert "### Response:" in prompt
        assert diff in prompt
        assert "Analyze this code" in prompt

    def test_format_prompt_truncation(self):
        """Test that long diffs are truncated."""
        long_diff = "x" * 10000
        prompt = format_alpaca_prompt(long_diff)

        assert "[truncated]" in prompt
        assert len(prompt) < len(long_diff) + 1000  # Some overhead for template


# ==============================================================================
# Test Model Response Parsing
# ==============================================================================

class TestParseModelResponse:
    """Tests for parse_model_response function."""

    def test_parse_valid_json_response(self):
        """Test parsing a valid JSON response."""
        response = '''### Response:
        {"code_quality_score": 85, "critical_issues": ["issue1"], "suggestions": ["suggestion1"], "fixed_code": "fixed"}'''

        result = parse_model_response(response)

        assert result.code_quality_score == 85
        assert result.critical_issues == ["issue1"]
        assert result.suggestions == ["suggestion1"]
        assert result.fixed_code == "fixed"

    def test_parse_response_without_response_marker(self):
        """Test parsing when ### Response: marker is missing."""
        response = '{"code_quality_score": 75, "critical_issues": [], "suggestions": [], "fixed_code": ""}'

        result = parse_model_response(response)

        assert result.code_quality_score == 75

    def test_parse_response_with_special_tokens(self):
        """Test parsing response with special tokens like <|endoftext|>."""
        response = '''### Response:
        {"code_quality_score": 90}<|endoftext|><|im_end|>'''

        result = parse_model_response(response)

        assert result.code_quality_score == 90

    def test_parse_invalid_json_response(self):
        """Test parsing when JSON is invalid."""
        response = "### Response:\nThis is not valid JSON at all"

        result = parse_model_response(response)

        assert result.code_quality_score == 0
        assert "Failed to parse model response" in result.critical_issues

    def test_parse_partial_json_fields(self):
        """Test parsing JSON with only some fields."""
        response = '### Response:\n{"code_quality_score": 50}'

        result = parse_model_response(response)

        assert result.code_quality_score == 50
        assert result.critical_issues == []
        assert result.suggestions == []
        assert result.fixed_code == ""


# ==============================================================================
# Test Mock Response
# ==============================================================================

class TestGetMockResponse:
    """Tests for get_mock_response function."""

    def test_mock_response_structure(self):
        """Test that mock response has correct structure."""
        result = get_mock_response()

        assert isinstance(result, CodeAnalysisResult)
        assert result.code_quality_score == 65
        assert len(result.critical_issues) == 2
        assert len(result.suggestions) == 3
        assert result.fixed_code != ""


# ==============================================================================
# Test Git Operations
# ==============================================================================

class TestCloneRepoAndGetDiff:
    """Tests for clone_repo_and_get_diff function."""

    @patch("app.main.shutil.rmtree")
    @patch("app.main.tempfile.mkdtemp")
    @patch("app.main.Repo.clone_from")
    def test_clone_and_diff_success(self, mock_clone, mock_mkdtemp, mock_rmtree):
        """Test successful repo clone and diff extraction."""
        mock_mkdtemp.return_value = "/tmp/test_dir"

        mock_repo = MagicMock()
        mock_clone.return_value = mock_repo
        mock_repo.git.diff.return_value = "diff --git a/file.py\n+new line"

        result = clone_repo_and_get_diff(
            "https://github.com/test/repo",
            "test_token",
            "abc123"
        )

        assert "diff" in result or "new line" in result
        mock_clone.assert_called_once()
        mock_repo.git.checkout.assert_called_with("abc123")
        mock_rmtree.assert_called()

    @patch("app.main.shutil.rmtree")
    @patch("app.main.tempfile.mkdtemp")
    @patch("app.main.Repo.clone_from")
    def test_clone_initial_commit(self, mock_clone, mock_mkdtemp, mock_rmtree):
        """Test handling of initial commit (no HEAD~1)."""
        mock_mkdtemp.return_value = "/tmp/test_dir"

        mock_repo = MagicMock()
        mock_clone.return_value = mock_repo
        # Simulate HEAD~1 not existing
        mock_repo.git.diff.side_effect = [GitCommandError("diff", "HEAD~1 not found"), "initial content"]
        mock_repo.git.show.return_value = "file.py"

        result = clone_repo_and_get_diff(
            "https://github.com/test/repo",
            "test_token",
            "abc123"
        )

        # Should have attempted show command for initial commit
        mock_repo.git.show.assert_called()

    @patch("app.main.shutil.rmtree")
    @patch("app.main.tempfile.mkdtemp")
    @patch("app.main.Repo.clone_from")
    def test_clone_non_github_url(self, mock_clone, mock_mkdtemp, mock_rmtree):
        """Test with non-GitHub URL (no token injection)."""
        mock_mkdtemp.return_value = "/tmp/test_dir"

        mock_repo = MagicMock()
        mock_clone.return_value = mock_repo
        mock_repo.git.diff.return_value = "diff content"

        result = clone_repo_and_get_diff(
            "https://gitlab.com/test/repo",
            "test_token",
            "abc123"
        )

        # Should still work
        assert result == "diff content"

    @patch("app.main.shutil.rmtree")
    @patch("app.main.tempfile.mkdtemp")
    @patch("app.main.Repo.clone_from")
    def test_clone_cleanup_on_error(self, mock_clone, mock_mkdtemp, mock_rmtree):
        """Test that temp directory is cleaned up even on error."""
        mock_mkdtemp.return_value = "/tmp/test_dir"
        mock_clone.side_effect = GitCommandError("clone", "failed")

        with pytest.raises(GitCommandError):
            clone_repo_and_get_diff(
                "https://github.com/test/repo",
                "test_token",
                "abc123"
            )

        # rmtree should still be called for cleanup
        mock_rmtree.assert_called_with("/tmp/test_dir", ignore_errors=True)


# ==============================================================================
# Test Inference
# ==============================================================================

class TestRunInference:
    """Tests for run_inference function."""

    def test_run_inference_model_not_loaded(self):
        """Test that run_inference raises error when model not loaded."""
        model_state.is_loaded = False

        with pytest.raises(RuntimeError, match="Model is not loaded"):
            run_inference("test prompt")

    def test_run_inference_success(self, mock_model_loaded):
        """Test successful inference."""
        fake_model, fake_tokenizer = mock_model_loaded

        # Set up return values
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        fake_tokenizer.return_value = mock_inputs
        fake_model.generate.return_value = [[1, 2, 3]]
        fake_tokenizer.batch_decode.return_value = ["Generated response"]

        result = run_inference("test prompt")

        assert result == "Generated response"
        fake_tokenizer.assert_called()
        fake_model.generate.assert_called()


# ==============================================================================
# Test TMS Callback
# ==============================================================================

class TestSendCallbackToTMS:
    """Tests for send_callback_to_tms function."""

    @pytest.mark.asyncio
    async def test_send_callback_success(self):
        """Test successful callback to TMS."""
        payload = CallbackPayload(
            tms_project_id="project_1",
            status="completed",
            analysis_result=CodeAnalysisResult(code_quality_score=85)
        )

        with patch("app.main.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()

            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            await send_callback_to_tms(payload)

            mock_instance.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_callback_http_error(self):
        """Test callback handling when HTTP error occurs."""
        import httpx

        payload = CallbackPayload(
            tms_project_id="project_1",
            status="completed"
        )

        with patch("app.main.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.side_effect = httpx.HTTPError("Connection failed")
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            # Should not raise, just log
            await send_callback_to_tms(payload)

    @pytest.mark.asyncio
    async def test_send_callback_unexpected_error(self):
        """Test callback handling when unexpected error occurs."""
        payload = CallbackPayload(
            tms_project_id="project_1",
            status="failed",
            error="Some error"
        )

        with patch("app.main.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.side_effect = Exception("Unexpected error")
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            # Should not raise, just log
            await send_callback_to_tms(payload)


# ==============================================================================
# Test Background Analysis Task
# ==============================================================================

class TestPerformAnalysis:
    """Tests for perform_analysis background task."""

    @pytest.mark.asyncio
    async def test_perform_analysis_success_mock_mode(self):
        """Test successful analysis in mock mode."""
        request = AnalyzeRequest(
            repo_url="https://github.com/test/repo",
            commit_sha="abc123",
            github_token="test_token",
            tms_project_id="project_1"
        )

        with patch("app.main.run_in_threadpool") as mock_threadpool:
            mock_threadpool.return_value = "diff --git a/file.py\n+code"

            with patch("app.main.send_callback_to_tms", new_callable=AsyncMock) as mock_callback:
                await perform_analysis(request)

                mock_callback.assert_called_once()
                call_args = mock_callback.call_args[0][0]
                assert call_args.status == "completed"
                assert call_args.tms_project_id == "project_1"

    @pytest.mark.asyncio
    async def test_perform_analysis_empty_diff(self):
        """Test analysis when diff is empty."""
        request = AnalyzeRequest(
            repo_url="https://github.com/test/repo",
            commit_sha="abc123",
            github_token="test_token",
            tms_project_id="project_1"
        )

        with patch("app.main.run_in_threadpool") as mock_threadpool:
            mock_threadpool.return_value = "   "  # Empty/whitespace diff

            with patch("app.main.send_callback_to_tms", new_callable=AsyncMock) as mock_callback:
                await perform_analysis(request)

                mock_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_perform_analysis_with_real_model(self):
        """Test analysis with real model inference (non-mock mode)."""
        request = AnalyzeRequest(
            repo_url="https://github.com/test/repo",
            commit_sha="abc123",
            github_token="test_token",
            tms_project_id="project_1"
        )

        with patch.object(settings, 'mock_inference', False):
            with patch("app.main.run_in_threadpool") as mock_threadpool:
                # First call for clone_repo_and_get_diff, second for run_inference
                mock_threadpool.side_effect = [
                    "diff content",
                    '### Response:\n{"code_quality_score": 90}'
                ]

                with patch("app.main.send_callback_to_tms", new_callable=AsyncMock) as mock_callback:
                    await perform_analysis(request)

                    mock_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_perform_analysis_git_error(self):
        """Test analysis when git operation fails."""
        request = AnalyzeRequest(
            repo_url="https://github.com/test/repo",
            commit_sha="abc123",
            github_token="test_token",
            tms_project_id="project_1"
        )

        with patch("app.main.run_in_threadpool") as mock_threadpool:
            mock_threadpool.side_effect = GitCommandError("clone", "Repository not found")

            with patch("app.main.send_callback_to_tms", new_callable=AsyncMock) as mock_callback:
                await perform_analysis(request)

                mock_callback.assert_called_once()
                call_args = mock_callback.call_args[0][0]
                assert call_args.status == "failed"
                assert "Git operation failed" in call_args.error

    @pytest.mark.asyncio
    async def test_perform_analysis_generic_error(self):
        """Test analysis when generic exception occurs."""
        request = AnalyzeRequest(
            repo_url="https://github.com/test/repo",
            commit_sha="abc123",
            github_token="test_token",
            tms_project_id="project_1"
        )

        with patch("app.main.run_in_threadpool") as mock_threadpool:
            mock_threadpool.side_effect = Exception("Something went wrong")

            with patch("app.main.send_callback_to_tms", new_callable=AsyncMock) as mock_callback:
                await perform_analysis(request)

                mock_callback.assert_called_once()
                call_args = mock_callback.call_args[0][0]
                assert call_args.status == "failed"
                assert "Something went wrong" in call_args.error

    @pytest.mark.asyncio
    async def test_perform_analysis_long_diff_preview(self):
        """Test that diff preview is truncated for long diffs."""
        request = AnalyzeRequest(
            repo_url="https://github.com/test/repo",
            commit_sha="abc123",
            github_token="test_token",
            tms_project_id="project_1"
        )

        long_diff = "x" * 1000

        with patch("app.main.run_in_threadpool") as mock_threadpool:
            mock_threadpool.return_value = long_diff

            with patch("app.main.send_callback_to_tms", new_callable=AsyncMock) as mock_callback:
                await perform_analysis(request)

                call_args = mock_callback.call_args[0][0]
                # diff_preview should be truncated to 500 chars + "..."
                assert len(call_args.diff_preview) <= 504


# ==============================================================================
# Test Lifespan
# ==============================================================================

class TestLifespan:
    """Tests for lifespan context manager."""

    @pytest.mark.asyncio
    async def test_lifespan_mock_mode(self):
        """Test lifespan in mock inference mode."""
        with patch.object(settings, 'mock_inference', True):
            async with lifespan(app):
                # In mock mode, model should not be loaded
                assert model_state.is_loaded == False

    @pytest.mark.asyncio
    async def test_lifespan_real_mode_success(self):
        """Test lifespan with real model loading (mocked)."""
        # Reset state before test
        model_state.model = None
        model_state.tokenizer = None
        model_state.is_loaded = False

        with patch.object(settings, 'mock_inference', False):
            # The FastLanguageModel is imported from unsloth inside lifespan
            # So we need to ensure our mocked unsloth module is used
            mock_flm = MagicMock()
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
            mock_flm.for_inference = MagicMock()
            sys.modules["unsloth"].FastLanguageModel = mock_flm

            async with lifespan(app):
                # Model should be loaded
                assert model_state.is_loaded == True
                assert model_state.model is not None
                assert model_state.tokenizer is not None

            # After context, model should be unloaded
            assert model_state.is_loaded == False
            assert model_state.model is None

    @pytest.mark.asyncio
    async def test_lifespan_model_load_failure(self):
        """Test lifespan when model loading fails."""
        with patch.object(settings, 'mock_inference', False):
            with patch.dict(sys.modules, {"unsloth": MagicMock()}):
                # Make FastLanguageModel.from_pretrained raise an error
                mock_flm = MagicMock()
                mock_flm.from_pretrained.side_effect = Exception("GPU not available")
                sys.modules["unsloth"].FastLanguageModel = mock_flm

                with pytest.raises(RuntimeError, match="Failed to load model"):
                    async with lifespan(app):
                        pass


# ==============================================================================
# Test Pydantic Models
# ==============================================================================

class TestPydanticModels:
    """Tests for Pydantic model schemas."""

    def test_code_analysis_result_defaults(self):
        """Test CodeAnalysisResult default values."""
        result = CodeAnalysisResult()

        assert result.code_quality_score == 0
        assert result.critical_issues == []
        assert result.suggestions == []
        assert result.fixed_code == ""

    def test_code_analysis_result_validation(self):
        """Test CodeAnalysisResult validation."""
        result = CodeAnalysisResult(
            code_quality_score=100,
            critical_issues=["issue"],
            suggestions=["suggestion"],
            fixed_code="fixed"
        )

        assert result.code_quality_score == 100

    def test_callback_payload_with_result(self):
        """Test CallbackPayload with analysis result."""
        result = CodeAnalysisResult(code_quality_score=85)
        payload = CallbackPayload(
            tms_project_id="proj1",
            status="completed",
            analysis_result=result
        )

        assert payload.tms_project_id == "proj1"
        assert payload.analysis_result.code_quality_score == 85

    def test_callback_payload_with_error(self):
        """Test CallbackPayload with error."""
        payload = CallbackPayload(
            tms_project_id="proj1",
            status="failed",
            error="Something failed"
        )

        assert payload.error == "Something failed"
        assert payload.analysis_result is None

    def test_settings_defaults(self):
        """Test Settings default values."""
        # Create fresh settings to test defaults
        with patch.dict(os.environ, {}, clear=False):
            test_settings = Settings()
            assert test_settings.port == 5002
            assert test_settings.max_seq_length == 2048
            assert test_settings.max_new_tokens == 512

    def test_model_state_initial(self):
        """Test ModelState initial values."""
        state = ModelState()
        assert state.model is None
        assert state.tokenizer is None
        assert state.is_loaded == False


# ==============================================================================
# Test Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_parse_response_with_nested_json(self):
        """Test parsing response with nested structures."""
        response = '''### Response:
        {
            "code_quality_score": 75,
            "critical_issues": ["issue1", "issue2"],
            "suggestions": ["sug1"],
            "fixed_code": "def fixed():\\n    pass"
        }'''

        result = parse_model_response(response)
        assert result.code_quality_score == 75
        assert len(result.critical_issues) == 2

    def test_parse_response_with_extra_text(self):
        """Test parsing when there's extra text around JSON."""
        response = '''### Response:
        Here is my analysis:
        {"code_quality_score": 60}
        That's my response.'''

        result = parse_model_response(response)
        assert result.code_quality_score == 60

    def test_format_prompt_with_special_chars(self):
        """Test prompt formatting with special characters."""
        diff = 'def test():\n    print("Hello\\nWorld")\n    x = {"key": "value"}'
        prompt = format_alpaca_prompt(diff)

        assert diff in prompt

    @patch("app.main.shutil.rmtree")
    @patch("app.main.tempfile.mkdtemp")
    @patch("app.main.Repo.clone_from")
    def test_clone_with_github_url_token_injection(self, mock_clone, mock_mkdtemp, mock_rmtree):
        """Test that token is properly injected into GitHub URLs."""
        mock_mkdtemp.return_value = "/tmp/test"
        mock_repo = MagicMock()
        mock_clone.return_value = mock_repo
        mock_repo.git.diff.return_value = "diff"

        clone_repo_and_get_diff(
            "https://github.com/owner/repo",
            "my_secret_token",
            "sha123"
        )

        # Verify the URL was called with token injected
        called_url = mock_clone.call_args[0][0]
        assert "my_secret_token@github.com" in called_url
