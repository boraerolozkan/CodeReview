#!/usr/bin/env python3
"""
Run the CodeReview Agent service.

Usage:
    python run_agent.py              # Run with model loading
    python run_agent.py --mock       # Run in mock mode (no GPU required)
    python run_agent.py --port 5003  # Run on custom port
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Run CodeReview Agent Service")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (skip model loading)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5002,
        help="Port to run the server on (default: 5002)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    # Set environment variables
    if args.mock:
        os.environ["MOCK_INFERENCE"] = "True"
        print("Running in MOCK mode - model will NOT be loaded")

    os.environ["PORT"] = str(args.port)

    # Import and run uvicorn
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
