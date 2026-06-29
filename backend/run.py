#!/usr/bin/env python3
"""
Entry point for running the AI Candidate Recommendation Engine.

Usage:
    python run.py              # Start server
    python run.py --reload     # Start with auto-reload
    python run.py --help       # Show help
"""

import argparse
import uvicorn
from app.core.config import settings


def main():
    parser = argparse.ArgumentParser(
        description="AI Candidate Recommendation Engine"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=settings.host,
        help=f"Host to bind to (default: {settings.host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help=f"Port to bind to (default: {settings.port})",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development mode)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )

    args = parser.parse_args()

    print(f"Starting AI Candidate Recommendation Engine...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    print(f"Workers: {args.workers}")
    print(f"\nAPI Documentation: http://{args.host}:{args.port}/docs")
    print(f"Health Check: http://{args.host}:{args.port}/api/health")
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
