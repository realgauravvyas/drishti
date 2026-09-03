#!/usr/bin/env python
"""Start the server.

    python run.py                 # http://127.0.0.1:8000
    python run.py --host 0.0.0.0  # reachable from phones on the same Wi-Fi
"""

from __future__ import annotations

import argparse

import uvicorn

from smriti.config import get_settings


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the Smriti server")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    parser.add_argument("--workers", type=int, default=1,
                        help="uvicorn processes; keep at 1 unless you have a shared DB")
    args = parser.parse_args()

    if args.host == "0.0.0.0":
        print("\n  Serving on your local network. Guests can reach it at:")
        print(f"    http://{_lan_ip()}:{args.port}/\n")

    uvicorn.run(
        "smriti.app:app",
        host=args.host, port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


def _lan_ip() -> str:
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))  # no packet is sent; this just picks a route
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


if __name__ == "__main__":
    main()
