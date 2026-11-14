"""
add_teammate.py - give a teammate their own MCP token (docs/CONTRACT.md "#36 change").

Run on the app host, from the repo root, inside .venv:

    python scripts/add_teammate.py EMAIL [--host IP] [--port 8000]   add (or rotate) a teammate
    python scripts/add_teammate.py --remove EMAIL                     revoke at once
    python scripts/add_teammate.py --list                             emails + added_at only

The users file (AUTOLABEL_MCP_USERS_FILE in .env, default data/mcp_users.json) keeps only the
sha256 of each token. The new token is printed ONCE to stdout, with the Claude Desktop config
block and the Claude Code command; it cannot be shown again (run the script again for a new one,
which also stops the old one). The running app picks up changes without a restart.

Exit codes: 0 ok, 1 --remove of an unknown email, 2 bad email or arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.assistant.users import add_user, clean_email, list_users, remove_user  # noqa: E402

ENV_FILE = ROOT / ".env"
DEFAULT_USERS_FILE = ROOT / "data" / "mcp_users.json"


def users_file(env_file: Path | None = None) -> Path:
    """AUTOLABEL_MCP_USERS_FILE (environment wins over .env), else data/mcp_users.json."""
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file or ENV_FILE, override=False)
    except ImportError:
        pass
    raw = (os.environ.get("AUTOLABEL_MCP_USERS_FILE") or "").strip()
    return Path(raw) if raw else DEFAULT_USERS_FILE


def lan_ip() -> str | None:
    """This PC's LAN address: a UDP "connect" picks the outgoing interface; nothing is sent."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            ip = s.getsockname()[0]
    except OSError:
        return None
    return ip if ip and not ip.startswith("0.") else None


def instructions(email: str, token: str, host: str, port: int) -> str:
    url = f"http://{host}:{port}/mcp/"
    desktop = {"mcpServers": {"autolabel": {
        "command": "python",
        "args": ["C:\\path\\to\\mcp_stdio.py", "--url", url],
        "env": {"AUTOLABEL_MCP_TOKEN": token}}}}
    return "\n".join([
        f"Teammate: {email}",
        "",
        f"Token (shown once, not recoverable): {token}",
        "",
        "Claude Desktop (claude_desktop_config.json; copy app/assistant/mcp_stdio.py to their PC,",
        "pip install mcp==2.3.0, fix the path below):",
        json.dumps(desktop, indent=2),
        "",
        "Claude Code:",
        f'claude mcp add --transport http autolabel {url} --header "Authorization: Bearer {token}"',
        "",
        f"Send this privately; anyone with it can use the app as {email}.",
    ])


def main(argv: list[str] | None = None, env_file: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description="add, remove or list MCP teammates")
    parser.add_argument("email", nargs="?", help="teammate email to add (or give a new token)")
    parser.add_argument("--host", default=None, help="the app host's LAN IP (default: detected)")
    parser.add_argument("--port", type=int, default=8000)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--remove", metavar="EMAIL", help="revoke this teammate's token")
    group.add_argument("--list", action="store_true", help="list emails and added_at")
    args = parser.parse_args(argv)
    path = users_file(env_file)

    if args.list:
        if args.email:
            parser.error("--list takes no email")
        for u in list_users(path):
            print(f"{u['email']}\t{u['added_at']}")
        return 0
    if args.remove:
        if args.email:
            parser.error("give the email once, after --remove")
        email = clean_email(args.remove)
        if email is None:
            print(f"error: not a valid email: {args.remove}", file=sys.stderr)
            return 2
        if not remove_user(path, email):
            print(f"error: {email} is not in {path}", file=sys.stderr)
            return 1
        print(f"removed {email}; their token stops working now")
        return 0
    if not args.email:
        parser.error("give an EMAIL, --remove EMAIL or --list")
    email = clean_email(args.email)
    if email is None:
        print(f"error: not a valid email: {args.email}", file=sys.stderr)
        return 2
    host = args.host
    if not host:
        host = lan_ip()
        if host is None:
            host = "127.0.0.1"
            print("warning: could not find this PC's LAN IP; using 127.0.0.1 (pass --host IP)",
                  file=sys.stderr)
    token = add_user(path, email)
    print(instructions(email, token, host, args.port))
    return 0


if __name__ == "__main__":
    sys.exit(main())
