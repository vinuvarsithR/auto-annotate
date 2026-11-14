"""scripts/add_teammate.py: stores the hash (never the token), prints the token once with the
Claude Desktop / Claude Code config for IP:PORT, re-add rotates, --remove, --list, bad email -> 2.
Every test uses a temp users file (AUTOLABEL_MCP_USERS_FILE) and no real .env."""

import importlib.util
import json
from pathlib import Path

import pytest

from app.assistant import users as users_mod

SCRIPT = Path(__file__).parents[1] / "scripts" / "add_teammate.py"
spec = importlib.util.spec_from_file_location("add_teammate", SCRIPT)
add_teammate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(add_teammate)


@pytest.fixture
def users_file(tmp_path, monkeypatch):
    f = tmp_path / "data" / "mcp_users.json"
    monkeypatch.setenv("AUTOLABEL_MCP_USERS_FILE", str(f))
    return f


def run(argv, tmp_path):
    return add_teammate.main(argv, env_file=tmp_path / "no-such.env")


def test_add_prints_the_token_once_and_stores_its_hash(users_file, tmp_path, capsys):
    assert run(["Dhyan@Example.com", "--host", "192.168.1.10", "--port", "8000"], tmp_path) == 0
    out = capsys.readouterr().out
    entry = json.loads(users_file.read_text(encoding="utf-8"))["users"][0]
    assert entry["email"] == "dhyan@example.com"
    token = next(w for w in out.split() if w.startswith("alu_"))
    assert entry["token_sha256"] == users_mod.token_hash(token)
    assert token not in users_file.read_text(encoding="utf-8")
    assert out.count(token) == 3  # the token line, the Desktop env, the Claude Code header
    assert "http://192.168.1.10:8000/mcp/" in out
    desktop = json.loads(out[out.index("{"):out.index("\n}\n") + 2])
    server = desktop["mcpServers"]["autolabel"]
    assert server["env"] == {"AUTOLABEL_MCP_TOKEN": token}
    assert "--token" not in server["args"] and "--url" in server["args"]
    assert "claude mcp add" in out
    assert "send this privately; anyone with it can use the app as dhyan@example.com" in out.lower()


def test_re_add_rotates(users_file, tmp_path, capsys):
    run(["a@example.com", "--host", "10.0.0.1"], tmp_path)
    first = users_mod.UsersFile(users_file).users()[0]["token_sha256"]
    run(["a@example.com", "--host", "10.0.0.1"], tmp_path)
    users = users_mod.UsersFile(users_file).users()
    assert len(users) == 1 and users[0]["token_sha256"] != first


def test_remove_and_list(users_file, tmp_path, capsys):
    run(["a@example.com", "--host", "10.0.0.1"], tmp_path)
    run(["b@example.com", "--host", "10.0.0.1"], tmp_path)
    capsys.readouterr()
    assert run(["--list"], tmp_path) == 0
    listed = capsys.readouterr().out
    assert "a@example.com" in listed and "b@example.com" in listed
    assert "alu_" not in listed and "sha256" not in listed
    assert run(["--remove", "A@example.com"], tmp_path) == 0
    assert [u["email"] for u in users_mod.list_users(users_file)] == ["b@example.com"]
    assert run(["--remove", "a@example.com"], tmp_path) == 1


@pytest.mark.parametrize("email", ["nope", "a@b", "a@@b.com", "@b.com", "a b@c.com",
                                   "a@" + "b" * 250 + ".com", "a@.com", "a@com."])
def test_bad_email_exit_2(users_file, tmp_path, email):
    assert run([email, "--host", "10.0.0.1"], tmp_path) == 2
    assert not users_file.exists()


def test_lan_ip_fallback_warns(users_file, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(add_teammate, "lan_ip", lambda: None)
    assert run(["a@example.com"], tmp_path) == 0
    captured = capsys.readouterr()
    assert "http://127.0.0.1:8000/mcp/" in captured.out and "warning" in captured.err


def test_lan_ip_is_used_by_default(users_file, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(add_teammate, "lan_ip", lambda: "192.168.5.7")
    run(["a@example.com", "--port", "8123"], tmp_path)
    assert "http://192.168.5.7:8123/mcp/" in capsys.readouterr().out


def test_users_file_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTOLABEL_MCP_USERS_FILE", raising=False)
    env = tmp_path / "test.env"
    target = tmp_path / "from_env.json"
    env.write_text(f"AUTOLABEL_MCP_USERS_FILE={target}\n", encoding="utf-8")
    try:
        assert add_teammate.users_file(env) == target
    finally:
        monkeypatch.delenv("AUTOLABEL_MCP_USERS_FILE", raising=False)  # load_dotenv set it


def test_default_users_file_is_data_mcp_users(monkeypatch, tmp_path):
    monkeypatch.delenv("AUTOLABEL_MCP_USERS_FILE", raising=False)
    assert add_teammate.users_file(tmp_path / "none.env") == add_teammate.ROOT / "data" / "mcp_users.json"
