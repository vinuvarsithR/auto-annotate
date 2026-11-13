"""Claude connector (docs/CONTRACT.md "Claude connector", #31 - #36).

* tools.py       - the one tool layer (17 tools) both front-ends use
* mcp_server.py  - MCP server at /mcp/ (teammates' own Claude Code / Claude Desktop) + token gate
* mcp_stdio.py   - standalone stdio launcher for Claude Desktop (forwards to /mcp/)
* chat.py        - sidebar chat: the Claude tool-use loop with the server's ANTHROPIC_API_KEY
* routes.py      - #31 - #35 under /api/assistant
* settings.py    - .env settings (secrets are never logged or returned)
"""
