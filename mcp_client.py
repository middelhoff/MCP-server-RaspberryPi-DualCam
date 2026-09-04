#!/usr/bin/env python3
"""Small MCP client for poking the Pi hardware server from a shell.

    ~/stereo-mcp/venv/bin/python mcp_client.py list
    ~/stereo-mcp/venv/bin/python mcp_client.py call ultraschall_abstand
    ~/stereo-mcp/venv/bin/python mcp_client.py call weg_frei mindestabstand_cm=80
    ~/stereo-mcp/venv/bin/python mcp_client.py call calibrate_stereo left_camera=1

Why this rather than curl: the streamable-HTTP transport needs a real MCP
initialize handshake and session handling, which is awkward by hand.

Two things that cost time when writing this, worth remembering:

* In mcp 2.x the client factory is `streamable_http_client`, NOT
  `streamablehttp_client`. The old name is gone.
* The server's DNS-rebinding guard only accepts Host headers on its allowlist.
  localhost and the Pi's LAN IP are both allowed; a hostname that is not will be
  rejected with a 421 even though the port is open.

Note the ultrasonic tools cannot be exercised by a second process that opens the
GPIO itself - the running service holds GPIO23/24 exclusively via lgpio, so a
parallel script gets "GPIO busy". Going through the MCP endpoint like this is
the way to test them while the service is up.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

DEFAULT_URL = "http://192.168.178.35:8000/mcp"


def parse_value(raw: str):
    """Turn a key=value argument into a JSON-ish Python value."""
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


async def run(url: str, action: str, tool: str | None, params: dict) -> int:
    async with streamable_http_client(url) as ctx:
        read_stream, write_stream = ctx[0], ctx[1]
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            if action == "list":
                tools = await session.list_tools()
                print(f"{len(tools.tools)} tools on {url}\n")
                for t in tools.tools:
                    summary = (t.description or "").strip().splitlines()
                    print(f"  {t.name}")
                    if summary:
                        print(f"      {summary[0]}")
                return 0

            result = await session.call_tool(tool, params)
            for item in result.content:
                # Images come back as content blocks without text; don't try to
                # print their payload into a terminal.
                text = getattr(item, "text", None)
                if text is not None:
                    print(text)
                else:
                    kind = getattr(item, "type", type(item).__name__)
                    data = getattr(item, "data", b"") or b""
                    print(f"<{kind} content, {len(data)} bytes - not printed>")
            return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["list", "call"])
    p.add_argument("tool", nargs="?", help="tool name, for 'call'")
    p.add_argument("params", nargs="*", metavar="key=value")
    p.add_argument("--url", default=DEFAULT_URL)
    args = p.parse_args()

    if args.action == "call" and not args.tool:
        p.error("'call' needs a tool name")

    params = {}
    for item in args.params:
        if "=" not in item:
            p.error(f"parameter {item!r} is not in key=value form")
        key, _, raw = item.partition("=")
        params[key] = parse_value(raw)

    return asyncio.run(run(args.url, args.action, args.tool, params))


if __name__ == "__main__":
    sys.exit(main())
