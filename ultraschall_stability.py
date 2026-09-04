#!/usr/bin/env python3
"""Sample the ultrasonic sensor repeatedly and report how steady it is.

    ~/stereo-mcp/venv/bin/python ultraschall_stability.py --samples 15

Use this to tell a SENSOR problem from a SCENE problem, which is the mistake
that is easy to make here. On 2026-09-03 a single reading showed 22 cm of spread
and the first guess was CPU contention from the MJPEG streamer inflating the
busy-wait timing. A run of this script disproved that: 15/15 readings came back
reliable with a median spread of 0.4 cm while the streamer was running. The
earlier spread had simply come from something moving in the sensor's ~15 degree
cone.

So: high spread that PERSISTS across a run points at wiring or timing; high
spread that comes and goes is the scene, and the `zuverlaessig` flag is already
reporting it correctly.

Goes through the MCP endpoint on purpose - the running service holds GPIO23/24
exclusively, so a script that opens the pins itself would fail with "GPIO busy".
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

DEFAULT_URL = "http://192.168.178.35:8000/mcp"

# The server flags a reading unreliable above this spread. Kept here only to
# report the margin - the authoritative value lives in server.py.
RELIABLE_SPREAD_CM = 5.0


async def run(url: str, samples: int, interval: float) -> int:
    distances: list[float] = []
    spreads: list[float] = []
    errors = 0

    async with streamable_http_client(url) as ctx:
        async with ClientSession(ctx[0], ctx[1]) as session:
            await session.initialize()

            print(f"{'#':>3}  {'abstand_cm':>10}  {'streuung_cm':>11}  zuverlaessig")
            for i in range(1, samples + 1):
                result = await session.call_tool("ultraschall_abstand", {})
                payload = json.loads(result.content[0].text)

                if "fehler" in payload:
                    errors += 1
                    print(f"{i:>3}  ERROR: {payload['fehler']}")
                else:
                    distances.append(payload["abstand_cm"])
                    spreads.append(payload["streuung_cm"])
                    print(f"{i:>3}  {payload['abstand_cm']:>10.1f}"
                          f"  {payload['streuung_cm']:>11.1f}  {payload['zuverlaessig']}")

                if i < samples:
                    await asyncio.sleep(interval)

    if not distances:
        print("\nNo valid readings at all - check wiring before anything else.")
        return 1

    reliable = sum(1 for s in spreads if s < RELIABLE_SPREAD_CM)
    worst = max(spreads)

    print()
    print(f"distance   median {statistics.median(distances):.1f} cm, "
          f"range {max(distances) - min(distances):.1f} cm "
          f"(min {min(distances):.1f}, max {max(distances):.1f})")
    print(f"spread     median {statistics.median(spreads):.1f} cm, worst {worst:.1f} cm")
    print(f"reliable   {reliable} of {len(spreads)}"
          + (f", {errors} failed reading(s)" if errors else ""))

    print()
    if reliable == len(spreads) and worst < RELIABLE_SPREAD_CM * 0.6:
        print("Verdict: sensor is healthy and the scene is quiet.")
    elif reliable == len(spreads):
        print(f"Verdict: all readings passed, but the worst ({worst:.1f} cm) sits close to "
              f"the {RELIABLE_SPREAD_CM:.0f} cm threshold. Fine for reporting distance; "
              f"consider a tighter threshold before driving an actuator off this flag.")
    elif reliable >= len(spreads) * 0.6:
        print("Verdict: intermittent spread - most likely something moving in the "
              "~15 degree cone rather than a hardware fault. Re-run against a static scene.")
    else:
        print("Verdict: spread is persistent, not occasional. THIS is when to suspect "
              "hardware: ECHO sits on GPIO24 without a divider (5 V into a 3.3 V pin). "
              "Fallback pin is GPIO25. Also confirm pulses stay >=60 ms apart.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--interval", type=float, default=1.0,
                   help="seconds between calls (default 1.0)")
    p.add_argument("--url", default=DEFAULT_URL)
    args = p.parse_args()
    return asyncio.run(run(args.url, args.samples, args.interval))


if __name__ == "__main__":
    sys.exit(main())
