import os
import sys

from .cli import main
from .render import perf


def _run() -> int:
    """Run the render. A render that was drawing with Metal (R3D_STD_METAL=1,
    macOS) and failed is run again on OpenGL in a fresh process, so the Metal
    path can never lose a job: the same rule as catch's Metal backend.

    Setting Metal up is already guarded inside the CLI (it falls back to GL in
    this process); this covers everything after that. Only a failure counts: a
    cancelled job (SIGTERM, Ctrl-C) is not re-run."""
    try:
        rc = main()
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (1 if e.code else 0)
    except Exception as e:  # noqa: BLE001 - never let Metal lose a render
        if not perf.METAL_IN_USE:
            raise
        print(f"[std] Metal render raised {e!r}", file=sys.stderr, flush=True)
        rc = 1
    if rc and perf.METAL_IN_USE:
        print(f"[std] Metal render failed (rc={rc}) -> re-running on OpenGL",
              file=sys.stderr, flush=True)
        env = dict(os.environ, R3D_STD_METAL="0")
        os.execve(sys.executable,
                  [sys.executable, "-m", "osu_std_renderer"] + sys.argv[1:], env)
    return rc or 0


raise SystemExit(_run())
