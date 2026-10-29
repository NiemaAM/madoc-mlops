"""Running a command of a job: its output goes to the job's log, a cancel request stops it."""

import os
import signal
import subprocess
import threading
import time


class Cancelled(Exception):
    """The person cancelled the job in the app."""


def run(cmd, log, should_stop=lambda: False, on_line=None, cwd=None, env=None):
    """Run `cmd`; every output line goes to `log` (and `on_line`); -> the whole output. Raises
    CalledProcessError when it fails, Cancelled when should_stop() becomes true (checked every 2 seconds)."""
    cmd = [str(c) for c in cmd]
    log("$ " + " ".join(cmd))
    # errors="replace": a byte that is not UTF-8 in the output (a file name, a library's message) cannot stop
    # the job
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                               errors="replace", cwd=cwd, env={**os.environ, **(env or {})}, bufsize=1,
                               start_new_session=True)
    stopped = threading.Event()

    def asked_to_stop():
        try:
            return should_stop()
        except Exception as e:            # the database unreachable for a moment: ask again 2 seconds later
            log(f"could not check for a cancel request: {e}")
            return False

    def watch():
        while process.poll() is None:
            if asked_to_stop():
                stopped.set()
                os.killpg(process.pid, signal.SIGTERM)   # ketos and its data loader processes
                for _ in range(20):
                    if process.poll() is not None:
                        return
                    time.sleep(0.5)
                os.killpg(process.pid, signal.SIGKILL)
                return
            time.sleep(2)

    threading.Thread(target=watch, daemon=True).start()
    lines = []
    for line in process.stdout:
        lines.append(line)
        log(line)
        if on_line:
            on_line(line)
    code = process.wait()
    if stopped.is_set():
        raise Cancelled("cancelled in the app")
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd, "".join(lines[-40:]))
    return "".join(lines)