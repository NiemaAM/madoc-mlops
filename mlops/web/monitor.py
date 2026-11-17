"""The resources of the server over time: what the Monitoring page draws and /metrics gives to Prometheus.

One thread takes a sample every INTERVAL seconds (CPU, memory, data disk, every GPU through NVML, the jobs
queued and running, the workers) and keeps the last hour in memory. Nothing of the dataset is in a sample.
"""

import os
import threading
import time
from collections import deque

from mlops import db

INTERVAL = 5
samples = deque(maxlen=3600 // INTERVAL)
started = time.time()
_thread = []


def sample():
    from mlops.resources import gpu_stats, host_stats
    point = {"at": time.time(), "cpu": None, "memory_used": None, "memory_total": None, "process_memory": None,
             "disk_used": None, "disk_total": None, "gpus": [], "queued": None, "running": None, "running_job": None,
             "workers_alive": None}
    try:
        host = host_stats()
        point.update(cpu=host["cpu_percent"], memory_used=host["memory_used_gib"],
                     memory_total=host["memory_total_gib"], disk_used=host["data_disk_used_gib"],
                     disk_total=host["data_disk_total_gib"])
        import psutil
        point["process_memory"] = round(psutil.Process().memory_info().rss / 2 ** 30, 2)
    except Exception:
        pass
    try:
        point["gpus"] = [{"name": g["name"], "util": g["utilization"], "memory_used": g["memory_used_mib"] / 1024,
                          "memory_total": g["memory_total_mib"] / 1024, "temperature": g["temperature"]}
                         for g in gpu_stats()]
    except Exception:
        pass
    try:
        with db.connection() as conn:
            counts = db.one(conn, "SELECT count(*) FILTER (WHERE status = 'queued') AS queued, "
                                  "count(*) FILTER (WHERE status = 'running') AS running FROM jobs")
            job = db.one(conn, "SELECT id, kind, task FROM jobs WHERE status = 'running' ORDER BY id LIMIT 1")
            alive = db.one(conn, "SELECT count(*) AS n FROM workers WHERE seen_at > now() - interval '5 minutes'")
        point.update(queued=counts["queued"], running=counts["running"], workers_alive=alive["n"],
                     running_job=job and f'#{job["id"]} {job["kind"]}{" " + job["task"] if job["task"] else ""}')
    except Exception:
        pass
    return point


def _run():
    while True:
        try:
            samples.append(sample())
        except Exception:      # a failed sample never stops the next ones
            pass
        time.sleep(INTERVAL)


def start():
    if not _thread:
        _thread.append(threading.Thread(target=_run, daemon=True, name="monitor"))
        _thread[0].start()


def snapshot(minutes=15):
    since = time.time() - minutes * 60
    return {"now": time.time(), "interval": INTERVAL, "uptime": time.time() - started,
            "cpu_count": os.cpu_count() or 1, "samples": [p for p in list(samples) if p["at"] >= since]}


def prometheus(job_stats=()):
    """The last sample and the job counts in the text format of Prometheus."""
    point = samples[-1] if samples else sample()
    lines = [f"# TYPE mlops_uptime_seconds gauge\nmlops_uptime_seconds {time.time() - started:.0f}"]
    for name, key, scale in (("mlops_cpu_percent", "cpu", 1), ("mlops_memory_used_bytes", "memory_used", 2 ** 30),
                             ("mlops_memory_total_bytes", "memory_total", 2 ** 30),
                             ("mlops_app_memory_bytes", "process_memory", 2 ** 30),
                             ("mlops_data_disk_used_bytes", "disk_used", 2 ** 30),
                             ("mlops_data_disk_total_bytes", "disk_total", 2 ** 30),
                             ("mlops_jobs_queued", "queued", 1), ("mlops_jobs_running", "running", 1),
                             ("mlops_workers_alive", "workers_alive", 1)):
        if point.get(key) is not None:
            lines.append(f"# TYPE {name} gauge\n{name} {round(point[key] * scale, 3)}")
    for name, key, scale in (("mlops_gpu_utilization_percent", "util", 1),
                             ("mlops_gpu_memory_used_bytes", "memory_used", 2 ** 30),
                             ("mlops_gpu_memory_total_bytes", "memory_total", 2 ** 30),
                             ("mlops_gpu_temperature_celsius", "temperature", 1)):
        rows = [f'{name}{{gpu="{i}"}} {round(g[key] * scale, 3)}' for i, g in enumerate(point["gpus"])
                if g.get(key) is not None]
        if rows:
            lines.append(f"# TYPE {name} gauge\n" + "\n".join(rows))
    if job_stats:
        lines.append("# TYPE mlops_jobs_total gauge")
        for s in job_stats:
            labels = f'kind="{s["kind"]}",task="{s["task"] or ""}"'
            lines.append(f"mlops_jobs_total{{{labels},status=\"failed\"}} {s['failed']}\n"
                         f"mlops_jobs_total{{{labels},status=\"all\"}} {s['runs']}")
            if s["mean"] is not None:
                lines.append(f"mlops_job_mean_seconds{{{labels}}} {s['mean']:.0f}")
    return "\n".join(lines) + "\n"
