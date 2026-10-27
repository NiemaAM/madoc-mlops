"""The server's resources (GPU, CPU, memory, disk) and the GPU shared with MaDoc."""

import json
import shutil
import urllib.request

from mlops.config import params, path


def gpu_stats():
    """[{name, utilization %, memory used / total MiB, temperature °C}] with NVML; [] without a GPU."""
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception:
        return []
    out = []
    try:
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            out.append({
                "name": name.decode() if isinstance(name, bytes) else name,
                "utilization": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                "memory_used_mib": memory.used // 2 ** 20, "memory_total_mib": memory.total // 2 ** 20,
                "temperature": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
            })
    finally:
        pynvml.nvmlShutdown()
    return out


def host_stats():
    import psutil
    data = shutil.disk_usage(path("data"))
    memory = psutil.virtual_memory()
    return {"cpu_percent": psutil.cpu_percent(interval=0.3), "memory_used_gib": round(memory.used / 2 ** 30, 1),
            "memory_total_gib": round(memory.total / 2 ** 30, 1),
            "data_disk_used_gib": round(data.used / 2 ** 30, 1), "data_disk_total_gib": round(data.total / 2 ** 30, 1)}


def gpu_name():
    stats = gpu_stats()
    return stats[0]["name"] if stats else "none"


def free_gpu(log=print):
    """Unload MaDoc's Ollama models (keep_alive 0): a training needs the GPU memory. MaDoc keeps working, Ollama
    loads the model again at its next request."""
    settings = params()["gpu"]
    for model in settings.get("unload_models") or []:
        body = json.dumps({"model": model, "keep_alive": 0}).encode()
        request = urllib.request.Request(settings["ollama_url"].rstrip("/") + "/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(request, timeout=30).read()
            log(f"Ollama: {model} unloaded from the GPU")
        except Exception as e:
            log(f"Ollama: {model} not unloaded ({e})")
