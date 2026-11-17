"""The control app of the platform (FastAPI + one static page): python -m mlops.web

    api.py       the routes (login, data, training, evaluation, experiments, jobs, the Editor, monitoring)
    views.py     what the pages show, read from the database and the data folder
    monitor.py   CPU, memory, disk, GPU and the queue over the last hour; /metrics for Prometheus
    static/      the interface: index.html, app.js (Alpine.js, in static/vendor), editor.js (the lines on the
                 scan), app.css (light and dark), theme.js

The app never trains anything: an action is a job of the queue, run by the worker. mlops/ui.py (Gradio) is the
previous app.
"""
