# MaDoc MLOps

The platform that improves MaDoc's kraken models (segmentation `blla`, recognition `htr`): upload the pages
annotated in eScriptorium, train, evaluate the new model against the one in production, and propose the better
one to MaDoc as a pull request. Everything runs on the GPU machine; the pages never leave it.

```
 eScriptorium export (zip)                                                      MaDoc repository
        │ upload in the app                                                            ▲
        ▼                                                                              │ pull request
 ┌──────────────┐  jobs  ┌────────────────────────────────────────────────────┐   ┌────┴──────────┐
 │  mlops-ui    │──────▶ │  mlops-worker (GPU)                                 │──▶│ merge = deploy │
 │  (Gradio)    │◀────── │  ingest → train → evaluate → release               │   └───────────────┘
 └──────┬───────┘ status └───────┬───────────────────────────┬────────────────┘
        │                        │ metrics, models           │ unload Gemma before a job
        ▼                        ▼                           ▼
 ┌──────────────┐        Weights & Biases            madoc-ollama (MaDoc's compose)
 │  db          │        (no page, no file name)
 │  PostgreSQL  │
 └──────────────┘
```

| Container | Project | What it does | GPU | Reachable from |
|---|---|---|---|---|
| `madoc` | MaDoc | the web app (kraken, NER) | yes | the public link (Tailscale funnel) |
| `madoc-ollama` | MaDoc | Ollama + Gemma (spelling, translation) | yes | `madoc`; the worker, to unload Gemma |
| `mlops-ui` | this repository | upload, buttons, jobs and live logs, comparisons, server resources | NVML only | your tailnet only (`tailscale serve`), with a login |
| `mlops-worker` | this repository | runs the jobs one at a time: ingestion, training, evaluation, pull request | yes | nothing (no port) |
| `db` | this repository | PostgreSQL: pages, dataset versions, jobs and logs, models, evaluations | no | the two containers above only |

The two projects are separate Docker Compose projects: deploying MaDoc never restarts a training, and the only
links between them are the shared GPU, the `madoc-shared` network (to unload Gemma) and the pull requests.

**What each step does**

| Step | Details |
|---|---|
| Ingest | Every page is checked (image, size, lines, baselines, transcription). It is renamed to an opaque id (start of the SHA-256 of its image): the original file name, which identifies a real record, is only in the private table `page_origins`. This is pseudonymisation, not anonymisation: the images and the texts still show names and dates, so the data stays personal data and stays on this machine. |
| Split | By group: the pages of the same register and year (from the original file name, `params.yaml: ingest.group_pattern`) land in the same split, decided by a keyed hash. A test page is never a near-copy of a train page; a group keeps its split forever. 70 / 15 / 15. |
| Dataset version | When a job starts, the pages and their splits are frozen into a version (`dataset_versions`). A model records the version it was trained on, an evaluation the version of its test pages. |
| Augmentation | `script/data_augmentation_xml.py` (augraphy), train pages only, made once per version and reused. |
| Train | Fine-tunes MaDoc's production model with ketos until early stopping; metrics live in W&B; the best checkpoint is exported and registered in W&B (alias `candidate`) and in the database. |
| Evaluate | The candidate and the production model on the same test pages (old and new): the ketos numbers, and a number per page (HTR: CER with the true lines; blla: segmentation + production HTR, words found). The gain per page and its 95 % confidence interval (bootstrap). The gate (`params.yaml: gate`) says "better" only when the whole interval is above zero and the model is not worse on the old test pages. |
| Release | One click opens the pull request on MaDoc: the model, `models/MODEL_CARD.json` (W&B version, dataset version, test metrics) and the comparison. You merge it, MaDoc's deploy workflow puts it in production. Reverting the pull request rolls it back. |

---

## Setup, step by step

Done once, on the GPU machine (Ubuntu), in this order. Commands with `sudo` need your password.

### 1. NVIDIA driver, Docker, NVIDIA Container Toolkit

```bash
nvidia-smi                       # the driver is installed if this lists the RTX 5060 Ti
```

Docker Engine (from Docker's repository; see https://docs.docker.com/engine/install/ubuntu/ if a command changed):

```bash
sudo apt-get update && sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER    # then log out and in again (or reboot)
```

The NVIDIA Container Toolkit, which gives containers the GPU (see
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html if a command changed):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all ubuntu nvidia-smi    # the GPU seen from a container: the same table as above
```

### 2. The network shared by the two projects

```bash
docker network create madoc-shared
```

### 3. MaDoc in containers

MaDoc's repository now has a `Dockerfile`, a `docker-compose.yml` and a `deploy.yml` that builds and restarts the
containers. Push those files to MaDoc first (the deploy will fail until steps 3.1 to 3.3 are done, that is
expected), then on the GPU machine:

```bash
# 3.1 stop what the containers replace
systemctl --user disable --now madoc          # the systemd service of the app
sudo systemctl disable --now ollama           # the Ollama of the host (the container replaces it)

# 3.2 the runner of MaDoc must be able to use Docker
sudo usermod -aG docker $USER
cd ~/Documents/MaDoc/actions-runner && sudo ./svc.sh stop && sudo ./svc.sh start

# 3.3 build and start MaDoc
cd ~/Documents/MaDoc
git pull
docker compose up -d --build                  # first build: 10-20 minutes (PyTorch)
docker compose exec madoc-ollama ollama pull gemma3:12b
docker compose ps                             # madoc and madoc-ollama "healthy" after a few minutes
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:7860/    # 200
```

The Tailscale funnel already points to `127.0.0.1:7860`: https://madoc.tail19f00f.ts.net/ works unchanged.
`~/Documents/MaDoc/.env` is still read (`MADOC_AUTH`, `HF_TOKEN`, …). Re-run the last failed `deploy` workflow in
GitHub (Actions → deploy → Re-run jobs): from now on every push to `master` rebuilds and restarts the containers.

### 4. This repository on GitHub and on the machine

1. On GitHub, create a **private** repository `madoc-mlops` and push these files to it.
2. On the GPU machine:
   ```bash
   cd ~ && git clone https://github.com/NiemaAM/madoc-mlops.git && cd madoc-mlops
   cp .env.example .env && chmod 600 .env
   openssl rand -hex 24      # run 3 times: for POSTGRES_PASSWORD, MLOPS_SECRET and the password of MLOPS_AUTH
   nano .env
   ```
3. Fill `.env`:

   | Variable | Value |
   |---|---|
   | `POSTGRES_PASSWORD` | a generated password (hexadecimal: no character that needs escaping) |
   | `MLOPS_AUTH` | `yourname:<generated password>` (several logins separated by commas) |
   | `MLOPS_SECRET` | a generated value. **Keep it**: it decides the split of the groups |
   | `WANDB_API_KEY` | from https://wandb.ai/authorize |
   | `GH_TOKEN` | GitHub → Settings → Developer settings → Fine-grained tokens → Generate: repository access **only NiemaAM/MaDoc**, permissions **Contents: Read and write**, **Pull requests: Read and write**. Note its expiry date |

### 5. Start the platform

```bash
cd ~/madoc-mlops
docker compose up -d --build
docker compose ps                    # db, mlops-ui, mlops-worker: "healthy" after 1-2 minutes
docker compose logs -f mlops-worker  # "worker gpu-worker ready" (Ctrl+C stops the view, not the worker)
```

### 6. Open the app (your tailnet only)

The app listens on `127.0.0.1:7870`. Publish it to your tailnet only (not the funnel: it stays on MaDoc):

```bash
sudo tailscale serve --bg --https=8443 http://127.0.0.1:7870
tailscale serve status
```

Open **https://madoc.tail19f00f.ts.net:8443** from a device of your tailnet and log in with `MLOPS_AUTH`.

### 7. Import the existing dataset

Your current pages are in the training repository (`dataset/birth_certificates_v2/{train,val,test}/base`, PageXML
next to the images). Two choices:

- **Split by group (recommended).** The old split was made page by page: pages of the same register are in train
  and in test, which makes the scores optimistic. Import everything without `--split`: the platform splits by
  register and year.
  ```bash
  cd ~/madoc-mlops
  for split in train val test; do
    docker compose run --rm -v ~/arabic-handwritten-document-extraction/dataset/birth_certificates_v2:/import:ro \
      mlops-ui python -m mlops.cli import /import/$split/base
  done
  ```
- **Keep the old split** (to compare with your previous W&B runs): the same with `--split $split` at the end.

The imports are queued: follow them in the **Jobs** tab, then check **Data** (pages and groups per split,
rejected pages and why). Do not import `train/all`: it holds your old augmented copies, the platform augments by
itself.

### 8. The first cycle

1. **Train**: "Train recognition (HTR)". The job appears in **Jobs**; its W&B link shows the curves live.
   A training takes hours; the app can be closed meanwhile.
2. **Evaluate**: choose the new model, "Evaluate against production". The current models of MaDoc have no card
   yet, so this first evaluation compares on all the test pages without the old / new breakdown.
3. **Compare & release**: choose the evaluation. If it says "Better", "Open a pull request on MaDoc".
4. On GitHub: MaDoc's tests run on the pull request; **merge it** to deploy the model.
5. Same for the segmentation ("Train segmentation (blla)").

---

## Daily use

1. Annotate in eScriptorium, export the document as **PageXML with images** (zip).
2. **Data** tab: drop the zip, "Ingest". The batch shows what was added, per split, and what was rejected.
3. **Train** the task that needs it (usually HTR after new transcriptions, blla after new layouts).
4. **Evaluate**, then **Compare & release**, then merge on GitHub.

What to watch: in **Compare & release**, the gain on the *old* test pages (the model must not forget) and on the
*new* ones (what the new data brought). In **Server**, the GPU memory during a training, and that the worker is
"alive".

## Operations

| Task | Command (in `~/madoc-mlops`) |
|---|---|
| state of the containers | `docker compose ps` |
| logs | `docker compose logs -f mlops-worker` (or `mlops-ui`, `db`) |
| the last jobs | `docker compose exec mlops-ui python -m mlops.cli jobs` |
| update the platform | check **Jobs** that nothing is running (a restarted worker marks its job failed), then `git pull && docker compose up -d --build` |
| stop / start | `docker compose stop` / `docker compose start` (the data stays in the volumes) |
| backup | `scripts/backup.sh /media/<encrypted-disk>/madoc-mlops-backup` (database dump + pages) |

**Backups.** The database and the `data` volume are the only copy of the dataset and of the private table of
original names. Back them up every night to an encrypted disk:

```bash
crontab -e
# 30 3 * * * cd /home/dil-station3/madoc-mlops && scripts/backup.sh /media/<encrypted-disk>/madoc-mlops-backup >> /home/dil-station3/mlops-backup.log 2>&1
```

Restore the database: `docker compose exec -T db pg_restore -U mlops -d mlops --clean < mlops-<date>.dump`.

**The GPU.** MaDoc, Ollama and the worker share the 16 GB of the GPU. Before every job the worker unloads Gemma
(Ollama reloads it at its next request: the app keeps working, a bit slower right after). Run trainings when
MaDoc is little used. A training stopping with "CUDA out of memory": lower `htr.batch_size` in `params.yaml`,
then `docker compose up -d --build`.

**The GitHub token** expires: when a release job fails with "401" or "403", make a new one and update `GH_TOKEN`
in `.env`, then `docker compose up -d`.

## Settings (`params.yaml`)

| Section | What it sets |
|---|---|
| `ingest` | the checks of a page, the group pattern of the split, the split fractions, the upload size limit |
| `augment` | augmentation on or off, copies per page, seed |
| `blla`, `htr` | the ketos options (epochs, batch size, lag, resize, extra arguments) |
| `gate` | the metric per page, the minimum gain, the tolerance on the old test pages, the bootstrap |
| `wandb` | entity, projects, model collections |
| `release` | MaDoc's repository, branch, model paths, model card |
| `gpu` | Ollama's address and the models to unload |

A change of `params.yaml` takes effect after `docker compose up -d --build`.

## Tests

The tests need neither GPU nor kraken nor W&B: `tests/fakes` stand for ketos and kraken, W&B is offline. They need
a PostgreSQL database that they empty:

```bash
docker run -d --name mlops-test-db -e POSTGRES_USER=mlops -e POSTGRES_PASSWORD=test -e POSTGRES_DB=mlops_test \
  -p 127.0.0.1:5433:5432 postgres:16
pip install -r requirements-ci.txt
MLOPS_TEST_DATABASE_URL=postgresql://mlops:test@127.0.0.1:5433/mlops_test python -m pytest -q
```

GitHub Actions runs them on every push (`.github/workflows/tests.yml`, with a PostgreSQL service). They cover the
checks and the pseudonymisation of the ingestion, the split by group, the dataset versions, the job queue and its
cancellation, the gate and the whole flow (upload → ingest → train → evaluate → pull request) with the worker.

## Files

```
madoc-mlops/
├── docker-compose.yml        db, mlops-ui, mlops-worker
├── Dockerfile                the image of mlops-ui and mlops-worker (PyTorch CUDA 12.8, kraken 7.0.2)
├── params.yaml               the settings
├── .env.example              the secrets to fill in .env
├── mlops/
│   ├── ui.py                 the control app
│   ├── worker.py             the job loop, heartbeat, cancellation
│   ├── jobs.py               the queue (PostgreSQL, SKIP LOCKED) and the job logs
│   ├── ingest.py             checks, pseudonymisation, split by group
│   ├── datasets.py           dataset versions
│   ├── augment.py            augmentation of the train pages of a version
│   ├── train.py              training job (ketos, W&B)
│   ├── evaluate.py           evaluation job, per-page metrics, the gate
│   ├── release.py            the production models (MaDoc clone) and the pull request
│   ├── textmetrics.py        CER, words found, bootstrap
│   ├── kraken_output.py      reading the output of ketos
│   ├── pagexml.py            reading PageXML
│   ├── resources.py          GPU / CPU / memory / disk, unloading Gemma
│   ├── process.py            running a command of a job (log, cancel)
│   ├── db.py, schema.sql     the database
│   ├── config.py             params.yaml and the environment
│   └── cli.py                health checks, import of a folder, list of jobs
├── script/data_augmentation_xml.py
├── scripts/backup.sh
└── tests/                    pytest, fake ketos / kraken
```
