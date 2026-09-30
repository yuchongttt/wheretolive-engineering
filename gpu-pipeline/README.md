# gpu-pipeline — batch ML on home GPUs, driven from a Mac production host

wheretolive.xyz runs on a Mac mini. The ML work behind it (image embeddings,
SigLIP "luxury/decor" scoring, VLM floorplan parsing) runs on two home GPU boxes.
This module holds the code on both sides of that boundary: the GPU-box daemons
and their systemd units, and the Mac-side scripts that dispatch work, check on
it, and alert.

## Topology

```
            Mac mini (production, launchd)                 private WireGuard mesh (Tailscale)
 ┌───────────────────────────────────────────────┐
 │ Next.js site + admin API  ◀── POST stats ─────┼──────────┐
 │ SQLite: evaluations.db, ops.db (skill_runs)   │          │ every 30 s / 1 min
 │ mac/probe_gpu_box_queues.py ── ssh + python ──┼────┐     │
 │ mac/reconcile_dispatched.py (every 10 min)    │    │     │
 │ mac/queue_status_report.py  (hourly, events)  │    ▼     │
 │ mac/gpu_watchdog.py ── ssh healthcheck ───────┼──▶ ┌─────┴──────────────────────────────┐
 │ admin "start/stop" ── HTTP :8200 ─────────────┼──▶ │ gpu-box: RTX 3060 12 GB, 16 GB RAM │
 │                                               │    │ wtl-embed-combined (DINO+SigLIP)   │
 │ mac/floorplan_vlm_analyze.py (daemon) ────┐   │    │ wtl-siglip-server, wtl-control     │
 └───────────────────────────────────────────┼───┘    │ reporters (systemd timers)         │
                                             │        │ dataset.db (queue + vectors)       │
                     OpenAI-compatible HTTP  ▼        └────────────────────────────────────┘
                     ┌────────────────────────────────────┐
                     │ vlm-box: RTX 3090 24 GB, WSL2      │
                     │ wtl-vlm = vllm serve Qwen3.6-27B   │
                     │ INT4 AutoRound, :8200              │
                     └────────────────────────────────────┘
```

Sources: 2026-05-18 design doc (RTX 3060 12 GB), 2026-05-27 index spec (16 GB
RAM), VLM setup notes + `systemd/vlm-box/wtl-vlm.service` (RTX 3090, WSL2).

## Why the boundary is "interactive on Mac, batch on Linux"

From the 2026-05-18 design doc: the Mac runs anything that must answer a user
request within ~6 s (page render, evaluation, retrieval) from its own SQLite;
the GPU boxes run anything GPU-bound or storage-heavy. The network hop is cheap,
but a user-facing dependency on a home box is felt whenever that box is off. So
GPU output is pulled back and stored on the Mac, and the GPU boxes can go
offline without taking pages down. The one exception that design doc lists is
photo-upload scoring, which calls `siglip_server.py` live.

## How work moves

**Embeddings (gpu-box).** `gpu-box/embed_daemon_combined.py` treats the
`images` table in `dataset.db` as its queue: a row is work if it has been
downloaded (`status='done'`) and has no vector yet. Images are fetched into a
local cache by a separate component, not included. Design points:

- *One process, one disk read, two models.* Two separate DINO and SigLIP daemons
  read every photo twice, saturated the data HDD and stalled in `D` state. The
  merged daemon opens each image once, runs both forward passes back to back on
  the same batch, and writes all rows in one transaction, so SQLite has one writer.
- *Implicit claim, idempotent write.* A single consumer selects missing rows and
  writes with `INSERT OR IGNORE`, so a replay after a crash is harmless.
  Undecodable images are marked `embed_unreadable` so they are not claimed again.
  A second copy would double-process (see `pause-embed.conf` below).
- *Watermark-gated scans.* The claim is an anti-join. Once the queue drained it
  cost 60.6 s of CPU per cycle to confirm there was nothing to do. The daemon
  now probes `MAX(completed_at)` (O(1) via an index) and skips the scan when it
  hasn't moved: 60.6 s → 0.027 s (`systemd/wtl-embed-combined.service.d/watermark-notes.conf`,
  2026-09-03). The same file lists two ways the watermark can silently skip work.
  The watermark is kept in memory on purpose, so a restart forces one full rescan.

**Fire-and-forget jobs, reconciled later.** Mac workflows record async work as
`status='dispatched'` rows in `ops.db` (`mac/skill_report.py`).
`mac/probe_gpu_box_queues.py` sshes to the GPU box, runs a read-only query and
writes `data/queue_stats.json` atomically (tmp + rename).
`mac/reconcile_dispatched.py` closes each dispatched row as `success` if its
downstream queue is empty or has processed anything since dispatch, and as
`failed` otherwise. Only queues that still have a live probe may be listed:
a retired queue left in the map produced 192 false failures in a week (comment
dated 2026-07-27).

**Floorplan VLM (Mac → vlm-box).** `mac/floorplan_vlm/dispatcher.py` (always-on
backfill) and `mac/floorplan_vlm_analyze.py` (one-shot or `--daemon`) select
properties newest first and send up to 4 concurrent requests to vLLM's
OpenAI-compatible API. Results are keyed `(property, idx, model_version)`, so
re-runs are idempotent; claims match a `v6.6%` prefix, so a schema bump (v6.6b)
shipped without reprocessing the backlog. Failures are split on purpose: an
image that cannot be fetched writes an `ok=0` row (24 h backoff); a VLM-server
error (503 during a vLLM restart, connection refused) writes nothing and is
retried next tick. When 404s were still "transient", 10 dead properties pinned at
the queue head made 398 of 400 ticks fail (2026-08-17). Tests also pin the
post-processing guards: alpha PNGs flattened onto white (PNG sources had a 24.4 %
empty-parse rate vs 0.1 %, 2026-08-14), room areas capped at total floor area,
outdoor spaces kept out of that cap.

## When a box is offline

- **GPU box down.** Reporter cards go stale; the probe fails and
  `queue_status_report.py` sends one alert once stats are >1 h old, then one
  "back to normal". `gpu_watchdog.py` stays silent (reachability is another
  monitor's job). No work is lost: the queue *is* database state and the next
  start rescans.
- **VLM box down.** Calls fail as server errors, no rows are written, the queue
  resumes later. `mac/vlm-maintenance.sh` silences alerts for a planned window
  that self-expires and is cleared once `/health` passes, so a forgotten `stop`
  cannot mute alerts indefinitely.
- **GPU present but unusable.** On 2026-07-24 a driver upgrade without a reboot
  left running processes on CUDA while new ones could not get it. The failure
  stayed hidden for 3 days, during which one unit restarted 14,536 times.
  `mac/gpu_box_healthcheck.py` now checks that the kernel-module and userspace
  driver versions match, starts a *fresh* process to test CUDA, and reads
  `NRestarts` (sampling `is-active` misses crash loops). Results that could not
  be determined are tagged `INCONCLUSIVE` and never page.

## Memory contention

- **VRAM.** DINOv3-L and SigLIP-2 share one process and one batch of 32 in bf16
  ("both fit in 12 GB VRAM with margin", daemon docstring), with
  `torch.cuda.empty_cache()` every 50 batches. `DINO_ENABLED=0` skips loading
  DINO entirely. The 27B VLM gets its own 24 GB card: vLLM with INT4 AutoRound,
  `--gpu-memory-utilization 0.92`, `--max-model-len 8192`, `--max-num-seqs 4`.
- **Host RAM (16 GB): the kill order is explicit.** `OOMScoreAdjust` ranks
  victims (EPC refresh 500 "be the first OOM victim", EPC server 300 "yield to"
  the ML daemons); `ManagedOOMSwap=kill` hands the embed daemon, SigLIP server
  and FAISS rebuild to systemd-oomd, while the user-facing (now retired) search
  server got `auto` because oomd killing it caused a restart death spiral. sshd
  and tailscaled carry `-900` drop-ins (kept outside the deploy glob so a bad
  deploy cannot break sshd; not in this repo) to keep the box reachable.
- **Resident indexes.** The FAISS delta index is off by default
  (`FAISS_DELTA_ENABLED`): on 2026-09-01 it pushed the box into repeated oomd
  kills. For the 5.7 GB flat DINO index (1.66 M × 1024 fp32, 2026-05-27 spec),
  `build_base_pq_index.py` builds an HNSW+SQ8 copy (~4× smaller, per its
  docstring) behind a server flag. HNSW+PQ was tried first, but its recall
  collapsed to 13 % with inner product. `eval_pq_recall.py` fails unless
  recall@10 ≥ 0.90 vs flat.

## systemd design (`systemd/`)

- Daemons: `Restart=on-failure` + `RestartSec`; the embed daemon finishes its GPU
  batch on SIGTERM (`TimeoutStopSec=60`). Reporters: `Type=oneshot` + timers on
  wall-clock `OnCalendar`, not `OnUnitActiveSec`, which drifted whenever a run was
  slow (SQLite busy) and left gaps in the admin charts.
- **Drop-ins are the override layer; the base unit alone misleads.** Without
  `offline.conf` (`HF_HUB_OFFLINE=1`) the daemon crash-loops trying to reach the
  Hugging Face Hub. `deploy.sh` always installs drop-ins and never restarts services.
- `pause-embed.conf` is stale (it targets the old `--user` embed unit, so a
  rebuild would not pause the daemon and could start a second copy); its timer is
  disabled and has never run, and the file is kept visible, not silently deleted.
- `journald.conf.d` caps logs at 4 G on the tight SSD; applied by hand, since it
  needs a journald restart.

## Reporter pattern

Each GPU-box job POSTs one JSON snapshot to `$MAC_ADMIN_URL/api/admin/<worker>`
(`x-admin-key`); the Mac keeps the latest and the admin UI renders it. No queue
or bus: latest-wins snapshots are idempotent and a missed push only makes a card
stale. `report_gpu_tasks.py` attributes GPU memory per process (`nvidia-smi` +
`/proc`; jobs may add `/data/ml/gpu_status/<pid>.json` progress). `host_report.sh`
validates every `nvidia-smi` field so a broken driver yields `"gpu": null`, not
invalid JSON. `control_server.py` is the reverse path: a root `systemctl` proxy
limited to a whitelist, so a leaked key can at most bounce the embed daemon.

## Layout, tests, dependencies

`mac/` runs on the Mac (writes to `./data/`), `gpu-box/` deploys to
`/data/ml/dataset/` (`host_report.sh` goes to `/opt/wheretolive/`), and
`systemd/` holds units for `/etc/systemd/system/` (`vlm-box/` for the VLM box).

```
cd gpu-pipeline && python -m pytest -q      # 89 passed
```

Tests are offline (in-memory SQLite, monkeypatched network, synthetic ids/URLs)
and need `pytest` + `Pillow`. Runtime deps: gpu-box `torch`, `transformers`,
`numpy`, `Pillow`, `faiss`, `scikit-learn` (PCA fit only), `nvidia-smi`/`curl`/`iostat`;
mac stdlib + `Pillow`, `ssh`, `curl`; vlm-box `vllm` (≥ 0.19 per setup notes).

Not included: the component that fills the local image cache, the retired DINO search
server (2026-06-28), EPC service code (its units remain for the OOM ordering),
the weekly FAISS build script, Mac-side result drains, the admin API, and the
Telegram helper.

## Environment variables

| Var | Used by | Meaning |
|---|---|---|
| `GPU_HOST` | `mac/*`, `deploy.sh` | `user@host` of the GPU box on the private network |
| `MAC_ADMIN_URL` | gpu-box reporters | Mac admin API base, e.g. `http://<mac>:3000` |
| `WTL_ADMIN_KEY` | reporters, `control_server.py` | shared admin key (`<set-me>`), from `/etc/wtl/wtl.env` (root, 0600) |
| `WTL_HOST_ID` | reporters | host id in snapshots (default `gpu-box`) |
| `ADMIN_KEY` | `queue_status_report.py` | admin key for the Mac API (or read from `web/.env*`) |
| `WTL_VLM_URL`, `WTL_VLM_*` | VLM dispatcher, `vlm-maintenance.sh` | vLLM base URL (default `http://vlm-box:8200`), batch/concurrency/model tag |
| `WTL_FP_LOCAL_IMAGES`, `WTL_FP_LOCAL_URL` | VLM dispatcher | optional local-first image source on the VLM box |
| `WTL_NOTIFY_CMD` | `gpu_watchdog.py` | alert command (message appended); unset = print only |
| `DINO_ENABLED`, `FAISS_DELTA_ENABLED` | embed daemon | model/index feature flags (set via drop-ins) |
