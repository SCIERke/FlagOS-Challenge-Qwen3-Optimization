# Sync Tools (No Git Required)

Use this when the remote instance cannot access GitHub.

## 1) Start receiver on instance

```bash
cd ~/workspace
python src/tools/sync/receive_bundle.py
```

This listens on `:30000` and accepts `POST /upload`.
Health check endpoint: `GET /health` (returns `ok`).

Quick check:

```bash
bash src/tools/sync/check_connection.sh "https://flagos.io/flagos-lab/hw/node/HW-gpu09/port/26385/health"
```

## 2) Send from local machine

```bash
cd /path/to/workspace
bash src/tools/sync/send_bundle.sh "https://flagos.io/flagos-lab/hw/node/HW-gpu09/port/26385/upload"
```

`send_bundle.sh` now supports interactive selection of files/folders to include.
It also supports **changed-only sync** (default), based on file hashes from the
previous send on the local machine. Paths inside the bundle are relative, so
local/cloud absolute workspace paths can differ safely.

## 3) Apply behavior

Receiver calls:

```bash
bash src/tools/sync/apply_bundle.sh /tmp/flagos_sync/sync_bundle_*.tgz
```

`apply_bundle.sh` supports interactive bundle picking and interactive selection
of which top-level paths from the archive to extract.
