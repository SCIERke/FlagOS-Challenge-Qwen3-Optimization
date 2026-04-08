# Cloud sync workflow

Edits under `workspace/` (including `FlagGems/`, `vllm-plugin-FL/`, `src/`) exist on your **local** tree first. The lab **cloud instance often cannot reach GitHub**, so **changing files locally is not enough** for runs on the instance: you must **submit the same changes to the cloud** after you modify code you care about there.

**Practical rule:** after any change that should affect benchmarks or runtime on the instance, run the bundle sender (from `workspace/`), then confirm apply finished (health + status).

```bash
# Example — replace URL with your node’s mapped `/upload` endpoint
printf '\n\n\n' | bash src/tools/sync/send_bundle.sh "https://flagos.io/.../upload"
curl -sS "https://flagos.io/.../status/latest"
```

Details, receiver setup, and apply behavior: `src/tools/sync/README.md`.
