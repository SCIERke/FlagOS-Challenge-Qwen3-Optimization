# Rotary Tools

Workspace-only helper tools for rotary compatibility checks.

- `check_rotary_compat.py`: static AST checker for source compatibility between:
  - `FlagGems/src/flag_gems/modules/rotary_embedding.py`
  - `FlagGems/src/flag_gems/runtime/backend/_ascend/fused/rotary_embedding.py`

Run:

```bash
python src/tools/rotary/check_rotary_compat.py
```

This tool is read-only and does not affect runtime behavior.
