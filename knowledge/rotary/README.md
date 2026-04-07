# Rotary Issue Notes

This folder contains human-readable notes for the rotary compatibility issue.
These notes are documentation only and are not imported by runtime code.

## Current source-of-truth fix

1. Align Ascend backend signature:
   - `FlagGems/src/flag_gems/runtime/backend/_ascend/fused/rotary_embedding.py`
   - `apply_rotary_pos_emb(..., inplace: bool = False)`
2. Keep forwarding from modules explicit and stable:
   - `FlagGems/src/flag_gems/modules/rotary_embedding.py`
   - pass `inplace` as keyword when calling `flag_gems.apply_rotary_pos_emb(...)`

## Validation

Use:

```bash
python src/tools/rotary/check_rotary_compat.py
```
