# FlagGems `patch_util.py` — อ่านย่อ (workspace note)

เอกสารนี้อยู่ใน `knowledge/` เพื่อให้ทีมอ่านโครงสร้างได้เร็ว **ไม่แทนที่ doc ใน FlagGems** และ **ไม่มีผลต่อ runtime** (ไม่มีโค้ด import จากไฟล์นี้)

**ไฟล์จริงใน FlagGems:** `FlagGems/src/flag_gems/patches/patch_util.py`

---

## ทำไมมีไฟล์นี้

บาง build ของ vLLM **ไม่มี** C extension / custom op ครบ (`torch.ops._C`, `_moe_C`, …) แต่ FlagGems ยังอยากเรียก path เดียวกับ vLLM ได้

`patch_util` จึงทำสองอย่างหลัก:

1. **ลอง import** โมดูล vLLM ที่ลงทะเบียน op จริง
2. ถ้า **ไม่มี op** → พยายาม **`torch.library.define`** ใส่ **ลายเซ็น (signature string)** ให้มีชื่อ op ก่อน แล้วค่อยให้ FlagGems **ผูก implementation** ทีหลัง

---

## โครงสร้างอ่านง่าย (mental model)

```
┌─────────────────────────────────────────────────────────┐
│  vLLM build มี C++ ops ครบ?                              │
└─────────────────────┬───────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
        ใช่                      ไม่
   import ได้ + มี op      import ไม่ได้ / op หาย
          │                       │
          │                       ▼
          │              _define_op_if_not_exists(...)
          │              (สร้างชื่อ op + signature เปล่า)
          │                       │
          └───────────┬───────────┘
                      ▼
          torch.library.Library("<lib>", "IMPL")
          → patch_vllm_lib(lib, name, fn, key)
             ผูก kernel / ฟังก์ชันของ FlagGems เข้า torch.ops
```

---

## ตารางสรุป (map ในโค้ด)

| ส่วนในไฟล์ | หน้าที่ |
|------------|---------|
| `_try_import_vllm_extension` | `import` โมดูล vLLM ตาม `module_map` |
| `_is_op_registered` | เช็ก `torch.ops.<lib>.<op>` มีจริงไหม |
| `_ensure_vllm_library_exists` | รวม import + เช็ก op (ถ้ามีรายการ `ops_to_check`) |
| `_LIB_OPS` | รายการ op ที่สนใจต่อ library |
| `_OP_SIGNATURES` | สตริง signature สำหรับ `torch.library.define` เมื่อของจริงไม่มี |
| ลูปต้นๆ ไฟล์ (~104–113) | ถ้า lib โหลดไม่สำเร็จ → `define` แต่ละ op ที่ขาด |
| `vllm_*_lib = Library(..., "IMPL")` | handle สำหรับลง `impl` |
| `patch_module_method` | แทนที่ method ของ **คลาส Python** (เช่น ของ vLLM) |
| `patch_vllm_lib` | `lib.impl(fn_name, fn, key)` — ผูก `torch.ops.<lib>.<fn>` |

---

## ความสัมพันธ์กับปัญหา rotary

| หัวข้อ | คำตอบสั้นๆ |
|--------|------------|
| rotary ที่เคย error `apply_rotary_pos_emb(..., 7 args)` | เป็นเรื่อง **signature ของฟังก์ชัน Python** ใน `runtime/backend/_ascend/fused/rotary_embedding.py` + `modules/rotary_embedding.gems_rope_forward` |
| `patch_util` แก้ rotary โดยตรงไหม | **ไม่** — มันจัดการ **`torch.ops` ของ vLLM** (`_C`, MoE, FA3, cache) ไม่ใช่ path `apply_rotary_pos_emb` ของ FlagGems |
| แก้ rotary ถูกที่ | ให้ Ascend `apply_rotary_pos_emb` รับ `inplace` ให้ตรงกับ `gems_rope_forward` (ดู commit / patch ใน FlagGems) |

---

## อ้างอิงด่วน

- **Plugin เรียก FlagGems rotary:** `vllm-plugin-FL/vllm_fl/dispatch/backends/flaggems/impl/rotary.py`
- **Forward กลาง:** `flag_gems.modules.rotary_embedding.gems_rope_forward`
- **Ascend kernel:** `flag_gems.runtime.backend._ascend.fused.rotary_embedding.apply_rotary_pos_emb`

---

*อัปเดตตามความเข้าใจจาก source ใน workspace; ถ้า FlagGems อัปสตรีมเปลี่ยน ให้เทียบกับไฟล์จริงใน tree.*
