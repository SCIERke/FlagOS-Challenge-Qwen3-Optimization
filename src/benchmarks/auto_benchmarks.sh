#!/bin/bash

# ==========================================
# 🚀 VLLM Ascend Benchmark Auto-Runner
# ==========================================

echo "🔥 [1/3] Initializing NPU Environment..."

# 1. ร่ายคาถาปลุก NPU และแก้บั๊กต่างๆ
export VLLM_TARGET_DEVICE="npu"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export TORCH_NPU_MULTIPROCESSING_START_METHOD="spawn"
export VLLM_PLUGINS="ascend"
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH

# แก้ปัญหา Socket c10d (The hostname of the client socket cannot be retrieved)
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29500"

# [ตัวเลือกเสริม] เอาเครื่องหมาย # ออกบรรทัดล่าง ถ้าอยากให้ปริ้นท์บอกว่าใช้ FlagGems ตัวไหนบ้าง
# export VLLM_FL_LOG_LEVEL=DEBUG

# 2. ตั้งค่าตัวแปรเส้นทางโมเดล
MODEL_PATH="/flagos-lab/op2/Qwen3-4B"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="bench_log_${TIMESTAMP}.txt"

echo "🎯 [2/3] Target Model: $MODEL_PATH"
echo "📝 Log will be saved to: $LOG_FILE"
echo "--------------------------------------------------------"

# 3. ยิง Benchmark และบันทึกผลลงไฟล์ (ใช้ tee เพื่อให้แสดงบนจอและเซฟลงไฟล์พร้อมกัน)
echo "⏳ [3/3] Running Benchmark... (Please wait for Warm-up)"
./run_benchmark.sh $MODEL_PATH 2>&1 | tee $LOG_FILE

echo "--------------------------------------------------------"
echo "✅ Benchmark Completed!"
echo "📊 To view the result again, run: cat $LOG_FILE"