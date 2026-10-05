#!/bin/bash
# Wait for teacher generation to finish, then train
echo "Waiting for teacher generation to complete..."
while pgrep -f "generate_teacher.py" > /dev/null; do
    sleep 60
done

echo "Teacher generation done. Starting distillation training..."
cd ~/GymPose-Lite
python3 train_distill.py \
    --data datasets/teacher_heatmaps_full \
    --epochs 50 \
    --use-cv \
    --lr 0.001 \
    2>&1 | tee distill_train_log.txt

echo "Training complete."
