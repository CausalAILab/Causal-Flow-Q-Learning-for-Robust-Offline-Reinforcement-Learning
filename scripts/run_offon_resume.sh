#!/bin/bash
TASK=(
    "visual-cube-single-play-singletask-task1-v0"
    "visual-cube-double-play-singletask-task1-v0"
    # might need to run puzzle again with the new online setting
    "visual-puzzle-4x4-play-singletask-task1-v0"
    # "visual-antmaze-medium-navigate-singletask-task1-v0"
    # "visual-antmaze-teleport-navigate-singletask-task1-v0"
)
SEEDS=(

    "1234:3456:5678:891011"
    # "1234:2345:3456:4567"
    # "1234:2345:5678:4567"
    # "1234:2345:3456:5678"
)
PATHS=(

    "sd1234_20260125_173151:sd5678_20260125_173136:sd891011_20260125_074903:sd3456_20260125_074903"
    # "sd1234_20260125_074903:sd2345_20260125_074903:sd3456_20260125_074903:sd4567_20260125_074903"
    # "sd1234_20260125_074903:sd4567_20260125_074903:sd2345_20260125_074903:sd5678_20260125_074903"
    # "sd3456_20260125_074903:sd1234_20260125_074903:sd2345_20260125_074903:sd5678_20260125_074903"
)
EPOCHS=(
    "500000:500000:500000:500000"
    "500000:500000:500000:500000"
    "500000:500000:500000:500000"
    # "900000:900000:500000:900000"
    # "800000:800000:800000:800000"
    # "800000:800000:800000:800000"
)
# Make sure you double check this for each set of tasks!
ALPHA=(
    300
    100
    100
    # 100
    # 300
)
DISC=(
    15.0
    5.0
    10.0
    # 20.0
    # 5.0
)
for i in "${!TASK[@]}"; do
    # if [ "$i" -lt 3 ]; then
    #     continue
    # fi
    # task index starts from 0
    IFS=':' read -ra SEED_ARRAY <<< "${SEEDS[$i]}"
    IFS=':' read -ra PATH_ARRAY <<< "${PATHS[$i]}"
    IFS=':' read -ra EPOCH_ARRAY <<< "${EPOCHS[$i]}"
    for j in {0..3}; do
        session_id="offon_$((i*5+j+1))"
        tmux has-session -t "$session_id" 2>/dev/null || tmux new-session -d -s "$session_id"
        COMMAND="python main.py --run_group=${TASK[$i]}_offon --env_name=${TASK[$i]} --offline_steps=500000 --online_steps=500000 --agent=agents/en_cfd_fql.py \
                    --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3 --device_id=$((i+2)) \
                    --mujoco_device_id=$(((i+1)%4+2)) --agent.alpha=${ALPHA[$i]} \
                    --seed=${SEED_ARRAY[$j]} --agent.disc_coef=${DISC[$i]} \
                    --restore_epoch=${EPOCH_ARRAY[$j]} \
                    --restore_path=~/explogs/fql/fql/${TASK[$i]}_offon/${PATH_ARRAY[$j]}"
        # COMMAND="echo TASK ${TASK[$i]}_offon SEED ${SEED_ARRAY[$j]} Did $((i+2)) Mid $(((i+1)%4+2))"
        tmux send-keys -t "$session_id" "cd ~/development/fql/" C-m
        tmux send-keys -t "$session_id" "$COMMAND" C-m
        # python main.py --run_group="visual-cube-double-play-singletask-task1-v0_offon_resume" --env_name="visual-cube-double-play-singletask-task1-v0" --offline_steps=500000 --online_steps=500000 --agent=agents/en_cfd_fql.py \
        #             --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3 --device_id=6 --mujoco_device_id=7 --agent.alpha=100 --seed=1234 --agent.disc_coef=15 --eval_interval=0 \
        #             --restore_epoch=500000 \
        #             --restore_path="~/explogs/fql/fql/visual-cube-double-play-singletask-task1-v0_offon/sd1234_20260125_173151" --log_interval=1000
    done
done
