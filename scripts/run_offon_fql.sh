#!/bin/bash
TASK=(
    # "visual-cube-single-play-singletask-task1-v0"
    # "visual-cube-double-play-singletask-task1-v0"
    # "visual-puzzle-4x4-play-singletask-task1-v0"
    "visual-puzzle-3x3-play-singletask-task5-v0"
    # "visual-antmaze-medium-navigate-singletask-task1-v0"
    # "visual-antmaze-teleport-navigate-singletask-task1-v0"
)
SEEDS=(
    "1234:3456:4567:891011"
    # "1234:2345:3456:4567"
    # "1234:2345:5678:4567"
    # "1234:2345:3456:5678"
)
# Make sure you double check this for each set of tasks!
ALPHA=(
    300
    # 100
    # 100
    # 100
    # 300
)
DISC=(
    5.0
    # 15.0
    # 10.0
    # 20.0
    # 5.0
)
for i in "${!TASK[@]}"; do
    # if [ "$i" -lt 3 ]; then
    #     continue
    # fi
    # task index starts from 0
    IFS=':' read -ra SEED_ARRAY <<< "${SEEDS[$i]}"
    for j in {0..2}; do
        session_id="offon_fql_$((i*3+j))"
        tmux has-session -t "$session_id" 2>/dev/null || tmux new-session -d -s "$session_id"
        COMMAND="python main.py --run_group=${TASK[$i]}_offon --env_name=${TASK[$i]} --offline_steps=500000 --online_steps=500000 --agent=agents/fql.py \
                    --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3 --device_id=$((j+5)) \
                    --mujoco_device_id=$(((j+1)%3+5)) --agent.alpha=${ALPHA[$i]} \
                    --seed=${SEED_ARRAY[$j]}"
        # COMMAND="echo TASK ${TASK[$i]}_offon SEED ${SEED_ARRAY[$j]} Did $((i+2)) Mid $(((i+1)%4+2))"
        tmux send-keys -t "$session_id" "cd ~/development/fql/" C-m
        tmux send-keys -t "$session_id" "$COMMAND" C-m
        # python main.py --run_group="DEBUG" --env_name="visual-cube-double-play-singletask-task1-v0" --offline_steps=1 --online_steps=500000 --agent=agents/en_cfd_fql.py \
        #             --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3 --device_id=6 --mujoco_device_id=7 --agent.alpha=100 --seed=1234 --agent.disc_coef=15 --eval_interval=0
    done
done
