#!/bin/bash
TASK=(
    # "visual-cube-double-play-singletask-task3-v0"
    # "visual-cube-double-play-singletask-task4-v0"
    # "visual-cube-double-play-singletask-task5-v0"

    "visual-cube-single-play-singletask-task1-v0"
    "visual-cube-double-play-singletask-task1-v0"
    
    # "visual-cube-single-play-singletask-task2-v0"
    # "visual-cube-single-play-singletask-task3-v0"
    # "visual-cube-single-play-singletask-task4-v0"
    # "visual-cube-single-play-singletask-task5-v0"

    # "visual-scene-play-singletask-task1-v0"
    # "visual-scene-play-singletask-task2-v0"
    # "visual-scene-play-singletask-task3-v0"
    # "visual-scene-play-singletask-task4-v0"
    # "visual-scene-play-singletask-task5-v0"

    # "visual-puzzle-3x3-play-singletask-task1-v0"
    # "visual-puzzle-3x3-play-singletask-task2-v0"
    # "visual-puzzle-3x3-play-singletask-task3-v0"
    # "visual-puzzle-3x3-play-singletask-task4-v0"
    # "visual-puzzle-3x3-play-singletask-task5-v0"
    # "visual-puzzle-4x4-play-singletask-task1-v0"
    
    # "visual-puzzle-4x4-play-singletask-task2-v0"
    # "visual-puzzle-4x4-play-singletask-task3-v0"
    # "visual-puzzle-4x4-play-singletask-task4-v0"
    # "visual-puzzle-4x4-play-singletask-task5-v0"

    # "visual-antmaze-medium-navigate-singletask-task1-v0"
    # "visual-antmaze-medium-navigate-singletask-task2-v0"
    # "visual-antmaze-medium-navigate-singletask-task3-v0"
    # "visual-antmaze-medium-navigate-singletask-task4-v0"
    # "visual-antmaze-medium-navigate-singletask-task5-v0"

    # "visual-antmaze-teleport-navigate-singletask-task1-v0"
    # "visual-antmaze-teleport-navigate-singletask-task2-v0"
    # "visual-antmaze-teleport-navigate-singletask-task3-v0"
    # "visual-antmaze-teleport-navigate-singletask-task4-v0"
    # "visual-antmaze-teleport-navigate-singletask-task5-v0"

    # May need to run again with disc 5
    # "visual-cube-single-play-singletask-task1-v0"
    # "visual-cube-single-play-singletask-task2-v0"
    # "visual-cube-single-play-singletask-task3-v0"
    # "visual-cube-single-play-singletask-task4-v0"
    # "visual-cube-single-play-singletask-task5-v0"


    # "visual-antmaze-medium-stitch-singletask-task1-v0"
    # "visual-antmaze-medium-stitch-singletask-task2-v0"

    # "visual-antmaze-medium-stitch-singletask-task3-v0"
    # "visual-antmaze-medium-stitch-singletask-task4-v0"
    # "visual-antmaze-medium-stitch-singletask-task5-v0"

    # "visual-antmaze-teleport-stitch-singletask-task1-v0"
    # "visual-antmaze-teleport-stitch-singletask-task2-v0"
    # "visual-antmaze-teleport-stitch-singletask-task3-v0"

    # "visual-antmaze-teleport-stitch-singletask-task4-v0"
    # "visual-antmaze-teleport-stitch-singletask-task5-v0"

)
SEEDS=(
    "1234"
    "2345"
    "3456"
    "4567"
    # "5678"
)
# Make sure you double check this for each set of tasks!
ALPHA=(
    300
    100
    # 300
    # 300
    # 300
    # 300
)
for i in "${!TASK[@]}"; do
    if [ "$i" -lt 1 ]; then
        continue
    fi
    # task index starts from 0
    for j in {0..3}; do
        session_id="$(((i+2)*4+j+1))_offon_fql"
        tmux has-session -t "$session_id" 2>/dev/null || tmux new-session -d -s "$session_id"
        COMMAND="python main.py --run_group=${TASK[$i]}_offon --env_name=${TASK[$i]} --offline_steps=500000 --online_steps=500000 --agent=agents/fql.py \
                    --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3 --device_id=$((i+6)) \
                    --mujoco_device_id=$((7-i)) --agent.alpha=${ALPHA[$i]} --seed=${SEEDS[$j]}"
        tmux send-keys -t "$session_id" "cd ~/development/fql/" C-m
        tmux send-keys -t "$session_id" "$COMMAND" C-m
        sleep 1s
        # tmux send-keys -t "$((i*5+j+1))" \
        # "echo \
        # 'Session: $((i*5+j+1)) Device ID: $((j+2)) Mujoco ID: $(((i*5+j+1)%2))\
        # Task: ${TASK[$i]} \
        # Seed: ${SEEDS[$j]}'" C-m
    done
done
