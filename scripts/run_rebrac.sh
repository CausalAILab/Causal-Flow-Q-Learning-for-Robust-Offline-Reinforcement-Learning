#!/bin/bash
TASK=(
    # "visual-cube-double-play-singletask-task2-v0"
    # "visual-cube-double-play-singletask-task3-v0"
    # "visual-cube-double-play-singletask-task4-v0"

    "visual-cube-double-play-singletask-task5-v0"
    "visual-cube-single-play-singletask-task1-v0"

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

    "visual-puzzle-4x4-play-singletask-task2-v0"
    "visual-puzzle-4x4-play-singletask-task3-v0"
    "visual-puzzle-4x4-play-singletask-task4-v0"
    "visual-puzzle-4x4-play-singletask-task5-v0"

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

for i in "${!TASK[@]}"; do
    # TODO: remeber to remove this skipping dummy first two tasks
    # Check if the loop counter variable 'i' is less than 2
    # If true, the condition will execute the following block of code
    # This is typically used to limit iterations or control loop behavior based on a threshold
    if [ "$i" -lt 2 ]; then
        continue
    fi
    # task index starts from 0
    for j in {0..3}; do
        session_id="$((i*4+j+1))"
        tmux has-session -t "$session_id" 2>/dev/null || tmux new-session -d -s "$session_id"
        COMMAND="systemd-run --user --scope -p MemoryMax=1T -p MemorySwapMax=0 python main.py --run_group=${TASK[$i]} --env_name=${TASK[$i]} --offline_steps=500000 --agent=agents/rebrac.py --agent.alpha_actor=1 --agent.alpha_critic=0 \
                    --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3 --device_id=$((i+2)) \
                    --mujoco_device_id=$(((i+1)%6+2)) \
                    --seed=${SEEDS[$j]}"
        tmux send-keys -t "$((i*4+j+1))" "cd ~/development/fql/" C-m
        tmux send-keys -t "$((i*4+j+1))" "$COMMAND" C-m
        sleep 1s
        # tmux send-keys -t "$((i*5+j+1))" \
        # "echo \
        # 'Session: $((i*5+j+1)) Device ID: $((j+2)) Mujoco ID: $(((i*5+j+1)%2))\
        # Task: ${TASK[$i]} \
        # Seed: ${SEEDS[$j]}'" C-m
        # systemd-run --user --scope -p MemoryMax=1T -p MemorySwapMax=0 python main.py --env_name="visual-cube-single-play-singletask-task2-v0" --offline_steps=500000 --agent=agents/rebrac.py --agent.alpha_actor=1 --agent.alpha_critic=0 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3 --device_id=4 --mujoco_device_id=5 --seed=1234
    done
done
