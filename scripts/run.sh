#!/bin/bash
TASK=(
    "main/cheetah_run/expert/64px"
    "main/cheetah_run/expert/64px"
    "main/cheetah_run/expert/64px"
    "main/cheetah_run/expert/64px"
    # "main/cheetah_run/medium/64px"
    # "visual-cube-double-play-singletask-task1-v0"
    # "visual-cube-double-play-singletask-task1-v0"
    # "visual-cube-double-play-singletask-task2-v0"
    # "visual-cube-double-play-singletask-task3-v0"
    # "visual-cube-double-play-singletask-task4-v0"

    # "visual-cube-double-play-singletask-task5-v0"
    # "visual-cube-single-play-singletask-task1-v0"
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

    # "visual-cube-single-play-singletask-task1-v0"
    # "visual-cube-single-play-singletask-task2-v0"
    # "visual-cube-single-play-singletask-task3-v0"
    # "visual-cube-single-play-singletask-task4-v0"

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
    # "1234"
    "2345"
    "3456"
    "4567"
    "5678"
    "6789"
)
# Make sure you double check this for each set of tasks!
ALPHA=(
    300
    500
    10000
    30000
)
DISC=(
    15.0
    15.0
    15.0
    15.0
)
NUM_ENSEMBELS=(
    4
    6
)
CONFOUND_MODE=(
    0
    1
    2
    3
)
for i in "${!TASK[@]}"; do
    # if [ "$i" -lt 3 ]; then
    #     continue
    # fi
    # task index starts from 0
    for j in {0..4}; do
        session_id="$((i*5+j))_cheetah"
        tmux has-session -t "$session_id" 2>/dev/null || tmux new-session -d -s "$session_id"
        COMMAND="python main.py --run_group=${TASK[$i]} --env_name=${TASK[$i]} --offline_steps=500000 --agent=agents/en_cfd_fql.py \
                    --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3 --device_id=$j \
                    --mujoco_device_id=$(((j+1)%5)) --agent.alpha=${ALPHA[$i]} \
                    --seed=${SEEDS[$j]} --agent.disc_coef=${DISC[$i]} --confound_mode=${CONFOUND_MODE[$i]}"
                    #  --agent.num_ensembles=${NUM_ENSEMBELS[$i]}"
        tmux send-keys -t "$session_id" "cd ~/development/fql/" C-m
        tmux send-keys -t "$session_id" "$COMMAND" C-m
        # tmux send-keys -t "$session_id" \
        # "echo \
        # 'Session: $((i*5+j+1)) Device ID: $((j+2)) Mujoco ID: $(((i*5+j+1)%2))\
        # Task: ${TASK[$i]} \
        # Seed: ${SEEDS[$j]}'" C-m
        # python main.py --run_group="DEBUG" --env_name="main/cheetah_run/expert/64px" --offline_steps=10 --agent=agents/en_cfd_fql.py \
        #             --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3 --device_id=6 --mujoco_device_id=7 --agent.alpha=100 --seed=1234 --agent.disc_coef=15 --eval_interval=0 --confound_mode=1
    done
done
