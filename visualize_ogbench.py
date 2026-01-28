import ogbench
import matplotlib.pyplot as plt
import numpy as np
import os

# To visualize goal, need to change cube_env.py: initialize_episode, 
# move render goal part after second initialize_arm and also comment out self._data.qpos, qvel reset part
# Also need to change resolution in manipspace.__init__

os.environ["MUJOCO_EGL_DEVICE_ID"] = "2"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "true"
env_name = "visual-cube-quadruple-play-singletask-task2-v0"
eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
obs, info = eval_env.reset(seed=100, options={"render_goal": True})
goal_img = info["goal_rendered"]
plt.imsave(f"{env_name}_goal_low.png", goal_img, dpi=500)
plt.imsave(f"{env_name}_low.png", obs, dpi=500)
# print(obs)
eval_env.close()