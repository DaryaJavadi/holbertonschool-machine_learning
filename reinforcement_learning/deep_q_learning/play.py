#!/usr/bin/env python3
"""
Loads the policy network trained by train.py (policy.h5) and uses it to play
Atari's Breakout, rendering the game to the screen with a fully greedy
(exploitation-only) policy.
"""
import ale_py
import gymnasium as gym
import numpy as np
from keras.models import load_model

gym.register_envs(ale_py)

ENV_NAME = "ALE/Breakout-v5"
FRAME_STACK = 4
EPISODES = 5


class FrameStack:
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frames = []

    def preprocess(self, frame):
        frame = np.mean(frame, axis=2)
        frame = frame.astype(np.float32)

        from PIL import Image

        frame = Image.fromarray(frame)
        frame = frame.resize((84, 84))
        frame = np.asarray(frame, dtype=np.float32)

        return frame / 255.0

    def reset(self, frame):
        processed = self.preprocess(frame)
        self.frames = [processed] * self.stack_size

        return np.stack(self.frames, axis=-1)

    def step(self, frame):
        processed = self.preprocess(frame)

        self.frames.append(processed)

        if len(self.frames) > self.stack_size:
            self.frames.pop(0)

        return np.stack(self.frames, axis=-1)


env = gym.make(
    ENV_NAME,
    frameskip=4,
    render_mode="human"
)

model = load_model(
    "policy.h5",
    compile=False
)

frame_processor = FrameStack(FRAME_STACK)

print("Environment:", ENV_NAME)
print("Number of actions:", env.action_space.n)

for episode in range(1, EPISODES + 1):

    frame, _ = env.reset()

    state = frame_processor.reset(frame)

    done = False
    total_reward = 0

    while not done:

        q_values = model.predict(
            np.expand_dims(state, axis=0),
            verbose=0
        )

        action = int(
            np.argmax(q_values[0])
        )

        next_frame, reward, terminated, truncated, _ = env.step(
            action
        )

        done = terminated or truncated

        state = frame_processor.step(
            next_frame
        )

        total_reward += reward

    print(
        f"Episode {episode}: "
        f"Reward = {total_reward}"
    )

env.close()
