#!/usr/bin/env python3
"""A model to train an agent to play Atari Breakout"""
import random
from collections import deque

import ale_py
import gymnasium as gym
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers

gym.register_envs(ale_py)

ENV_NAME = "ALE/Breakout-v5"

GAMMA = 0.99
LEARNING_RATE = 2.5e-4
MEMORY_SIZE = 100_000
BATCH_SIZE = 32
TRAIN_STEPS = 10_000
TARGET_UPDATE_FREQUENCY = 10_000
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.999995
FRAME_STACK = 4


class FrameStack:
    """Frame stacking functionality"""

    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

    def preprocess(self, frame):
        frame = np.mean(frame, axis=2)

        frame = tf.image.resize(
            frame[..., np.newaxis],
            (84, 84),
            method="area"
        )

        frame = frame.numpy().squeeze()
        return frame.astype(np.float32) / 255.0

    def reset(self, frame):
        processed = self.preprocess(frame)
        self.frames.clear()

        for _ in range(self.stack_size):
            self.frames.append(processed)

        return np.stack(self.frames, axis=-1)

    def step(self, frame):
        processed = self.preprocess(frame)
        self.frames.append(processed)

        return np.stack(self.frames, axis=-1)


def build_model(input_shape, num_actions):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 8, strides=4, activation="relu"),
        layers.Conv2D(64, 4, strides=2, activation="relu"),
        layers.Conv2D(64, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(int(num_actions), activation="linear")
    ])

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="huber"
    )

    return model


memory = deque(maxlen=MEMORY_SIZE)


def train_step(model, target_model):
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)

    states = np.asarray(
        [x[0] for x in batch],
        dtype=np.float32
    )

    actions = np.asarray(
        [x[1] for x in batch],
        dtype=np.int32
    )

    rewards = np.asarray(
        [x[2] for x in batch],
        dtype=np.float32
    )

    next_states = np.asarray(
        [x[3] for x in batch],
        dtype=np.float32
    )

    dones = np.asarray(
        [x[4] for x in batch],
        dtype=np.float32
    )

    current_q = model.predict(
        states,
        verbose=0
    )

    next_q = target_model.predict(
        next_states,
        verbose=0
    )

    targets = rewards + (
        GAMMA * np.max(next_q, axis=1) * (1.0 - dones)
    )

    for i in range(BATCH_SIZE):
        current_q[i, actions[i]] = targets[i]

    model.train_on_batch(
        states,
        current_q
    )


env = gym.make(
    ENV_NAME,
    frameskip=4
)

num_actions = int(env.action_space.n)

print("Environment:", ENV_NAME)
print("Number of actions:", num_actions)
print("Observation shape:", env.observation_space.shape)

frame_processor = FrameStack(FRAME_STACK)

input_shape = (84, 84, FRAME_STACK)

model = build_model(
    input_shape,
    num_actions
)

target_model = build_model(
    input_shape,
    num_actions
)

target_model.set_weights(
    model.get_weights()
)

model.summary()

epsilon = EPSILON_START

state, _ = env.reset()
state = frame_processor.reset(state)

for step in range(1, TRAIN_STEPS + 1):

    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        q_values = model.predict(
            np.expand_dims(state, axis=0),
            verbose=0
        )
        action = int(np.argmax(q_values[0]))

    next_frame, reward, terminated, truncated, _ = env.step(
        action
    )

    done = terminated or truncated

    next_state = frame_processor.step(next_frame)

    memory.append(
        (state, action, reward, next_state, done)
    )

    train_step(
        model,
        target_model
    )

    state = next_state

    if done:
        state, _ = env.reset()
        state = frame_processor.reset(state)

    epsilon = max(
        EPSILON_MIN,
        epsilon * EPSILON_DECAY
    )

    if step % TARGET_UPDATE_FREQUENCY == 0:
        target_model.set_weights(
            model.get_weights()
        )

        print(
            f"Step: {step:,} | "
            f"Epsilon: {epsilon:.4f} | "
            f"Memory: {len(memory):,}"
        )

model.save("policy.h5")

print("Training complete.")
print("Policy saved to policy.h5")

env.close()
