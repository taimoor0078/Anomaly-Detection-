# ==========================================================
# INSIDER THREAT DETECTION USING DOUBLE DEEP Q NETWORK
# PYTORCH VERSION (GPU / CUDA) - OPTIMIZED FOR 1.6M ROWS
# MODEL 2 OF 3 : DOUBLE DQN
# ==========================================================
#
# KEY DIFFERENCE VS VANILLA DQN (Van Hasselt et al., 2016):
# The ONLINE network SELECTS the best next action and the
# TARGET network EVALUATES it, reducing the Q value
# overestimation bias of vanilla DQN.
# ==========================================================
#
# OPTIMIZATIONS VS ORIGINAL:
# 1. SAMPLED EPISODES : each episode trains on a random
#    sample of SAMPLE_SIZE rows instead of all 1.28M rows,
#    giving a meaningful reward curve across episodes.
# 2. REPLAY_FREQUENCY : network trains once every 4 steps
#    (standard practice, Mnih et al. 2015) instead of every
#    step.
# 3. SLOW EPSILON DECAY : exploration is scheduled to span
#    roughly the first third of training instead of ending
#    within the first minutes of episode 1.
# 4. BATCHED EVALUATION : the test set is predicted in one
#    GPU forward pass instead of a 320k-step loop.
# ==========================================================

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os
import time
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# ==========================================================
# DEVICE CONFIGURATION (GPU)
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*80)
print("DEVICE CONFIGURATION")
print("="*80)
print("Device :", DEVICE)
if DEVICE.type == "cuda":
    print("GPU    :", torch.cuda.get_device_name(0))

# ==========================================================
# RANDOM SEED
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==========================================================
# CREATE PROJECT DIRECTORIES
# ==========================================================

os.makedirs("Models/DoubleDQN", exist_ok=True)
os.makedirs("Results/DoubleDQN", exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

DATASET_PATH = "final_preprocessed_dataset.csv"

print("="*80)
print("LOADING DATASET")
print("="*80)

df = pd.read_csv(DATASET_PATH)

print("Dataset Loaded Successfully")
print("Dataset Shape :", df.shape)

print("\nLabel Distribution")
print(df["label"].value_counts())

# ==========================================================
# TRAIN TEST SPLIT (SAME SEED ACROSS ALL 3 MODELS)
# ==========================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=SEED
)

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

print("\nTraining Shape :", train_df.shape)
print("Testing Shape  :", test_df.shape)

# ==========================================================
# STATE FEATURES
# ==========================================================

STATE_FEATURES = [
    "login_events",
    "device_events",
    "file_events",
    "email_events",
    "total_email_size",
    "total_attachments",
    "https_events",
    "O", "C", "E", "A", "N",
    "month", "day", "hour", "weekday",
    "is_weekend", "working_hours",
    "security_score"
]

STATE_SIZE = len(STATE_FEATURES)
print("\nState Size :", STATE_SIZE)

# ==========================================================
# ACTION SPACE
# ==========================================================

ACTIONS = {
    0 : "Allow",
    1 : "Monitor",
    2 : "Warn",
    3 : "Block"
}

ACTION_SIZE = len(ACTIONS)
print("\nAction Space")
print(ACTIONS)

# ==========================================================
# REWARD MATRIX
# ==========================================================

REWARD_MATRIX = {
    # True Secure
    0:{0:150,1:-25,2:-75,3:-150},
    # True Suspicious
    1:{0:-40,1:100,2:-25,3:-75},
    # True Unsecure
    2:{0:-80,1:-40,2:50,3:-25},
    # True Breached
    3:{0:-200,1:-150,2:-75,3:0}
}

def calculate_reward(predicted_action, true_state):
    return REWARD_MATRIX[true_state][predicted_action]

# ==========================================================
# HYPERPARAMETERS (IDENTICAL ACROSS ALL 3 MODELS)
# ==========================================================

EPISODES = 25

# Rows randomly sampled from the training set per episode
SAMPLE_SIZE = 100000

BATCH_SIZE = 64
GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 50000

EPSILON = 1.0
EPSILON_MIN = 0.01

# Decay applied once per replay call.
# Replay calls per episode = SAMPLE_SIZE / REPLAY_FREQUENCY = 25000.
# 0.99998 reaches EPSILON_MIN after ~230000 calls, i.e. around
# episode 9 of 25, so exploration spans the first third of training.
EPSILON_DECAY = 0.99998

TARGET_UPDATE = 5

# Train the network once every N environment steps (Mnih et al. 2015)
REPLAY_FREQUENCY = 4

print("\nHyperparameters")
print("Episodes         :", EPISODES)
print("Sample Size      :", SAMPLE_SIZE)
print("Batch Size       :", BATCH_SIZE)
print("Gamma            :", GAMMA)
print("Learning Rate    :", LEARNING_RATE)
print("Memory Size      :", MEMORY_SIZE)
print("Initial Epsilon  :", EPSILON)
print("Epsilon Decay    :", EPSILON_DECAY)
print("Replay Frequency :", REPLAY_FREQUENCY)
print("Target Update    :", TARGET_UPDATE)

print("="*80)
print("PART 1 COMPLETED")
print("="*80)

# ==========================================================
# PART 2A
# DEEP Q NETWORK + ENVIRONMENT
# ==========================================================

class DQNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_SIZE)
        )

    def forward(self, x):
        return self.network(x)


def build_model():
    return DQNetwork().to(DEVICE)


print("="*80)
print("DOUBLE DQN NETWORK CREATED")
print("="*80)

temp_model = build_model()
print(temp_model)
print("Total Parameters :", sum(p.numel() for p in temp_model.parameters()))
del temp_model

# ==========================================================
# CYBER SECURITY ENVIRONMENT (WITH PER-EPISODE SAMPLING)
# ==========================================================

class CyberSecurityEnv:

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

        # Pre-extract as numpy arrays for speed
        self.states = self.df[STATE_FEATURES].values.astype(np.float32)
        self.labels = self.df["label"].values.astype(int)

        self.ep_states = self.states
        self.ep_labels = self.labels
        self.max_steps = len(self.df)
        self.current_step = 0

    # ------------------------------------------------------
    # reset(sample_size=None) -> full data
    # reset(sample_size=N)    -> random sample of N rows
    # ------------------------------------------------------

    def reset(self, sample_size=None):

        if sample_size is not None and sample_size < len(self.states):
            idx = np.random.choice(
                len(self.states),
                size=sample_size,
                replace=False
            )
            self.ep_states = self.states[idx]
            self.ep_labels = self.labels[idx]
        else:
            self.ep_states = self.states
            self.ep_labels = self.labels

        self.max_steps = len(self.ep_states)
        self.current_step = 0

        return self.ep_states[self.current_step]

    # ------------------------------------------------------

    def step(self, action):

        true_state = int(self.ep_labels[self.current_step])
        reward = calculate_reward(action, true_state)

        self.current_step += 1
        done = self.current_step >= self.max_steps

        if done:
            next_state = np.zeros(STATE_SIZE, dtype=np.float32)
        else:
            next_state = self.ep_states[self.current_step]

        return next_state, reward, done, {}

# ==========================================================
# CREATE TRAIN & TEST ENVIRONMENTS
# ==========================================================

train_env = CyberSecurityEnv(train_df)
test_env = CyberSecurityEnv(test_df)

print("="*80)
print("ENVIRONMENT READY")
print("="*80)
print("Training Samples :", len(train_env.states))
print("Testing Samples  :", len(test_env.states))
print("Steps Per Episode:", SAMPLE_SIZE)

# ==========================================================
# PART 2B
# DQN AGENT
# ==========================================================

class DoubleDQNAgent:

    def __init__(self):

        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE

        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE

        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = build_model()
        self.target_model = build_model()
        self.update_target_network()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        self.loss_fn = nn.MSELoss()

    # ======================================================
    # Copy Main Network -> Target Network
    # ======================================================

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # ======================================================
    # Store Experience
    # ======================================================

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # ======================================================
    # Epsilon Greedy Action Selection
    # ======================================================

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)
            q_values = self.model(state_tensor)

        return int(torch.argmax(q_values[0]).item())

    # ======================================================
    # Experience Replay (DOUBLE DQN TARGET)
    # ======================================================

    def replay(self):

        if len(self.memory) < BATCH_SIZE:
            return None

        minibatch = random.sample(self.memory, BATCH_SIZE)

        states = torch.as_tensor(
            np.array([s[0] for s in minibatch]),
            dtype=torch.float32, device=DEVICE
        )
        actions = torch.as_tensor(
            np.array([s[1] for s in minibatch]),
            dtype=torch.long, device=DEVICE
        )
        rewards = torch.as_tensor(
            np.array([s[2] for s in minibatch]),
            dtype=torch.float32, device=DEVICE
        )
        next_states = torch.as_tensor(
            np.array([s[3] for s in minibatch]),
            dtype=torch.float32, device=DEVICE
        )
        dones = torch.as_tensor(
            np.array([s[4] for s in minibatch]),
            dtype=torch.bool, device=DEVICE
        )

        # Current Q Values
        self.model.train()
        q_values = self.model(states)

        # ---------------- DOUBLE DQN TARGET ----------------
        # 1. ONLINE network SELECTS best next action
        # 2. TARGET network EVALUATES that action's Q value
        # ----------------------------------------------------
        with torch.no_grad():
            next_q_online = self.model(next_states)
            best_next_actions = next_q_online.argmax(dim=1)

            next_q_target = self.target_model(next_states)
            evaluated_q = next_q_target.gather(
                1, best_next_actions.unsqueeze(1)
            ).squeeze(1)

            target_q = rewards + self.gamma * evaluated_q * (~dones)

        target = q_values.detach().clone()
        target[torch.arange(BATCH_SIZE, device=DEVICE), actions] = target_q

        # Train Network
        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

# ==========================================================
# CREATE AGENT
# ==========================================================

agent = DoubleDQNAgent()

print("="*80)
print("DOUBLE DQN AGENT INITIALIZED")
print("="*80)
print("State Size      :", agent.state_size)
print("Action Size     :", agent.action_size)
print("Memory Size     :", MEMORY_SIZE)
print("Discount Factor :", agent.gamma)
print("Learning Rate   :", agent.learning_rate)
print("Initial Epsilon :", agent.epsilon)

# ==========================================================
# PART 3A
# TRAINING
# ==========================================================

print("=" * 80)
print("DOUBLE DQN TRAINING STARTED")
print("=" * 80)

reward_history = []
loss_history = []
epsilon_history = []

best_reward = -999999999
best_episode = 0

start_time = time.time()

for episode in range(1, EPISODES + 1):

    state = train_env.reset(sample_size=SAMPLE_SIZE)
    done = False
    episode_reward = 0
    episode_loss = []
    step_counter = 0

    print("\n" + "=" * 80)
    print(f"Episode {episode}/{EPISODES}")
    print("=" * 80)

    progress = tqdm(
        total=train_env.max_steps,
        desc=f"Episode {episode}",
        unit="step"
    )

    while not done:

        # Select Action
        action = agent.act(state)

        # Environment Step
        next_state, reward, done, _ = train_env.step(action)

        # Store Transition
        agent.remember(state, action, reward, next_state, done)

        # Train Agent (every REPLAY_FREQUENCY steps)
        if step_counter % REPLAY_FREQUENCY == 0:
            loss = agent.replay()
            if loss is not None:
                episode_loss.append(float(loss))

        # Update State
        state = next_state
        episode_reward += reward
        step_counter += 1
        progress.update(1)

    progress.close()

    # Episode Statistics
    average_loss = np.mean(episode_loss) if len(episode_loss) > 0 else 0
    reward_history.append(episode_reward)
    loss_history.append(average_loss)
    epsilon_history.append(agent.epsilon)

    # Update Target Network
    if episode % TARGET_UPDATE == 0:
        agent.update_target_network()
        print("Target Network Updated")

    # Save Best Model
    if episode_reward > best_reward:
        best_reward = episode_reward
        best_episode = episode
        torch.save(
            agent.model.state_dict(),
            "Models/DoubleDQN/best_double_dqn_model.pth"
        )
        print("Best Model Saved")

    # Episode Summary
    print("-" * 60)
    print("Episode           :", episode)
    print("Steps             :", step_counter)
    print("Episode Reward    :", round(episode_reward, 2))
    print("Average Loss      :", round(average_loss, 6))
    print("Replay Memory     :", len(agent.memory))
    print("Current Epsilon   :", round(agent.epsilon, 4))
    print("-" * 60)

# ==========================================================
# TRAINING COMPLETED
# ==========================================================

training_time = time.time() - start_time

print("\n")
print("=" * 80)
print("TRAINING COMPLETED")
print("=" * 80)
print("Best Episode      :", best_episode)
print("Best Reward       :", best_reward)
print("Training Time     :", round(training_time, 2), "Seconds")
print("Final Epsilon     :", round(agent.epsilon, 4))

# ==========================================================
# SAVE TRAINING CURVES (FOR REPORT PLOTS)
# ==========================================================

pd.DataFrame({
    "Episode": range(1, EPISODES + 1),
    "Reward": reward_history,
    "AvgLoss": loss_history,
    "Epsilon": epsilon_history
}).to_csv("Results/DoubleDQN/DoubleDQN_TrainingHistory.csv", index=False)

print("Training History Saved")

# ==========================================================
# PART 3B
# EVALUATION (BATCHED - SINGLE GPU FORWARD PASS)
# ==========================================================
# Actions do not affect the next state in this environment,
# so the full test set can be predicted in one shot instead
# of stepping through 320k rows one at a time.
# ==========================================================

print("="*80)
print("MODEL EVALUATION (BATCHED)")
print("="*80)

eval_start = time.time()

agent.model.eval()

y_pred = []

with torch.no_grad():

    # Chunk to stay well within VRAM even for huge test sets
    CHUNK = 200000

    for i in range(0, len(test_env.states), CHUNK):

        chunk_tensor = torch.as_tensor(
            test_env.states[i:i+CHUNK],
            dtype=torch.float32,
            device=DEVICE
        )

        chunk_pred = agent.model(chunk_tensor).argmax(dim=1)

        y_pred.extend(chunk_pred.cpu().numpy().tolist())

y_true = test_env.labels.tolist()

test_reward = sum(
    calculate_reward(p, t) for p, t in zip(y_pred, y_true)
)

print("Evaluation Time :", round(time.time() - eval_start, 2), "Seconds")

# ==========================================================
# PERFORMANCE METRICS
# ==========================================================

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("\n")
print("="*80)
print("MODEL PERFORMANCE (DOUBLE DQN)")
print("="*80)
print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"Total Reward  : {test_reward}")
print("="*80)

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\n")
print("="*80)
print("CLASSIFICATION REPORT")
print("="*80)

print(classification_report(
    y_true,
    y_pred,
    target_names=["Secure", "Suspicious", "Unsecure", "Breached"],
    zero_division=0
))

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(y_true, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Secure", "Suspicious", "Unsecure", "Breached"],
    columns=["Secure", "Suspicious", "Unsecure", "Breached"]
)

print("\n")
print("="*80)
print("CONFUSION MATRIX")
print("="*80)
print(cm_df)

# ==========================================================
# SAVE METRICS
# ==========================================================

metrics = pd.DataFrame({
    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Test Reward",
        "Training Time (s)"
    ],
    "Value":[
        accuracy,
        precision,
        recall,
        f1,
        test_reward,
        round(training_time, 2)
    ]
})

metrics.to_csv("Results/DoubleDQN/DoubleDQN_Metrics.csv", index=False)
print("\nMetrics Saved Successfully")

# ==========================================================
# SAVE PREDICTIONS
# ==========================================================

predictions = pd.DataFrame({
    "Actual": y_true,
    "Predicted": y_pred
})

predictions.to_csv("Results/DoubleDQN/DoubleDQN_Predictions.csv", index=False)
print("Predictions Saved Successfully")
