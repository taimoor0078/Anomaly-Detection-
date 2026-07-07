# ==========================================================
# INSIDER THREAT DETECTION USING DEEP Q NETWORK (DQN)
# PYTORCH VERSION (GPU / CUDA)
# PART 1 : IMPORTS + DATASET + SETTINGS
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

os.makedirs("Models/DQN", exist_ok=True)
os.makedirs("Results/DQN", exist_ok=True)

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

# ==========================================================
# DATASET INFORMATION
# ==========================================================

print("\nColumns")
print(df.columns.tolist())

print("\nMissing Values")
print(df.isnull().sum())

print("\nSecurity Distribution")
print(df["security_status"].value_counts())

print("\nLabel Distribution")
print(df["label"].value_counts())

# ==========================================================
# TRAIN TEST SPLIT
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

    "O",
    "C",
    "E",
    "A",
    "N",

    "month",
    "day",
    "hour",
    "weekday",

    "is_weekend",
    "working_hours",

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
# DQN HYPERPARAMETERS
# ==========================================================

EPISODES = 50

BATCH_SIZE = 64

GAMMA = 0.95

LEARNING_RATE = 0.001

MEMORY_SIZE = 50000

EPSILON = 1.0

EPSILON_MIN = 0.01

EPSILON_DECAY = 0.995

TARGET_UPDATE = 10

print("\nHyperparameters")

print("Episodes        :", EPISODES)
print("Batch Size      :", BATCH_SIZE)
print("Gamma           :", GAMMA)
print("Learning Rate   :", LEARNING_RATE)
print("Memory Size     :", MEMORY_SIZE)
print("Initial Epsilon :", EPSILON)

# ==========================================================
# REPLAY MEMORY
# ==========================================================

memory = deque(maxlen=MEMORY_SIZE)

print("\nReplay Memory Created")

print("Capacity :", MEMORY_SIZE)

print("="*80)
print("PART 1 COMPLETED")
print("="*80)

# ==========================================================
# PART 2A
# DEEP Q NETWORK + ENVIRONMENT
# ==========================================================

# ==========================================================
# BUILD DQN MODEL
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

    model = DQNetwork().to(DEVICE)

    return model


print("="*80)
print("DQN NETWORK CREATED")
print("="*80)

temp_model = build_model()
print(temp_model)

total_params = sum(p.numel() for p in temp_model.parameters())
print("Total Parameters :", total_params)

del temp_model

# ==========================================================
# CYBER SECURITY ENVIRONMENT
# ==========================================================

class CyberSecurityEnv:

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)

        self.max_steps = len(self.df)

        self.current_step = 0

        # Pre-extract as numpy arrays for speed
        self.states = self.df[STATE_FEATURES].values.astype(np.float32)
        self.labels = self.df["label"].values.astype(int)

    # ------------------------------------------------------

    def reset(self):

        self.current_step = 0

        state = self.states[self.current_step]

        return state

    # ------------------------------------------------------

    def step(self, action):

        true_state = int(self.labels[self.current_step])

        reward = calculate_reward(

            action,
            true_state

        )

        self.current_step += 1

        done = self.current_step >= self.max_steps

        if done:

            next_state = np.zeros(

                STATE_SIZE,

                dtype=np.float32

            )

        else:

            next_state = self.states[self.current_step]

        return (

            next_state,

            reward,

            done,

            {}

        )

# ==========================================================
# CREATE TRAIN & TEST ENVIRONMENTS
# ==========================================================

train_env = CyberSecurityEnv(train_df)

test_env = CyberSecurityEnv(test_df)

print("="*80)
print("ENVIRONMENT READY")
print("="*80)

print("Training Samples :", train_env.max_steps)

print("Testing Samples  :", test_env.max_steps)

print("="*80)
print("PART 2A COMPLETED")
print("="*80)

# ==========================================================
# PART 2B
# DQN AGENT
# ==========================================================

class DQNAgent:

    def __init__(self):

        # -----------------------------
        # State & Action
        # -----------------------------

        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE

        # -----------------------------
        # Hyperparameters
        # -----------------------------

        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE

        # -----------------------------
        # Replay Memory
        # -----------------------------

        self.memory = deque(maxlen=MEMORY_SIZE)

        # -----------------------------
        # Main & Target Networks
        # -----------------------------

        self.model = build_model()

        self.target_model = build_model()

        self.update_target_network()

        # -----------------------------
        # Optimizer & Loss (MSE + Adam, same as Keras version)
        # -----------------------------

        self.optimizer = optim.Adam(

            self.model.parameters(),

            lr=self.learning_rate

        )

        self.loss_fn = nn.MSELoss()

    # ======================================================
    # Copy Main Network -> Target Network
    # ======================================================

    def update_target_network(self):

        self.target_model.load_state_dict(

            self.model.state_dict()

        )

    # ======================================================
    # Store Experience
    # ======================================================

    def remember(

        self,

        state,

        action,

        reward,

        next_state,

        done

    ):

        self.memory.append(

            (

                state,

                action,

                reward,

                next_state,

                done

            )

        )

    # ======================================================
    # Epsilon Greedy Action Selection
    # ======================================================

    def act(self, state):

        if np.random.rand() <= self.epsilon:

            return random.randint(

                0,

                self.action_size - 1

            )

        with torch.no_grad():

            state_tensor = torch.as_tensor(

                state,

                dtype=torch.float32,

                device=DEVICE

            ).unsqueeze(0)

            q_values = self.model(state_tensor)

        return int(

            torch.argmax(

                q_values[0]

            ).item()

        )

    # ======================================================
    # Experience Replay
    # ======================================================

    def replay(self):

        if len(self.memory) < BATCH_SIZE:

            return None

        minibatch = random.sample(

            self.memory,

            BATCH_SIZE

        )

        states = torch.as_tensor(

            np.array([sample[0] for sample in minibatch]),

            dtype=torch.float32,

            device=DEVICE

        )

        actions = torch.as_tensor(

            np.array([sample[1] for sample in minibatch]),

            dtype=torch.long,

            device=DEVICE

        )

        rewards = torch.as_tensor(

            np.array([sample[2] for sample in minibatch]),

            dtype=torch.float32,

            device=DEVICE

        )

        next_states = torch.as_tensor(

            np.array([sample[3] for sample in minibatch]),

            dtype=torch.float32,

            device=DEVICE

        )

        dones = torch.as_tensor(

            np.array([sample[4] for sample in minibatch]),

            dtype=torch.bool,

            device=DEVICE

        )

        # Current Q Values

        self.model.train()

        q_values = self.model(states)

        # Target Q Values (Bellman Update, vectorized)

        with torch.no_grad():

            target_next = self.target_model(next_states)

            max_next_q = target_next.max(dim=1).values

            target_q = rewards + self.gamma * max_next_q * (~dones)

        # Build full target matrix (same as Keras train_on_batch behavior)

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

agent = DQNAgent()

print("="*80)
print("DQN AGENT INITIALIZED")
print("="*80)

print("State Size      :", agent.state_size)

print("Action Size     :", agent.action_size)

print("Memory Size     :", MEMORY_SIZE)

print("Discount Factor :", agent.gamma)

print("Learning Rate   :", agent.learning_rate)

print("Initial Epsilon :", agent.epsilon)

print("="*80)
print("="*80)

# ==========================================================
# PART 3A
# DQN TRAINING
# ==========================================================

print("=" * 80)
print("DQN TRAINING STARTED")
print("=" * 80)

# ----------------------------------------------------------
# Training Variables
# ----------------------------------------------------------

reward_history = []
loss_history = []

best_reward = -999999999
best_episode = 0

start_time = time.time()

# ==========================================================
# TRAINING LOOP
# ==========================================================

for episode in range(1, EPISODES + 1):

    state = train_env.reset()

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

        # --------------------------------------------------
        # Select Action
        # --------------------------------------------------

        action = agent.act(state)

        # --------------------------------------------------
        # Environment Step
        # --------------------------------------------------

        next_state, reward, done, _ = train_env.step(action)

        # --------------------------------------------------
        # Store Transition
        # --------------------------------------------------

        agent.remember(

            state,

            action,

            reward,

            next_state,

            done

        )

        # --------------------------------------------------
        # Train Agent
        # --------------------------------------------------

        loss = agent.replay()

        if loss is not None:

            episode_loss.append(float(loss))

        # --------------------------------------------------
        # Update State
        # --------------------------------------------------

        state = next_state

        episode_reward += reward

        step_counter += 1

        progress.update(1)

    progress.close()

    # ======================================================
    # Episode Statistics
    # ======================================================

    average_loss = (

        np.mean(episode_loss)

        if len(episode_loss) > 0

        else 0

    )

    reward_history.append(

        episode_reward

    )

    loss_history.append(

        average_loss

    )

    # ======================================================
    # Update Target Network
    # ======================================================

    if episode % TARGET_UPDATE == 0:

        agent.update_target_network()

        print("Target Network Updated")

    # ======================================================
    # Save Best Model
    # ======================================================

    if episode_reward > best_reward:

        best_reward = episode_reward

        best_episode = episode

        torch.save(

            agent.model.state_dict(),

            "Models/DQN/best_dqn_model.pth"

        )

        print("Best Model Saved")

    # ======================================================
    # Episode Summary
    # ======================================================

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
# PART 3B
# DQN MODEL EVALUATION
# ==========================================================

print("="*80)
print("MODEL EVALUATION")
print("="*80)

y_true = []
y_pred = []

state = test_env.reset()

done = False

test_reward = 0

agent.model.eval()

progress = tqdm(
    total=test_env.max_steps,
    desc="Testing",
    unit="step"
)

while not done:

    # ---------------------------------------------
    # True Label
    # ---------------------------------------------

    true_label = int(

        test_env.labels[test_env.current_step]

    )

    # ---------------------------------------------
    # Predict Action
    # ---------------------------------------------

    with torch.no_grad():

        state_tensor = torch.as_tensor(

            state,

            dtype=torch.float32,

            device=DEVICE

        ).unsqueeze(0)

        q_values = agent.model(state_tensor)

    predicted_action = int(torch.argmax(q_values[0]).item())

    # ---------------------------------------------
    # Save Labels
    # ---------------------------------------------

    y_true.append(true_label)

    y_pred.append(predicted_action)

    # ---------------------------------------------
    # Environment Step
    # ---------------------------------------------

    state,reward,done,_ = test_env.step(predicted_action)

    test_reward += reward

    progress.update(1)

progress.close()

# ==========================================================
# PERFORMANCE METRICS
# ==========================================================

accuracy = accuracy_score(

    y_true,

    y_pred

)

precision = precision_score(

    y_true,

    y_pred,

    average="weighted",

    zero_division=0

)

recall = recall_score(

    y_true,

    y_pred,

    average="weighted",

    zero_division=0

)

f1 = f1_score(

    y_true,

    y_pred,

    average="weighted",

    zero_division=0

)

# ==========================================================
# PRINT RESULTS
# ==========================================================

print("\n")

print("="*80)

print("MODEL PERFORMANCE")

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

print(

classification_report(

    y_true,

    y_pred,

    target_names=[

        "Secure",

        "Suspicious",

        "Unsecure",

        "Breached"

    ],

    zero_division=0

)

)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(

    y_true,

    y_pred

)

cm_df = pd.DataFrame(

    cm,

    index=[

        "Secure",

        "Suspicious",

        "Unsecure",

        "Breached"

    ],

    columns=[

        "Secure",

        "Suspicious",

        "Unsecure",

        "Breached"

    ]

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

        "Test Reward"

    ],

    "Value":[

        accuracy,

        precision,

        recall,

        f1,

        test_reward

    ]

})

metrics.to_csv(

    "Results/DQN/DQN_Metrics.csv",

    index=False

)

print("\nMetrics Saved Successfully")

# ==========================================================
# SAVE PREDICTIONS
# ==========================================================

predictions = pd.DataFrame({

    "Actual":y_true,

    "Predicted":y_pred

})

predictions.to_csv(

    "Results/DQN/DQN_Predictions.csv",

    index=False

)

print("Predictions Saved Successfully")
