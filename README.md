Deep Reinforcement Learning-Based Anomaly Detection

Anomaly Detection with Learning Agents Using DQN, Double DQN, and Dueling DQN

This project presents a Deep Reinforcement Learning (DRL)-based approach for insider threat anomaly detection using the CERT Insider Threat Dataset.

The project formulates insider threat detection as a reinforcement learning problem in which user activities are represented as environment states, detection decisions are actions, and rewards/penalties guide the learning process.

Three value-based Deep Reinforcement Learning algorithms are implemented and compared:

Deep Q-Network (DQN)

Double Deep Q-Network (Double DQN)

Dueling Deep Q-Network (Dueling DQN)

The project evaluates the models using detection accuracy, reward behavior, training loss, and convergence stability.

Project Objective

The main objective is to design a Deep Reinforcement Learning framework that can learn user behavior patterns and detect potentially anomalous insider activities.

Specific objectives include:

Preprocess and analyse CERT user activity data.

Formulate insider threat detection as an RL problem.

Define the RL environment, state space, action space, and reward mechanism.

Implement DQN, Double DQN, and Dueling DQN.

Train agents to distinguish normal and suspicious user behavior.

Compare the models using detection accuracy, cumulative rewards, training loss, and convergence behaviour.

Analyse the strengths and limitations of different DQN architectures.

Identify future improvements for adaptive cybersecurity systems.

Problem Statement

Insider threat detection is difficult because malicious activity can originate from users who already have legitimate access to organizational systems.

Suspicious behaviour may also resemble normal employee activity, while user behaviour can change over time. Traditional rule-based and static machine-learning approaches may therefore struggle with evolving behaviour and previously unseen activity patterns.

This project treats anomaly detection as a sequential decision-making problem, allowing an RL agent to learn from interaction and reward feedback.

Why Reinforcement Learning?

Traditional anomaly detection approaches generally learn patterns from historical data or depend on predefined rules.

Reinforcement Learning instead allows an agent to:

Observe the current environment state

Select an action

Receive a reward or penalty

Update its policy

Improve decisions over repeated interactions

This makes the RL formulation suitable for the project's objective of adaptive insider-threat anomaly detection.

Deep Reinforcement Learning extends this approach by using neural networks to approximate Q-values and process more complex behavioural state representations.

System Architecture

The project architecture follows this workflow:

                  ┌─────────────────────────┐
                  │ CERT Insider Threat     │
                  │ Dataset                 │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Data Preprocessing      │
                  │                         │
                  │ • Cleaning              │
                  │ • Aggregation           │
                  │ • Encoding              │
                  │ • Normalisation         │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ RL Environment Creation │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ State Representation    │
                  │ User Behaviour Features │
                  └────────────┬────────────┘
                               │
                               ▼
             ┌─────────────────┴─────────────────┐
             │                                   │
             ▼                                   ▼
      ┌───────────────┐                  ┌────────────────┐
      │ DQN           │                  │ Double DQN     │
      │ Agent         │                  │ Agent          │
      └───────┬───────┘                  └───────┬────────┘
              │                                  │
              └──────────────┬───────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Dueling DQN      │
                    │ Agent            │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Action Selection │
                    │ Normal / Anomaly │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Reward / Penalty │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Policy / Q-value │
                    │ Update           │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Model Evaluation │
                    │ Accuracy         │
                    │ Reward           │
                    │ Loss             │
                    │ Convergence      │
                    └──────────────────┘

The project report's flowchart also represents the pipeline from CERT data and preprocessing through RL environment creation, state representation, agent training, epsilon-greedy action selection, detection decision, reward calculation, policy optimisation, and performance evaluation.

Dataset

The project uses the CERT Insider Threat Dataset, which is designed to support research into abnormal and potentially malicious activities performed by authorised users.

The dataset contains multiple types of user activity and behavioural information.

The project combines activity sources into a structured feature-based dataset suitable for reinforcement learning.

Raw Data Sources

1. Logon Activity

Logon records represent user authentication behaviour and can help identify unusual access patterns.

2. Device Activity

Device logs capture interactions with external devices. Unusual device usage may indicate suspicious data movement.

3. File Activity

File logs represent interactions with organizational files. Unusual or high-volume file access may be an indicator of insider threat behaviour.

4. Email Activity

Email behaviour is analysed through communication frequency, attachments, and other email patterns.

5. Web Activity

HTTP activity represents web browsing behaviour and access patterns.

6. Psychometric Data

The dataset also includes personality-related indicators:

Openness (O)

Conscientiousness (C)

Extraversion (E)

Agreeableness (A)

Neuroticism (N)

Feature Engineering

After preprocessing, multiple activity sources are combined into behavioural states.

The reported state features include:

Feature

Description

user

Unique user identifier

hour_window

Aggregated time interval representing user activity

login_events

Number of authentication activities

device_events

Device usage frequency

file_events

Number of file access operations

email_events

Email communication activity count

total_email_size

Total size of emails exchanged

total_attachments

Number of email attachments

https_events

Web browsing activity count

O

Openness personality score

C

Conscientiousness personality score

E

Extraversion personality score

A

Agreeableness personality score

N

Neuroticism personality score

These features form the state representation supplied to the reinforcement learning environment.

Data Preparation

The reported dataset preparation process includes:

Combining multiple user activity sources.

Removing missing or inconsistent values.

Aggregating behavioural activities.

Encoding features.

Normalising features.

Preparing numerical state representations for RL training.

The processed dataset is then used as input to the RL environment.

Reinforcement Learning Formulation

The project represents insider threat detection using the standard RL components:

RL Component

Project Definition

Agent

DQN, Double DQN, Dueling DQN

Environment

CERT user behaviour data

State

Behavioural features

Action

Normal or anomalous classification

Reward

Positive/negative feedback based on detection correctness

Policy

Strategy learned to maximise future rewards

Environment Design

Each user activity record represents a situation observed by the agent.

At every decision step:

Current User Behaviour State
            │
            ▼
       RL Agent
            │
            ▼
    Select Action
     /          \
    /            \
Normal         Anomaly
    \            /
     \          /
       Evaluation
            │
            ▼
       Reward / Penalty
            │
            ▼
        Next State
            │
            ▼
       Policy Update

The agent continuously interacts with the environment and updates its Q-values based on received feedback.

State Space

The state represents the behavioural information available to the agent at a particular time step.

The project defines the state conceptually as:

Sₜ = {x₁, x₂, x₃, ..., xₙ}

where each x represents an individual behavioural feature.

State features include:

Login events

Device events

File events

Email events

Total email size

Total attachments

HTTPS events

Openness

Conscientiousness

Extraversion

Agreeableness

Neuroticism

Action Space

The project uses a discrete binary action space.

Action

Meaning

0

Normal user behaviour

1

Anomalous / suspicious behaviour

The agent selects actions based on learned Q-values.

The policy is represented as:

π(s) = argmaxₐ Q(s, a)

This means the agent selects the action with the highest expected future reward.

Epsilon-Greedy Exploration

During training, an epsilon-greedy strategy is used.

Initially, the agent explores different actions. As training progresses, exploration decreases and the agent increasingly selects actions based on learned Q-values.

High Exploration
      │
      ▼
Try Different Actions
      │
      ▼
Learn Q-values
      │
      ▼
Reduce Exploration
      │
      ▼
More Exploitation
      │
      ▼
Learned Detection Policy

Reward Function

The reward function is based on the correctness of anomaly detection.

R(s, a) =
    +1   Correct Detection
    -1   Incorrect Detection

The reward strategy is:

Actual Behaviour

Agent Decision

Result

Reward

Normal

Normal

Correct

+1

Anomaly

Anomaly

Correct

+1

Normal

Anomaly

False Positive

-1

Anomaly

Normal

False Negative

-1

The objective is to maximise cumulative reward over time.

The cumulative reward is represented conceptually as:

Gₜ = Σ γᵗ Rₜ

where γ controls the importance of future rewards.

DQN

Deep Q-Network

DQN is used as the baseline reinforcement learning model.

Instead of storing Q-values in a traditional Q-table, a neural network approximates the Q-value function.

The agent:

Observes the current state.

Selects an action.

Receives a reward.

Moves to the next state.

Updates its Q-value estimates.

Repeats the process.

The project uses:

Experience replay

Target network

Epsilon-greedy exploration

Experience Replay

Previous experiences containing state, action, reward, and next-state information are stored in memory. Random samples are used during training to improve learning stability.

Target Network

A separate target network provides more stable Q-value targets and is periodically updated.

Q-Learning

The project describes the Q-learning update conceptually as:

Q(s, a) = r + γ max Q(s', a')

where:

s = current state

a = selected action

r = reward

s' = next state

γ = discount factor

Double DQN

Double DQN is implemented to reduce Q-value overestimation.

In standard DQN, the same network is involved in action selection and evaluation. Double DQN separates these processes.

Current State
      │
      ▼
Online Network
      │
      ▼
Select Best Action
      │
      ▼
Target Network
      │
      ▼
Evaluate Selected Action
      │
      ▼
More Reliable Q-value

Main Components

Online Network: selects the best action.

Target Network: evaluates the selected action.

This separation is designed to reduce estimation bias and improve learning stability.

Dueling DQN

Dueling DQN introduces two separate learning streams:

State Value Function V(s)

Advantage Function A(s,a)

The project describes the final Q-value as:

Q(s, a) = V(s) + A(s, a)

The architecture therefore learns:

How important the current user behaviour state is.

Which action provides the highest advantage.

Conceptually:

                 Input State
                     │
                     ▼
              Shared Network
                /          \
               /            \
              ▼              ▼
       Value Stream     Advantage Stream
          V(s)             A(s,a)
              \            /
               \          /
                ▼        ▼
                Combine
                   │
                   ▼
                Q(s,a)
                   │
                   ▼
              Final Action

This structure is intended to improve representation learning and decision quality in complex behavioural states.

Comparison of the Three Algorithms

Model

Main Purpose

DQN

Baseline anomaly detection

Double DQN

Reduce Q-value overestimation

Dueling DQN

Separate state value and action advantage learning

Experimental Setup

The experiments use the processed CERT dataset and apply the same general state representation, action space, and reward mechanism to the three agents.

Implementation Environment

Component

Technology

Programming Language

Python

Deep Learning Framework

PyTorch

Data Processing

Pandas, NumPy

Visualisation

Matplotlib

Dataset

CERT Insider Threat Dataset

Models

DQN, Double DQN, Dueling DQN

Optimizer

Adam

Exploration

Epsilon-Greedy

Training Process

The reported training workflow is:

1. Initialise RL environment
             │
             ▼
2. Initialise DQN-based agent
             │
             ▼
3. Observe current user state
             │
             ▼
4. Select action using epsilon-greedy
             │
             ▼
5. Receive reward
             │
             ▼
6. Store experience in replay memory
             │
             ▼
7. Sample experiences
             │
             ▼
8. Train neural network
             │
             ▼
9. Update policy
             │
             ▼
10. Repeat until training completion

Hyperparameters

The project identifies the following main training hyperparameters:

Hyperparameter

Purpose

Learning Rate

Controls neural-network weight updates

Discount Factor (γ)

Determines the importance of future rewards

Batch Size

Number of experiences used during training

Replay Memory

Stores previous agent experiences

Epsilon

Controls exploration vs exploitation

Target Update Frequency

Controls target-network updates

Evaluation Metrics

The RL models are evaluated using multiple measures.

Accuracy

Measures the percentage of correctly detected normal and anomalous behaviours.

Accuracy =
Correct Predictions
────────────────────
Total Predictions

Cumulative Reward

Measures how effectively the agent improves its decisions through training.

Total Reward = Σ Rₜ

Higher cumulative reward indicates improved reward-based learning behaviour.

Training Loss

Training loss measures the difference between predicted Q-values and target Q-values.

Lower loss indicates more stable Q-value estimation according to the project's evaluation framework.

Convergence Behaviour

Convergence analysis evaluates how quickly and consistently the agent reaches stable learning performance.

Experimental Results

DQN

DQN was implemented as the baseline RL agent.

The reported DQN performance is:

Metric

Result

Accuracy

0.999318

Precision

0.999319

Recall

0.999318

F1 Score

0.999317

The report states that DQN learned meaningful behavioural patterns and successfully classified activities as normal or anomalous.

However, standard DQN showed some instability attributed to Q-value overestimation.

Double DQN

Double DQN was designed to reduce the Q-value overestimation problem.

The reported model comparison gives:

Metric

Result

Detection Accuracy

0.999084

Average Reward

12.16 Million

Final Training Loss

171.18

Convergence

More Stable

The project reports that separating action selection and action evaluation improved the stability of learned Q-values.

Dueling DQN

Dueling DQN separately estimates state value and action advantage.

The reported results are:

Metric

Result

Detection Accuracy

0.999763

Average Reward

12.16 Million

Final Training Loss

163.3870

Convergence

Highly Stable

The report states that the architecture provided more detailed representations of user behaviour states and improved decision stability.

Model Comparison

The project's comparison table reports:

Model

Accuracy

Reward Performance

Training Stability

DQN

0.999318

Good

Moderate

Double DQN

0.999084

Better

Stable

Dueling DQN

0.999763

Best/Compare after values

Highly Stable

The project reports that all three models successfully learned anomaly-detection policies.

The reported interpretation is that:

DQN provides a strong baseline but is sensitive to Q-value overestimation.

Double DQN improves Q-value reliability and training stability.

Dueling DQN separately learns state importance and action advantage, supporting more stable decision-making.

Learning Behaviour

During early training, the agents use more exploration through epsilon-greedy action selection.

As training progresses:

Exploration ↓
      │
      ▼
Learned Q-values ↑
      │
      ▼
Exploitation ↑
      │
      ▼
Detection Policy Improves

The report associates increasing rewards and decreasing training loss with improvement in detection policies.

Impact of Double DQN

Standard DQN may produce overly optimistic Q-value estimates because of its action-selection/evaluation mechanism.

Double DQN addresses this by separating:

Action Selection
       │
       ▼
Online Network
       │
       ▼
Selected Action
       │
       ▼
Action Evaluation
       │
       ▼
Target Network

The project reports that this produces more reliable Q-value updates and a more stable detection policy.

Impact of Dueling DQN

Dueling DQN separates:

State Value
    V(s)
     +
Action Advantage
   A(s,a)
     │
     ▼
 Q(s,a)

This allows the model to learn both:

The importance of the current behavioural state.

The relative usefulness of available actions.

The report describes this as useful for complex cybersecurity environments where different behavioural features can contribute differently to anomaly detection.

Performance Trade-offs

The project identifies the following trade-offs:

Model

Strength

Limitation

DQN

Simple architecture and effective baseline learning

Sensitive to Q-value overestimation

Double DQN

More stable learning and reliable Q-value estimation

Slightly increased computational complexity

Dueling DQN

Better understanding of important states and improved decision quality

More complex network architecture

Advanced architectures may therefore require additional computational resources.

Limitations

The project identifies several limitations.

1. Simulated Dataset

The CERT Insider Threat Dataset is a simulated dataset and may not represent every real-world cybersecurity situation.

Real organizations can contain more complex user behaviour and unpredictable attack strategies.

2. Binary Action Space

The current agent only performs two classifications:

Normal

Anomalous

It does not directly select security response actions.

3. Real-Time Deployment

The project focuses on the designed RL environment and experimental evaluation. A production real-time environment would require additional engineering and validation.

4. Reward Design

The reward function is currently based on a simple +1/-1 correctness mechanism. More sophisticated reward functions could incorporate different cybersecurity costs.

5. Computational Complexity

More advanced architectures can require additional computational resources.

Future Improvements

The project identifies several potential future directions:

Advanced RL algorithms such as PPO

Double Dueling DQN

More complex reward functions

Real-time anomaly detection environments

Explainable AI for security interpretation

Expanded security-response action spaces

Risk scoring

Access restriction

Additional authentication

Automated security alerts

A future system could therefore move beyond binary classification toward adaptive security response.

Potential Extended Architecture

A future extension could conceptually evolve the current binary detector into:

User Behaviour Stream
          │
          ▼
   Feature Extraction
          │
          ▼
    RL Anomaly Agent
          │
          ▼
     Risk Assessment
          │
     ┌────┼────┬─────────┐
     ▼    ▼    ▼         ▼
   Allow  MFA  Restrict  Alert
          │
          ▼
   Security Feedback
          │
          ▼
     Reward Update
          │
          ▼
   Adaptive Policy

This represents a future direction described in the report rather than an implemented component of the current project.

Conclusion

This project demonstrates a Deep Reinforcement Learning approach for insider threat anomaly detection using the CERT Insider Threat Dataset.

The system converts user behaviour into RL states, uses binary detection actions, and applies reward feedback to learn an adaptive detection policy.

Three DQN-based architectures were implemented:

DQN as the baseline.

Double DQN to reduce Q-value overestimation.

Dueling DQN to separately learn state values and action advantages.

The reported experiments show high detection performance across the three models. The project also highlights differences in learning stability and architectural complexity.

The reported DQN accuracy is 0.999318, Double DQN detection accuracy is 0.999084, and Dueling DQN detection accuracy is 0.999763.

The project concludes that Deep Reinforcement Learning can provide a foundation for adaptive insider-threat anomaly detection, while future work can extend the system with richer reward functions, real-time monitoring, advanced RL algorithms, broader security actions, and explainable AI.

Technologies

Python

PyTorch

Pandas

NumPy

Matplotlib

Deep Reinforcement Learning

DQN

Double DQN

Dueling DQN

Adam Optimizer

Epsilon-Greedy Exploration

Experience Replay

Target Networks

CERT Insider Threat Dataset




Module: Reinforcement Learning

