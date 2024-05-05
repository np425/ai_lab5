from matplotlib import pyplot as plt
import numpy as np

PLACE_WIN = 3
PLACE_LOSE = 2
PLACE_EMPTY = 0
PLACE_WALL = 1

model = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 0, 0, 3],
    [0, 1, 1, 0, 2],
    [0, 0, 0, 0, 0],
])

def is_valid_move(y, x):
    return 0 <= y < model.shape[0] and 0 <= x < model.shape[1] and model[y][x] != PLACE_WALL

Q = np.zeros((model.size, model.size))

REWARD_LIVE = -0.5
REWARD_WIN = 10
REWARD_LOSE = -10


for y in range(model.shape[0]):
    for x in range(model.shape[1]):
        index = y * model.shape[1] + x

        if model[y,x] == PLACE_WALL:
            continue
        elif model[y,x] == PLACE_WIN:
            Q[index, :] = REWARD_WIN
        elif model[y,x] == PLACE_LOSE:
            Q[index, :] = REWARD_LOSE
        else:
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                ny, nx = y + dy, x + dx
                if is_valid_move(ny, nx):
                    next_index = ny * model.shape[1] + nx
                    Q[index, next_index] = REWARD_LIVE


# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.3  # Discount factor
epsilon = 0.2  # Exploration-exploitation trade-off

# Q-learning algorithm with valid move check
def q_learning(Q, num_episodes=100):
    for _ in range(num_episodes):
        state = np.random.randint(0, model.size)  # Start from a random state
        while True:
            valid_actions = [a for a in range(model.size) if is_valid_move(a // model.shape[1], a % model.shape[1])]

            if np.random.rand() < epsilon:
                action = np.random.choice(valid_actions)
            else:
                # Exploit: Choose the valid action with the highest Q-value
                valid_Q_values = [Q[state, a] for a in valid_actions]
                action = valid_actions[np.argmax(valid_Q_values)]

            next_state = action
            reward = Q[state, action]

            # Q-learning update rule
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            if model[state // model.shape[1], state % model.shape[1]] in (PLACE_WIN, PLACE_LOSE):
                break
    return Q


# Run Q-learning
Q = q_learning(Q)

q_values = np.max(Q, axis=1).reshape(model.shape)

plt.imshow(q_values, cmap='hot')
for i in range(model.shape[0]):
    for j in range(model.shape[1]):
        text = plt.text(j, i, str(round(q_values[i, j], 3)), ha="center", va="center", color="b")
plt.show()

