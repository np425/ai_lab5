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

REWARDS = np.zeros((model.size, model.size))
TRANSITIONS = np.zeros((model.size, model.size))
VALUES = np.zeros((model.size))

# --------------------------------------- REWARDS --------------------

REWARD_LIVE = -0.5
REWARD_WIN = 10
REWARD_LOSE = -10

for y in range(model.shape[0]):
    for x in range(model.shape[1]):
        place = model[y,x]
        if place == PLACE_WIN:
            reward = REWARD_WIN
        elif place == PLACE_LOSE:
            reward = REWARD_LOSE
        elif place == PLACE_WALL:
            reward = 0
        else:
            reward = REWARD_LIVE

        REWARDS[:, y*model.shape[1]+x] = reward

# --------------------------------------- TRANSITIONS --------------------

for y in range(model.shape[0]):
    for x in range(model.shape[1]):
        places = []
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < model.shape[0] and 0 <= nx < model.shape[1] and model[ny, nx] != PLACE_WALL:
                places.append((dy, dx))
        if places:
            prob = 1 / len(places)
            for dy, dx in places:
                TRANSITIONS[y * model.shape[1] + x, (y + dy) * model.shape[1] + x + dx] = prob


# --------------------------------------- VALUES --------------------

GAMMA = 0.9  # Discount factor
ITERATIONS = 10000

for _ in range(ITERATIONS):
    for i in range(model.size):
        if model.flat[i] in (PLACE_WIN, PLACE_LOSE):
            # No update needed for terminal states
            continue

        max_val = 0
        for j in range(model.size):
            max_val += TRANSITIONS[i, j] * (REWARDS[i, j] + (GAMMA * VALUES[j]))

        VALUES[i] = max_val

# --------------------------------------- VISUALIZATION --------------------

result = VALUES.reshape(model.shape)

plt.imshow(result, cmap='hot')
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        text = plt.text(j, i, str(round(result[i, j], 3)), ha="center", va="center", color="b")
plt.show()

