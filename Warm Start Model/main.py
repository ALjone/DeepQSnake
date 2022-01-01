import math
from trainer import DQTrainer
from random import sample
import torch

def makeMap(size):
    game_map = []
    for i in range(size):
        game_map.append([])
        for _ in range(size):
            game_map[i].append(0)
    return game_map

def get_features(game_map):
    features = torch.zeros((3, len(game_map), len(game_map)))
    for i in range(len(game_map)):
        for j in range(len(game_map)):
            tile = game_map[i][j]
            if tile != 0:
                features[tile-1, i, j] = 1
    return features

def distToApple(apple, head):
        a = abs(apple[0]-head[0]) ** 2
        b = abs(apple[1]-head[1]) ** 2
        return math.sqrt(a + b)

trainer = DQTrainer()
#trainer.model = torch.load('warm_start_model')
steps = 1000000
size = 10

X = torch.zeros((steps, 3, size, size))
y = torch.zeros((steps, 5))

for i in range(steps):
    game_map = makeMap(size)
    samples_x = sample(range(0, size), 2)
    samples_y = sample(range(0, size), 2)
    apple = [samples_x[0], samples_y[0]]
    head = [samples_x[1], samples_y[1]]
    game_map[head[0]][head[1]] = 3
    game_map[apple[0]][apple[1]] = 1
    X[i] = get_features(game_map)

    labels = []
    labels.append(distToApple(apple, head))

    if apple[0] > head[0]:
        labels.append(1)
    else:
        labels.append(0)

    if apple[0] < head[0]:
        labels.append(1)
    else:
        labels.append(0)

    if apple[1] > head[1]:
        labels.append(1)
    else:
        labels.append(0)

    if apple[1] < head[1]:
        labels.append(1)
    else:
        labels.append(0)

    y[i] = torch.tensor(labels)
    
trainer.train(X, y, epochs=10)

torch.save(X, "X_tensor")
torch.save(y, "y_tensor")
torch.save(trainer.model, "Warm_start_model")