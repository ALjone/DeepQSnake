import numpy as np

class Game_map:
    def __init__(self, mapsize) -> None:
        self.mapsize = mapsize

        self.reset()

    def has_tail(self, x, y):
        return 0 < self.game_map[1, x, y] < 1
    
    def update(self, snake, apples):
        """Resets and updates the position of all the objectives on the map"""
        #Reset map
        self.reset()

        #Add head

        self.game_map[0, snake[0, 0], snake[0, 1]] = 1 
        self.possible_apple_pos_map[snake[0, 0], snake[0, 1]] = 1

        #Add tail
        for i, tail in enumerate(reversed(snake[1:])):
            self.game_map[1, tail[0], tail[1]] = (i+1)/(len(snake))
            self.possible_apple_pos_map[tail[0], tail[1]] = 1

        #Add apple

        #Fix this
        for apple_x, apple_y in apples:
            if apple_x is None or apple_y is None:
                continue
            self.game_map[2, apple_x, apple_y] = 1

    def get_map(self):
        return self.game_map

    def reset(self):
        self.game_map = np.zeros((3, self.mapsize, self.mapsize))
        self.possible_apple_pos_map = np.zeros((self.mapsize, self.mapsize), dtype = np.int)
