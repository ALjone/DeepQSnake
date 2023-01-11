import numpy as np
np.import_array()
cimport numpy as np


cdef class SnakeGame:
    # The game map
    
    # The position of the snake's head
    cdef (int, int) head_position
    # The positions of the snake's body
    cdef list body_positions
    # The position of the apple
    cdef (int, int) apple_position

    cdef int moves

    cdef int lifespan 
    
    cdef int[:, :, :] game_map

    def __init__(self, rows: int, cols: int, lifespan: int):
        self.game_map = np.zeros((3, rows, cols), dtype=int)
        self.lifespan = lifespan
        self.reset()

    cdef place_apple(self):
        # Create a boolean mask indicating where the snake is on the game map
        mask = np.zeros((self.game_map.shape[1], self.game_map.shape[2]), dtype=bool)
        mask[self.head_position] = True
        for body_part in self.body_positions:
            mask[body_part] = True

        # Randomly select a valid position for the apple by sampling from the non-masked
        # positions of the game map.
        valid_positions = np.nonzero(~mask)
        index = np.random.randint(0, len(valid_positions[0]))
        #self.game_map[2, self.apple_position[0], self.apple_position[1]] = 0
        self.apple_position = (valid_positions[0][index], valid_positions[1][index])
        #self.game_map[2, self.apple_position[0], self.apple_position[1]] = 1

    cpdef tuple step(self, action: int):
        reward = 0.0
        cdef (int, int) new_head_position
        # Calculate new position of snake's head based on action
        if action == 0:  # Move up
            new_head_position = (self.head_position[0] - 1, self.head_position[1])
        elif action == 1:  # Move right
            new_head_position = (self.head_position[0], self.head_position[1] + 1)
        elif action == 2:  # Move down
            new_head_position = (self.head_position[0] + 1, self.head_position[1])
        elif action == 3:  # Move left
            new_head_position = (self.head_position[0], self.head_position[1] - 1)

        # Check if new position is valid (i.e. not outside the bounds of the game map)
        if new_head_position[0] < 0 or new_head_position[0] >= self.game_map.shape[1]:
            return np.asarray(self.game_map), -1.0, True  # Invalid position, return reward -1 and done = True
        if new_head_position[1] < 0 or new_head_position[1] >= self.game_map.shape[2]:
            return np.asarray(self.game_map), -1.0, True  # Invalid position, return reward -1 and done = True

        # Check if snake has eaten an apple
        if new_head_position == self.apple_position:
            # Snake has eaten an apple, update body positions and place a new apple
            self.body_positions.append(self.head_position)
            self.place_apple()
            self.moves = 0
            reward = 1.0
        else:
            self.body_positions.append(self.head_position)
            self.body_positions = self.body_positions[1:]


        self.head_position = new_head_position
        self.update_state()

        # Check if snake has collided with itself
        if new_head_position in self.body_positions:
            return np.asarray(self.game_map), -1, True  # Collision, return reward -1 and done = True

        # Check if game is done (i.e. snake has eaten enough apples)
        if len(self.body_positions) >= (self.game_map.size**2)-1:
            return np.asarray(self.game_map), 10, True  # Game won, return reward 1 and done = True

        self.moves += 1

        if self.moves == self.lifespan:
            return np.asarray(self.game_map), reward, True

        return np.asarray(self.game_map), reward, False  # Game not done, return reward 0 and done = False


    cpdef np.ndarray reset(self):
        self.game_map[:, :, :] = 0
        self.head_position = (self.game_map.shape[1]//2, self.game_map.shape[1]//2)
        self.body_positions = []
        self.moves = 0
        self.place_apple()
        self.update_state()

        return np.asarray(self.game_map)

    cpdef np.ndarray valid_moves(self):
        valid_moves = []

        # Check if moving up is a valid move (i.e. not moving into a wall or the snake's body)
        if self.head_position[0] > 0 and (self.head_position[0] - 1, self.head_position[1]) not in self.body_positions:
            valid_moves.append(1)
        else:
            valid_moves.append(0)

        # Check if moving right is a valid move
        if self.head_position[1] < self.game_map.shape[2] - 1 and (self.head_position[0], self.head_position[1] + 1) not in self.body_positions:
            valid_moves.append(1)
        else:
            valid_moves.append(0)

        # Check if moving down is a valid move
        if self.head_position[0] < self.game_map.shape[1] - 1 and (self.head_position[0] + 1, self.head_position[1]) not in self.body_positions:
            valid_moves.append(1)
        else:
            valid_moves.append(0)

        # Check if moving left is a valid move
        if self.head_position[1] > 0 and (self.head_position[0], self.head_position[1] - 1) not in self.body_positions:
            valid_moves.append(1)
        else:
            valid_moves.append(0)

        return np.asarray(valid_moves)


    cdef void update_state(self):
        # Create a 3-layer game map
        self.game_map[:, :, :] = 0
        # Set the snake's head and tail in the first two layers of the game map
        self.game_map[0, self.head_position[0], self.head_position[1]] = 1  # Snake's head is in layer 0
        cdef (int, int) body_pos 
        for body_pos in self.body_positions:
            self.game_map[1, body_pos[0], body_pos[1]] = 1  # Snake's tail is in layer 1

        # Set the apple in the third layer of the game map
        self.game_map[2, self.apple_position[0], self.apple_position[1]] = 1  # Apple is in layer 2

    cpdef int[:, :, :] get_state(self):
        return self.game_map


    def __repr__(self):
        # Get the maximum x and y coordinates of the snake's body

        # Create a list of strings representing each row of the game board
        rows = []
        for x in range(self.game_map.shape[1]):
            row = []
            for y in range(self.game_map.shape[2]):
                space = True
                if self.game_map[0, x, y] == 1:#(x, y) == self.head_position:
                    row.append("h")
                    space = False
                if self.game_map[1, x, y] == 1:#(x, y) in self.body_positions:
                    row.append("b")
                    space = False
                if self.game_map[2, x, y] == 1:#(x, y) == self.apple_position:
                    row.append("a")
                    space = False
                if space:
                    row.append(" ")
            rows.append("|" + " ".join(row) + "|")

        # Add top and bottom borders to the game board
        border = "+" + "-" * (self.game_map.shape[2]*2-1) + "+"
        rows.insert(0, border)
        rows.append(border)

        # Return the game board as a string
        return "\n".join(rows)