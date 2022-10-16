import random as rn
from typing import Tuple
import numpy as np
from game_map import Game_map
from collections import deque

reverse = {0: 1, 1: 0, 2: 3, 3: 2}

class Game:
    def __init__(self, size, lifespan, apple_reward, death_reward):
        """Initializes the game with the correct size and lifespan, and puts a head in the middle, as well as an apple randomly on the map"""
        self.mapsize: int = size
        self.size: int = size**2
        self.lifespan: int = lifespan
        self.__game_map: Game_map = Game_map(self.mapsize)
        self.apple_reward = apple_reward
        self.death_reward = death_reward
        self.death_penalty = 0
        self.reset()    
    
    def _addApple(self):
        """Adds and apple to the map at a random legal position"""
        #NOTE: Might be sort of a little bit bugged here, given that 
        true_idx = np.argwhere(self.__game_map.possible_apple_pos_map == 0)
        random_idx = np.random.randint(len(true_idx), size=1)
        random_index = true_idx[random_idx][0]
        self.apple_x = random_index[0]
        self.apple_y = random_index[1]
        
    def __get_reward(self) -> float:
        if self.ate_last_turn:
            return self.apple_reward
        if self.dead:
            return self.death_penalty
        else:
            return 0.0

    def reset(self, test = False) -> np.ndarray:
        """Resets the game, making it a fresh instance with no memory of previous happenings"""
        self.apple_x: int = None
        self.apple_y: int = None

        #3 LAYERS, FIRST IS HEAD, SECOND IS TAIL, THIRD IS APPLES
        self.__game_map.reset()

        #Set head and tail
        self.snake = np.zeros((self.mapsize**2, 2), dtype = np.int8)
        self.snake[0, :] = [self.mapsize//2, self.mapsize//2]    
        self.snake_length = 1

        #Set variables
        self.moves_since_ate: int = 0
        self.moves: int = 0
        self.score = 0
        self.dead: bool = False
        self.final_state: bool = False
        self.ate_last_turn: bool = False

        #Add apple
        self.__game_map.update(self.snake[:self.snake_length], [(self.apple_x, self.apple_y)])
        self._addApple()
        self.__game_map.update(self.snake[:self.snake_length], [(self.apple_x, self.apple_y)])

        return self._get_map()

    def __can_move(self, x, y):
        return (0 <= x < self.mapsize) and (0 <= y < self.mapsize and not self.__game_map.has_tail(x, y))

    def valid_moves(self):
        #return np.array([1, 1, 1, 1])
        x = self.snake[0, 0]
        y = self.snake[0, 1]
        valid = np.zeros(4)
        for i, move in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            is_valid = int(self.__can_move(x+move[0], y+move[1]))
            valid[i] = is_valid
        return np.array(valid)
        
    def __is_game_over(self):
        """Check if either the snake is out of bounds, hasn't eaten enough, or ate itself."""
        x = self.snake[0, 0]
        y = self.snake[0, 1]
        if self.moves_since_ate >= self.lifespan:
            self.final_state = True
            self.death_penalty = 0

        if not self.__can_move(x, y):
            self.dead = True
            self.final_state = True
            self.death_penalty = -1.0
        
        if self.snake_length == self.size:
            self.dead = False
            self.final_state = True
            self.death_penalty = 10

        return self.dead + self.final_state

    def do_action(self, action) -> Tuple[np.ndarray, float, bool]:
        """Completes the given action and returns the new map"""
        self.ate_last_turn = False
        if(action == 0):
            #West?
            self.__move(-1, 0)
        if(action == 1):
            #East?
            self.__move(1, 0)
        if(action == 2):
            #South?
            self.__move(0, -1)
        if(action == 3):
            #North?
            self.__move(0, 1)

        is_game_over = self.__is_game_over()
        return self._get_map(), self.__get_reward(), is_game_over

    def __contains_apple(self, x_pos, y_pos):
        return (x_pos == self.apple_x and y_pos == self.apple_y)
    
    def __remove_apple(self):
        self.apple_x = None
        self.apple_y = None

    def __try_to_eat(self, x_pos, y_pos):
        if not self.__contains_apple(x_pos, y_pos):
            self.moves_since_ate += 1
            return False
        self.ate_last_turn = True
        self.score += 1
        self.moves_since_ate = 0
        
        #Adds a new tail to the end
        #This should work because it just moves it to the next's position
        self.snake[self.snake_length] = [self.snake[self.snake_length-1, 0], self.snake[self.snake_length-1, 1]]

        #Remove apple
        self.__remove_apple()
        self.snake_length += 1
        return True
            

    def __move_snake(self, x_dir, y_dir):
        for i in range(self.snake_length-1, 0, -1):
            self.snake[i, 0] = self.snake[i-1, 0]
            self.snake[i, 1] = self.snake[i-1, 1]
            
        self.snake[0, 0] = self.snake[0, 0]+x_dir
        self.snake[0, 1] = self.snake[0, 1]+y_dir

    def __move(self, x_dir, y_dir):
        """Moves the snake in the given direction, eating anything in its path. Includes a death check."""

        x = self.snake[0, 0] + x_dir
        y = self.snake[0, 1] + y_dir

        if not self.__can_move(x, y):
            self.dead = True
            self.final_state = True
            return

        self.moves += 1
        
        ate = self.__try_to_eat(x, y)
        self.__move_snake(x_dir, y_dir)
        self.__is_game_over()
        self.__game_map.update(self.snake[:self.snake_length], [(self.apple_x, self.apple_y)])
        if ate:
            if self.snake_length < self.size: 
                self._addApple()
            self.__game_map.update(self.snake[:self.snake_length], [(self.apple_x, self.apple_y)])


    def _get_map(self):
        return self.__game_map.get_map()