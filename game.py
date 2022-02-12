import math
import random as rn
from typing import List
import torch
from game_map import Game_map

class Tail:
    def __init__(self, x_pos, y_pos, next):    
        self.x_pos: int = x_pos
        self.y_pos: int = y_pos
        self.next: Tail = next

class Game:
    def __init__(self, size, lifespan, apple_reward, death_reward):
        """Initializes the game with the correct size and lifespan, and puts a head in the middle, as well as an apple randomly on the map"""
        self.mapsize: int = size
        self.lifespan: int = lifespan
        self.__game_map: Game_map = Game_map(self.mapsize)
        self.apple_reward = apple_reward
        self.death_reward = death_reward
        self.reset()    
    
    def _addApple(self):
        """Adds and apple to the map at a random legal position"""
        #TODO Make this more algorithmically correct
        a = rn.randint(0, self.mapsize-1)
        b = rn.randint(0, self.mapsize-1)
        while(not self.__can_place_apple(a, b)):
            a = rn.randint(0, self.mapsize-1)
            b = rn.randint(0, self.mapsize-1)
        self.apple_x = a
        self.apple_y = b
        
    def get_reward(self) -> float:
        if self.ate_last_turn:
            return self.apple_reward
        if self.dead:
            return self.death_reward
        else:
            return 0.0

    def distToApple(self):
        """Returns the distance from the head to the apple"""
        x = abs(self.apple_x-self.head.x_pos) ** 2
        y = abs(self.apple_y-self.head.y_pos) ** 2
        return math.sqrt(x + y)

    def reset(self) -> None:
        """Resets the game, making it a fresh instance with no memory of previous happenings"""
        self.apple_x: int = None
        self.apple_y: int = None

        #3 LAYERS, FIRST IS HEAD, SECOND IS TAIL, THIRD IS APPLES
        self.__game_map.reset()

        #Set head and tail
        head: Tail = Tail(rn.randint(1, self.mapsize-2), rn.randint(1, self.mapsize-2), None)
        self.tail: Tail = head
        self.head: Tail = head
        self.__game_map.update(self.head, self.tail, self.apple_x, self.apple_y)

        #Set variables
        self.moves_since_ate: int = 0
        self.moves: int = 0
        self.score = 0
        self.dead: bool = False
        self.final_state: bool = False
        self.ate_last_turn: bool = False

        #Add apple
        self._addApple()
        self.__game_map.update(self.head, self.tail, self.apple_x, self.apple_y)
        self.previousAppleDistance: float = self.distToApple()

    def __can_move(self, x, y):
        return (0 <= x < self.mapsize) and (0 <= y < self.mapsize and not self.__game_map.has_tail(x, y))
        
    def is_game_over(self):
        """Check if either the snake is out of bounds, hasn't eaten enough, or ate itself."""
        x = self.head.x_pos
        y = self.head.y_pos
        if self.moves_since_ate >= self.lifespan:
            self.final_state = True

        tail = self.tail
        while (tail.next != None):
            if (tail.x_pos == x and tail.y_pos == y):
                self.dead = True
                self.final_state = True
            tail = tail.next
        
        if not self.__can_move(x, y):
            self.dead = True
            self.final_state = True

        return self.dead + self.final_state

    def do_action(self, action):
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
        return self.__game_map

    def __contains_apple(self, x_pos, y_pos):
        return (x_pos == self.apple_x and y_pos == self.apple_y)
    
    def __can_place_apple(self, x_pos, y_pos):
        return(self.get_map()[0, x_pos, y_pos] == 0 and self.get_map()[1, x_pos, y_pos] == 0)

    def __remove_apple(self):
        self.apple_x = None
        self.apple_y = None

    def __try_to_eat(self, x_pos, y_pos):
        if self.__contains_apple(x_pos, y_pos):
            self.ate_last_turn = True
            self.score += 1
            self.moves_since_ate = 0
            
            #Adds a new tail to the end
            #This should work because it just moves it to the next's position
            self.tail = Tail(self.tail.x_pos, self.tail.y_pos, self.tail)

            #Remove apple
            self.__remove_apple()
            return True
        else:
            self.moves_since_ate += 1
            return False

    def __move_snake(self, x_dir, y_dir):
        tail = self.tail
        while (tail.next != None):
            tail.x_pos = tail.next.x_pos
            tail.y_pos = tail.next.y_pos
            tail = tail.next
        self.head.x_pos = self.head.x_pos+x_dir
        self.head.y_pos = self.head.y_pos+y_dir

    def __move(self, x_dir, y_dir):
        """Moves the snake in the given direction, eating anything in its path. Includes a death check."""

        x = self.head.x_pos + x_dir
        y = self.head.y_pos + y_dir

        if not self.__can_move(x, y):
            self.dead = True
            self.final_state = True
            return

        self.previousAppleDistance = self.distToApple()
        self.moves += 1
        
        ate = self.__try_to_eat(x, y)
        self.__move_snake(x_dir, y_dir)
        self.is_game_over()
        self.__game_map.update(self.head, self.tail, self.apple_x, self.apple_y)
        if ate: 
            self._addApple()
            self.__game_map.update(self.head, self.tail, self.apple_x, self.apple_y)


    def get_map(self):
        return self.__game_map.get_map()