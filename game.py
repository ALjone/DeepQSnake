import math
import random as rn
from typing import List
import torch

class Tail:
    def __init__(self, x_pos, y_pos, next):    
        self.x_pos: int = x_pos
        self.y_pos: int = y_pos
        self.next: Tail = next

class Game:
    def __init__(self, size, lifespan):
        """Initializes the game with the correct size and lifespan, and puts a head in the middle, as well as an apple randomly on the map"""
        self.mapsize: int = size
        self.lifespan: int = lifespan
        self.mid: int = (self.mapsize-1)//2
        self.reset()
    
    def _addApple(self):
        """Adds and apple to the map at a random legal position"""
        #TODO Make this more algorithmically correct
        a = rn.randint(0, self.mapsize-1)
        b = rn.randint(0, self.mapsize-1)
        while(self.__contains_apple(a, b)):
            a = rn.randint(0, self.mapsize-1)
            b = rn.randint(0, self.mapsize-1)
        self.apple_x = a
        self.apple_y = b
        
        

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
        self.game_map: torch.Tensor = torch.zeros(3, self.mapsize, self.mapsize)

        #Set head and tail
        head: Tail = Tail(self.mid, self.mid, None)
        self.tail: Tail = head
        self.head: Tail = head
        self.__update()

        #Set variables
        self.moves_since_ate: int = 0
        self.moves: int = 0
        self.score = 0
        self.dead: bool = False
        self.ate_last_turn: bool = False

        #Add apple
        self._addApple()
        self.__update
        self.previousAppleDistance: float = self.distToApple()

    def __can_move(self, x, y):
        return (0 <= x < self.mapsize) and (0 <= y < self.mapsize)
        
    def is_game_over(self):
        """Check if either the snake is out of bounds, hasn't eaten enough, or ate itself."""
        x = self.head.x_pos
        y = self.head.y_pos
        if self.moves_since_ate >= self.lifespan:
            self.dead = True

        tail = self.tail
        while (tail.next != None):
            if (tail.x_pos == x and tail.y_pos == y):
                self.dead = True
            tail = tail.next
        
        if not self.__can_move(x, y):
            self.dead = True

        return self.dead

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
        return self.game_map

    def __try_to_eat(self, x_pos, y_pos):
        if self.__contains_apple(x_pos, y_pos):
            self.ate_last_turn = True
            self.score += 1
            self.moves_since_ate = 0
            
            #Adds a new tail to the end
            #This should work because it just moves it to the next's position
            self.tail = Tail(self.tail.x_pos, self.tail.y_pos, self.tail)

            #Remove apple
            self.__remove_apple(x_pos, y_pos)
            return True
        else:
            self.moves_since_ate += 1
            return False
        

    def __contains_apple(self, x, y):
        return self.game_map[2, x, y] == 1

    def __remove_apple(self, x, y):
        self.game_map[2, x, y] == 0

    def __move_snake(self, x_dir, y_dir):
        tail = self.tail
        while (tail.next != None):
            tail.x_pos = tail.next.x_pos
            tail.y_pos = tail.next.y_pos
            tail = tail.next
        tail.x_pos = tail.x_pos+x_dir
        tail.y_pos = tail.y_pos+y_dir

    def __move(self, x_dir, y_dir):
        """Moves the snake in the given direction, eating anything in its path. Includes a death check."""

        x = self.head.x_pos + x_dir
        y = self.head.y_pos + y_dir

        if not self.__can_move(x, y):
            self.dead = True
            return

        self.previousAppleDistance = self.distToApple()
        self.moves += 1
        
        ate = self.__try_to_eat(x, y)
        self.__move_snake(x_dir, y_dir)
        self.is_game_over()
        if ate: self._addApple()
        self.__update()
        
    def __update(self):
        """Resets and updates the position of all the objectives on the map"""
        #Reset map
        self.game_map: torch.Tensor = torch.zeros(3, self.mapsize, self.mapsize)

        #Add head
        self.game_map[0, self.head.x_pos, self.head.y_pos] = 1

        #Add tail
        tail = self.tail
        while (tail.next != None):
            self.game_map[1, tail.x_pos, tail.y_pos] = 1
            tail = tail.next

        #Add apple
        if self.apple_x is not None and self.apple_y is not None:
            self.game_map[2, self.apple_x, self.apple_y] = 1