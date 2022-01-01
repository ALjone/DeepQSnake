import math
import random as rn
from typing import List

class Tail:
    def __init__(self, pos, nxt):    
        self.pos: List[int] = pos
        self.next: Tail = nxt

class Game:
    def __init__(self, size, lifespan, range):
        self.mapsize: int = size
        self.lifespan: int = lifespan
        self.range: int = range
        self.mid: int = int((self.mapsize-1)/2)
        self.illegal_move = None

        
        self.reset()
        


    def _makeMap(self):
        for i in range(self.mapsize):
            self.game_map.append([])
            for _ in range(self.mapsize):
                self.game_map[i].append(0)
    
    def _addAppel(self):
        a = rn.randint(self.mid-self.range, self.mid+1+self.range)
        b = rn.randint(self.mid-self.range, self.mid+1+self.range)
        while(self.game_map[a][b] != 0):
            a = rn.randint(self.mid-self.range, self.mid+1+self.range)
            b = rn.randint(self.mid-self.range, self.mid+1+self.range)
            #rn.randint(0, self.mapsize-1)
        self.game_map[a][b] = 3
        self.apple_pos = (a, b)

    def distToApple(self):
        x = abs(self.apple_pos[0]-self.head.pos[0]) ** 2
        y = abs(self.apple_pos[1]-self.head.pos[1]) ** 2
        return math.sqrt(x + y)

    def reset(self, warm_start = 1):
        #TODO Fix
        self.apple_pos: List[int] = None
        self.game_map: List[List[int]] = []
        self._makeMap()
        self.game_map[int((self.mapsize-1)/2)][int((self.mapsize-1)/2)] = 1
        self.moves_since_ate: int = 0
        self.moves: int = 0
        self.score = 0
        self.dead: bool = False
        self.ate_last_turn = False

        head: Tail = Tail((int(self.mapsize/2), int(self.mapsize/2)) , None)
        self.tail: Tail = head
        self.head: Tail = head

        self.__update()
        #TODO reset this
        if warm_start:
            a = rn.randint(1, 4)
            if a == 1:
                self.game_map[4+warm_start][4] = 3
                self.apple_pos = (4+warm_start, 4)
            elif a == 1:
                self.game_map[4+warm_start][4] = 3
                self.apple_pos = (4+warm_start, 4)
            elif a == 1:
                self.game_map[4][4+warm_start] = 3
                self.apple_pos = (4, 4+warm_start)
            else:
                self.game_map[4][4+warm_start] = 3
                self.apple_pos = (4, 4+warm_start)
        else:
            self._addAppel()
        self.previousAppleDistance: float = self.distToApple()

    def __can_move(self, x, y):
        if (x < 0 or x >= self.mapsize or y < 0 or y >= self.mapsize):
            return False
        return True
        
    def is_dead(self):
        """Check if either the snake is out of bounds, hasn't eaten enough, or ate itself"""
        x = self.head.pos[0]
        y = self.head.pos[1]
        if self.moves_since_ate == self.lifespan:
            self.dead = True

        tail = self.tail
        while (tail.next != None):
            if (tail.pos == [x, y]):
                self.dead = True
            tail = tail.next

        return self.dead

    def do_action(self, action):
        self.ate_last_turn = False
        if(action == 0):
            self.illegal_move = 1
            self.__move([-1, 0])
        if(action == 1):
            self.illegal_move = 0
            self.__move([1, 0])
        if(action == 2):
            self.illegal_move = 3
            self.__move([0, -1])
        if(action == 3):
            self.illegal_move = 2
            self.__move([0, 1])
        return self.game_map

    def __eat(self, new_pos):
        self.ate_last_turn = True
        self.score += 1
        self.moves_since_ate = 0
        
        #This should work because it just moves it to the next's position
        self.tail = Tail(self.tail.pos, self.tail)

        #Remove apple
        self.game_map[new_pos[0]][new_pos[1]] = 0
        

    def __move(self, direction):
        ate = False
        self.previousAppleDistance = self.distToApple()

        self.moves += 1
        x = self.head.pos[0]+direction[0]
        y = self.head.pos[1]+direction[1]
        if not self.__can_move(x, y):
            #Should be incorporated to _is_dead
            self.dead = True
            return

        if(self.game_map[x][y] == 3):
            self.__eat([x, y])
            ate = True
        else:
            self.moves_since_ate+=1

        tail = self.tail
        while (tail.next != None):
            tail.pos = tail.next.pos
            tail = tail.next
        tail.pos = [tail.pos[0]+direction[0], tail.pos[1]+direction[1]]
        self.__update()
        self.is_dead()
        if ate: self._addAppel()

        if self.dead: return
        
    def __update(self):
        for i in range(self.mapsize):
            for j in range(self.mapsize):
                if(self.game_map[i][j] == 1 or 
                   self.game_map[i][j] == 2): self.game_map[i][j] = 0
        
        tail = self.tail
        while (tail.next != None):
            self.game_map[tail.pos[0]][tail.pos[1]] = 2
            tail = tail.next
        self.game_map[tail.pos[0]][tail.pos[1]] = 1