#TODO Make a visualizer for the replay memory in order to sanity check the rewards

from ReplayMemory import ReplayMemory
from game import Game
import graphics
import torch

class ReplayGraphics:
    def __init__(self, size, bank: ReplayMemory):
        self.size = size
        self.win = graphics.GraphWin(height = 250, width = 400) # create a window
        self.win.setCoords(0, 0, size+4, size) # set the coordinates of the window; bottom left is (0, 0) and top right is (10, 10)
        self.squares = []
        self.reward = graphics.Text(graphics.Point(size+2, size-5), 0)
        self.dead = graphics.Text(graphics.Point(size+2, size-4), 0)
        self.action = graphics.Text(graphics.Point(size+2, size-6), 0)
        self.reward.draw(self.win)
        self.dead.draw(self.win)
        self.action.draw(self.win)
        for i in range(size):
            self.squares.append([])
            for j in range(size):
                mySquare = graphics.Rectangle(graphics.Point(i, j),
                                          graphics.Point(i+1, j+1))
                mySquare.draw(self.win)
                self.squares[i].append(mySquare)

        for memory in bank.sample(len(bank)):
            self.updateWin(memory.state, memory.reward, memory.action, True if memory.next_state is None else False)
            self.win.getKey()



    def updateWin(self, game_map, reward, action, dead):
        self.reward.setText("Reward: " + str(round(reward.item(), 1)))
        self.dead.setText("Final state: True" if dead else "Final state: False")
        if action == 0:
            self.action.setText("West")
        if action == 1:
            self.action.setText("East")
        if action == 2:
            self.action.setText("South")
        if action == 3:
            self.action.setText("North")
        for i in range(self.size):
                for j in range(self.size):
                    if torch.sum(game_map[:, i, j] == 0):# and self.squares[i][j].config["fill"] != "black":
                        self.squares[i][j].setFill("black")
                    if game_map[0, i, j] == 1:# and self.squares[i][j].config["fill"] != "white":
                        self.squares[i][j].setFill("white")
                    if game_map[1, i, j] == 1:# and self.squares[i][j].config["fill"] != "grey":
                        self.squares[i][j].setFill("grey")
                    if game_map[2, i, j] == 1:# and self.squares[i][j].config["fill"] != "red":
                        self.squares[i][j].setFill("red")    
        self.win.update()