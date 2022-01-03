from game_map import Game
import graphics
import torch

class Graphics:
    def __init__(self, size):
        self.size = size
        self.win = graphics.GraphWin(height = 250, width = 400) # create a window
        self.win.setCoords(0, 0, size+4, size) # set the coordinates of the window; bottom left is (0, 0) and top right is (10, 10)
        self.squares = []
        self.hunger = graphics.Text(graphics.Point(size+2, size-7), 0)
        self.moves = graphics.Text(graphics.Point(size+2, size-6), 0)
        self.reward = graphics.Text(graphics.Point(size+2, size-5), 0)
        self.moves.draw(self.win)
        self.hunger.draw(self.win)
        self.reward.draw(self.win)
        for i in range(size):
            self.squares.append([])
            for j in range(size):
                mySquare = graphics.Rectangle(graphics.Point(i, j),
                                          graphics.Point(i+1, j+1))
                mySquare.draw(self.win)
                self.squares[i].append(mySquare)

    def updateWin(self, game: Game, reward):
        self.moves.setText("Moves: " + str(game.moves))
        self.hunger.setText("Hunger: " + str(game.moves_since_ate))
        self.reward.setText("Reward: " + str(round(reward, 1)))
        game_map = game.game_map
        for i in range(self.size):
                for j in range(self.size):
                    if torch.sum(game_map[:, i, j] == 0) and self.squares[i][j].config["fill"] != "black":
                        self.squares[i][j].setFill("black")
                    if game_map[0, i, j] == 1 and self.squares[i][j].config["fill"] != "white":
                        self.squares[i][j].setFill("white")
                    if game_map[1, i, j] == 1 and self.squares[i][j].config["fill"] != "grey":
                        self.squares[i][j].setFill("grey")
                    if game_map[2, i, j] == 1 and self.squares[i][j].config["fill"] != "red":
                        self.squares[i][j].setFill("red")    
        self.win.update()