from game import Game
import graphics
import numpy as np
from tkinter import *

class Graphics:
    def __init__(self, size):
        self.size = size
        self.win = graphics.GraphWin(height = 250, width = 400) # create a window
        self.win.setCoords(0, 0, size+4, size) # set the coordinates of the window; bottom left is (0, 0) and top right is (10, 10)
        self.squares = []
        self.hunger = graphics.Text(graphics.Point(size+2, size-7), 0)
        self.moves = graphics.Text(graphics.Point(size+2, size-6), 0)
        self.reward = graphics.Text(graphics.Point(size+2, size-5), 0)
        self.dead = graphics.Text(graphics.Point(size+2, size-4), 0)
        self.moves.draw(self.win)
        self.hunger.draw(self.win)
        self.reward.draw(self.win)
        self.dead.draw(self.win)
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
        self.dead.setText("Dead: True" if game.dead else "Dead: False")
        game_map = game._get_map()
        for i in range(self.size):
                for j in range(self.size):
                    if self.squares[i][j].config["fill"] != "black" and np.sum(game_map[:, i, j] == 0) :
                        self.squares[i][j].setFill("black")
                    if game_map[0, i, j] == 1 and self.squares[i][j].config["fill"] != "white":
                        self.squares[i][j].setFill("white")
                    if self.squares[i][j].config["fill"] != "grey" and game_map[1, i, j] == 1:
                        self.squares[i][j].setFill("grey")
                    if game_map[2, i, j] == 1 and self.squares[i][j].config["fill"] != "red":
                        self.squares[i][j].setFill("red")    
        self.win.update()







class Graphics:
    def __init__(self, size) -> None:
        self.root = Tk()
        self.size = size
        block_width = int(800/size)
        block_height = int(800/size)
        width = block_width*size
        height = block_height*size
        self.canvas = Canvas(self.root, width=width, height=height)

        self.rects = {}
        for x in range(size):
            for y in range(size):
                fill = "black"
                self.rects[(x, y)] = self.canvas.create_rectangle(y*block_width, x*block_height, (y+1)*block_width, (x+1)*block_height, fill=fill)

        self.canvas.focus_set()
        self.canvas.pack()

        self.root.focus_set()
        #self.root.mainloop()


    def updateWin(self, game: Game, reward, expected):
        self.root.title(f"Total reward: {reward}, Moves: {game.moves}, Hunger: {game.moves_since_ate}, Expected: {expected}")
        game_map = np.ceil(game._get_map())
        for x in range(self.size):
                for y in range(self.size):
                    color = self.canvas.itemcget(self.rects[(y, x)], "fill")

                    if color != "black" and np.sum(game_map[:, x, y]) < 1:
                        self.canvas.itemconfig(self.rects[(y, x)], fill="black")
                    if game_map[0, x, y] == 1 and color != "white":
                        self.canvas.itemconfig(self.rects[(y, x)], fill="white")

                    if color != "grey" and game_map[1, x, y] == 1:
                        self.canvas.itemconfig(self.rects[(y, x)], fill="grey")

                    if game_map[2, x, y] == 1 and color != "red":
                        apple = (x, y)
                        self.canvas.itemconfig(self.rects[(y, x)], fill="red")


        self.root.update()

        