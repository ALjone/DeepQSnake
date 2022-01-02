from game_map import Game
import graphics

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

    def updateWin(self, game_map: Game, reward):
        self.moves.setText("Moves: " + str(game_map.moves))
        self.hunger.setText("Hunger: " + str(game_map.moves_since_ate))
        self.reward.setText("Reward: " + str(round(reward, 1)))
        for i in range(self.size):
                for j in range(self.size):
                    if game_map.game_map[i][j] == 0:# and self.squares[i][j].config["fill"] != "black":
                        self.squares[i][j].setFill("black")
                    if game_map.game_map[i][j] == 1:# and self.squares[i][j].config["fill"] != "white":
                        self.squares[i][j].setFill("white")
                    if game_map.game_map[i][j] == 2:# and self.squares[i][j].config["fill"] != "grey":
                        self.squares[i][j].setFill("grey")
                    if game_map.game_map[i][j] == 3:# and self.squares[i][j].config["fill"] != "red":
                        self.squares[i][j].setFill("red")    
        self.win.update()