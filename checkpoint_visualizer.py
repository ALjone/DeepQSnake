from game import Game
from agent import DQAgent
from hyperparams import Hyperparams
import time



class Visualizer:
    def __init__(self, path: str = None, hyperparams: Hyperparams = None) -> None:
        #TODO add corresponding hyperparams
        self.game_loaded = False
        

    def load_game(self, path: str = None, hyperparams: Hyperparams = None):
        #TODO Fix path stuff
        if path is None:
            path = str(input("What is the name of the file in the checkpoints folder? "))
        
        if hyperparams is None:
            self.hyperparams = Hyperparams()
        else:
            self.hyperparams = hyperparams

        self.hyperparams.set_load_path("checkpoints/" + path)
        
        self.game: Game = self.hyperparams.game
                
        self.agent: DQAgent = DQAgent(self.hyperparams)

        self.game_loaded = True
        
    def visualize(self):
        if not self.game_loaded: 
            #Should be a better exception probably.
            raise BaseException("A game has not been loaded for the visualizer. Please call load_game first.")
        from graphics_module import Graphics
        self.graphics: Graphics = Graphics(self.hyperparams.game.mapsize)

        self.agent.testing = True
        while(True):
            self.__run()
            try:
                self.graphics.win.getKey()
            except:
                return

    def __run(self):
        reward = 0
        while(True):
            #Should probably time how long everything takes rather than using a flat 0.1s
            #time.sleep(0.01)
            self.graphics.updateWin(self.game, reward)
            move = self.agent.get_move(self.game)
            self.game.do_action(move)
            reward += self.game.get_reward()


            if(self.game.final_state):
                self.graphics.updateWin(self.game, reward)
                self.game.reset()
                break