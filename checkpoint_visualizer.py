from game import Game
from agent import DQAgent
from hyperparams import Hyperparams
import time
import torch
import numpy as np
reverse = {0: 1, 1: 0, 2: 3, 3: 2}


class Visualizer:
    def __init__(self, path: str = None, hyperparams: Hyperparams = None) -> None:
        #TODO add corresponding hyperparams
        self.game_loaded = False
        

    def load_game(self, path: str = None, hyperparams: Hyperparams = None):
        #TODO Fix path stuff
        if path is None:
            path = str(input("What is the name of the file in the checkpoints folder? "))
            if path == "":
                path = "last_checkpoint_in_case_of_crash"
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
            self.agent.trainer.reload_model()

    def __run(self):
        total_reward = 0
        state = self.game.reset()
        prev_move = -1
        self.agent.trainer.model.eval()
        ls = [[], [], [], []]
        counter = 0
        add = ""
        while(True):
            #Should probably time how long everything takes rather than using a flat 0.1s
            time.sleep(0.05  )
            _, V, A = self.agent.trainer.model(state, return_separate = True)
            expected = [x.item() for x in A.to("cpu")[0]]
            for i, e in enumerate(expected):
                ls[i].append(e)
            #Left, Right, Down, Up
            expected = "State: {4} Left: {0}, Right: {1}, Down: {2}, Up: {3}".format(*[round(e, 3) for e in expected], round(V.to("cpu").item(), 3))
            #print(expected)
            self.graphics.updateWin(self.game, total_reward, expected)
            #print(self.game.valid_moves())
            move = self.agent.get_move(state, self.game.valid_moves())

            
            #print(move)
            state, reward, done = self.game.do_action(move)
            total_reward += reward
            if prev_move == reverse[move]:
                done = True
            prev_move = move
            counter += 1
            if(done):
                self.graphics.updateWin(self.game, total_reward, expected)
                self.game.reset()
                print("Stds:", [torch.round(torch.std(torch.tensor(a)), decimals = 3).item() for a in ls], "Moves:", counter)
                print("Expected before final state", expected, "Move:", move)
                if prev_move == reverse[move]:
                    print("Aborted due to repeated move ")
                print()
                break

    def get_expected(self, state):
        expected = [x.item() for x in self.agent.trainer.model(state)[0]]
        #Left, Right, Down, Up
        expected = "Left: {0}, Right: {1}, Down: {2}, Up: {3}".format(*[round(e, 3) for e in expected])
        return expected