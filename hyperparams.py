import os.path 
import torch
from game import Game

class Hyperparams:
    def __init__(self) -> None:
        #trainer
        self.gamma: float = 0.99 #TRY HIGH AT SOME POINT
        self.lr: float = 1e-5
        self.batch_size: int = 256
        self.clip: int = 10 #Suggested in Dueling heads paper
        self.tau = 0.001
        self.load_path: str = None

        #replay
        self.beta = 0.7
        self.alpha = 0.4
        
        #main
        self.max_episodes: int = 120000
        self.replay_size: int = 10000
        self.update_rate: int = 1000
        self.test_games: int = self.update_rate//10
        self.model_save_rate: int = 100

        #agent
        self.exploration_rate_start: float = 0.1
        self.exploration_rate_end: float =  0.01
        self.epsilon: float = 1/(self.max_episodes*1.8 )

        #game
        self.size: int = 7
        self.lifespan: int = 100
        self.apple_reward = 1.0
        self.death_reward = -1.0

        self.game: Game = Game(self.size, self.lifespan, self.apple_reward, self.death_reward)
        self.action_space = 4
        
        self.device: torch.DeviceObjType = torch.device("cpu")


    def set_load_path(self, load_path: str) -> None:
        print("Loading a model, this is not a fresh run.")
        if os.path.isfile(load_path):
            self.load_path = load_path
        else:
            print("Couldn't find a file named", load_path)
