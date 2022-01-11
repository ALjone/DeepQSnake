import os.path 
import torch
from game import Game

class Hyperparams:
    def __init__(self) -> None:
        #trainer
        self.gamma: float = 0.9 #TRY HIGH AT SOME POINT
        self.lr: float = 1e-3
        self.prime_update_rate: int = 15
        self.batch_size: int = 128
        self.load_path: str = None
        self.frame_stacks = 2 #Max 2 as of now
        
        #main
        self.max_episodes: int = 100000
        self.replay_size: int = 10000
        self.update_rate: int = 1000
        self.test_games: int = 100  

        #agent
        self.exploration_rate_start: float = 0.95
        self.exploration_rate_end: float =  0.02
        self.first_epsilon: float = 1/(self.max_episodes*0.5)
        self.second_epsilon: float = 1/(self.max_episodes*0.9)
        self.epsilon_cutoff = 1 #0.3 for the future
        
        self.device: torch.DeviceObjType = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #game
        self.size: int = 10
        self.lifespan: int = 100
        self.game: Game = Game(self.size, self.lifespan, self.device)
        self.action_space = 4


    def set_load_path(self, load_path: str) -> None:
        print("Loading a model, this is not a fresh run.")
        if os.path.isfile(load_path):
            self.load_path = load_path
        else:
            print("Couldn't find a file named", load_path)
