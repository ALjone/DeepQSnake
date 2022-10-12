import os.path 
import torch
from game import Game

class Hyperparams:
    def __init__(self) -> None:
        #trainer
        self.gamma: float = 0.99 #TRY HIGH AT SOME POINT
        self.lr: float = 1e-5
        self.prime_update_rate: int = 50
        self.batch_size: int = 256
        self.load_path: str = None
        self.frame_stacks = 1 #Max 2 as of now
        
        #main
        self.max_episodes: int = 5000
        self.replay_size: int = 5000
        self.update_rate: int = 1000
        self.test_games: int = self.update_rate//10

        #agent
        self.exploration_rate_start: float = 0.5
        self.exploration_rate_end: float =  0.1
        self.epsilon: float = 1/(self.max_episodes*1.8 )

        #game
        self.size: int = 7
        self.lifespan: int = 200
        self.apple_reward = 1.0
        self.death_reward = -20.0 #This cannot be the same as no reward due to bad replaymemorystuff
        
        checkpoint_timestamps = [20, 40, 60, 80, 100, 120, 140, 160, 180]
        checkpoint_length = 400
        checkpoint_probability = 0.3

        self.game: Game = Game(self.size, self.lifespan, self.apple_reward, self.death_reward, checkpoint_timestamps, checkpoint_length, checkpoint_probability)
        self.action_space = 4
        
        self.device: torch.DeviceObjType = torch.device("cpu")


    def set_load_path(self, load_path: str) -> None:
        print("Loading a model, this is not a fresh run.")
        if os.path.isfile(load_path):
            self.load_path = load_path
        else:
            print("Couldn't find a file named", load_path)
