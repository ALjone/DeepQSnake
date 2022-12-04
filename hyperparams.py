import os.path 
import torch
from game import Game

class Hyperparams:
    def __init__(self) -> None:
        
        self.device: torch.DeviceObjType = torch.device("cpu")
        
        #trainer
        self.gamma: float = 0.99 #TRY HIGH AT SOME POINT
        self.lr: float = 0.0000625 #0.0002 Recommended from https://arxiv.org/pdf/1710.02298.pdf
        self.reg: float = 1e-4
        self.batch_size: int = 64
        self.train_times = 5
        self.clip: int = 10 # 10 was suggested in Dueling heads paper
        self.tau = 0.001#0.0005
        self.wait_frames = 10000 #Wait at least this many frames before starting, according to https://arxiv.org/pdf/1710.02298.pdf
        self.noise = 0.5 #Noise for the noisy nets exploration
        self.load_path: str = None


        #replay
        self.beta = 0.7
        self.alpha = 0.5
        
        #main
        self.max_episodes: int = 5000000
        self.replay_size: int = 100000
        self.update_rate: int = 500
        self.test_games: int = self.update_rate//10
        self.model_save_rate: int = self.update_rate//10
        #Important, as we need to train _at least_ once per move we make, otherwise priorization is dumb
        self.train_rate: int = 100

        #game
        self.size: int = 10
        self.lifespan: int = 300
        self.apple_reward = 1.0
        self.death_reward = -1.0
        
        self.game: Game = Game(self.size, self.lifespan, self.apple_reward, self.death_reward, self.device)

        self.action_space = 4
        #Note that this only affects the moves done by the network. Random moves will still be action masked
        self.action_masking = False
        


    def set_load_path(self, load_path: str) -> None:
        print("Loading a model, this is not a fresh run.")
        if os.path.isfile(load_path):
            self.load_path = load_path
        else:
            print("Couldn't find a file named", load_path)
