import os.path 
import torch

class Hyperparams:
    def __init__(self) -> None:
        
        self.device: torch.DeviceObjType = torch.device("cuda:0")
        
        #trainer
        self.gamma: float = 0.99 #TRY HIGH AT SOME POINT
        self.lr: float = 0.0002 #0.0000625 Recommended from https://arxiv.org/pdf/1710.02298.pdf
        self.reg: float = 1e-4
        self.batch_size: int = 1024
        self.clip: int = 10 # 10 was suggested in Dueling heads paper
        self.tau = 0.001#0.0005
        self.wait_frames = 1000 #Wait at least this many frames before starting, according to https://arxiv.org/pdf/1710.02298.pdf
        self.load_path: str = None
        self.frame_stack = 4


        #replay
        self.beta = 0.7
        self.alpha = 0.5
        
        #main
        self.frames: int = 200_000_000
        self.replay_size: int = 1000000
        #Number of cores to use
        self.workers = 24
        #How many games per core to train
        self.frames_per_iteration_per_game: int = 100000
        self.test_games: int = 400#max((self.game_batch_size)*self.workers//10, 1)
        #Important, as we need to train _at least_ once per move we make, otherwise priorization is dumb
        self.train_per_memory: int = 3

        #game
        self.size: int = 7
        self.lifespan: int = 100

        self.action_space = 4
        #Note that this only affects the moves done by the network. Random moves will still be action masked
        self.action_masking = False
        


    def set_load_path(self, load_path: str) -> None:
        print("Loading a model, this is not a fresh run.")
        if os.path.isfile(load_path):
            self.load_path = load_path
        else:
            print("Couldn't find a file named", load_path)
