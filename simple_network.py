import torch
from new_model import SnakeBrain
#from experimental_model import SnakeBrain
import numpy as np

class simple_network:
    def __init__(self, mapsize, action_space, action_masking, frame_stack, state_dict, device) -> None:
        self.model: SnakeBrain = SnakeBrain(mapsize, action_space, frame_stack)
        

        self.model.load_state_dict(state_dict)


        #Try high
        self.device = torch.device("cpu")
        self.action_masking = action_masking
        self.model.eval()
    
    def reload_model(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

    def random(self, valid_moves: torch.Tensor):
        max_inds, = torch.where(valid_moves == valid_moves.max())
        idx = np.random.randint(0, max_inds.shape[0])
        return max_inds[idx]#np.random.choice(max_inds).item()

    def rand_argmax(self, tens):
        argmaxes = torch.where(tens == tens.max())[1]
        idx = np.random.randint(0, argmaxes.shape[0])
        return argmaxes[idx]#np.random.choice(argmaxes.to(self.device))

    def predict(self, features: torch.Tensor, valid_moves: torch.Tensor):
        if np.random.uniform() >= 0.95:
            return self.random(valid_moves)
        features = torch.tensor(features) if type(features) == np.ndarray else features
        valid_moves = torch.tensor(valid_moves) if type(valid_moves) == np.ndarray else valid_moves
        with torch.no_grad():
            features = features.to(self.device)
            pred = self.model(features)
            if not self.action_masking:
                return self.rand_argmax(pred)
            #NOTE this is action masked unless the line above is uncommented
            #https://discuss.pytorch.org/t/masked-argmax-in-pytorch/105341
            large = torch.finfo (pred.dtype).max
            return self.rand_argmax((pred - large * (1 - valid_moves) - large * (1 - valid_moves)))
