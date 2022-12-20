from game import snake_env
from agent import DQAgent
from hyperparams import Hyperparams
import time
import torch
from collections import deque
from snake import SnakeGame as snake_env
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
        
        #Always run visualizations on CPU
        self.hyperparams.device = "cpu"
        self.hyperparams.set_load_path("checkpoints/" + path)
        
        self.frame_stack = self.hyperparams.frame_stack
        self.game: snake_env = snake_env(self.hyperparams.size, self.hyperparams.size, 300)
        print(self.hyperparams.device)
        self.agent: DQAgent = DQAgent(self.hyperparams)

        self.game_loaded = True
        
    def visualize(self):
        if not self.game_loaded: 
            #Should be a better exception probably.
            raise BaseException("A game has not been loaded for the visualizer. Please call load_game first.")
        from graphics_module import Graphics
        self.graphics: Graphics = Graphics(self.hyperparams.size)

        self.agent.testing = True
        while(True):
            self.__run()
            self.agent.trainer.reload_model()

    def __run(self):
        total_reward = 0
        queue = deque([], maxlen=self.frame_stack)
        state = torch.tensor(self.game.reset())
        for _ in range(self.frame_stack):
            queue.append(state)
        prev_move = -1
        self.agent.trainer.model.eval()
        counter = 0
        while(True):
            #Should probably time how long everything takes rather than using a flat 0.1s
            time.sleep(0.05)
            self.graphics.updateWin(state, total_reward)
            state = torch.cat(tuple(queue))
            move = self.agent.get_move(state, self.game.valid_moves())
            #print(move)
            state, reward, done = self.game.step(move)
            queue.append(torch.tensor(state))
            state = torch.tensor(state)
            total_reward += reward
            if prev_move == reverse[move]:
                done = True
            prev_move = move
            counter += 1
            if(done):
                self.graphics.updateWin(state, total_reward)
                self.game.reset()
                if prev_move == reverse[move]:
                    print("Aborted due to repeated move ")
                break