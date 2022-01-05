from agent import DQAgent
import torch
from game import Game
from datetime import datetime
import time
from hyperparams import Hyperparams

class Trainer:
    def __init__(self, hyperparams: Hyperparams) -> None:
        # params
        size: int = hyperparams.game.mapsize
        lifespan: int = hyperparams.game.lifespan
        self.max_episodes: int = hyperparams.max_episodes
        self.episodes = 0 
        self.game: Game = Game(size, lifespan)
        
        self.agent = DQAgent(hyperparams)

        self.action_space = hyperparams.action_space
        self.moves = [0]*self.action_space
        self.update_rate = hyperparams.update_rate
    
    #TODO Rename to test, make it take in graphic: bool, and print stuff, maybe
    def run(self):
        reward = 0
        while(True):
            if self.graphics is not None: 
                self.graphics.updateWin(self.game, reward)
            
            move = self.agent.get_move(self.game)
            self.game.do_action(move)
            reward += self.game.get_reward()

            if(self.game.final_state):
                move = self.agent.get_move(self.game)
                self.graphics.updateWin(self.game, reward)
                break

    def play_episode(self):
        while(True):
            move = self.agent.get_move(self.game)

            self.agent.make_memory(self.game, move)
            self.game.do_action(move)

            self.moves[move] += 1

            if(self.game.final_state):
                score = self.game.score
                self.episodes += 1
                self.agent.make_memory(self.game, None)
                self.game.reset()

                return score

    def main(self):
        print(f"GPU available: {torch.cuda.is_available()}")

        avgscore = 0
        while (self.episodes < self.max_episodes):
            avgscore += self.play_episode()
            self.agent.train()

            #TODO save models to a checkpoint folder, and make a script that easily visualizes it
            if self.episodes%self.update_rate == 0 and self.episodes != 0:
                printlist = [round(num/sum(self.moves), 2) for num in self.moves]
                print(f"Over the last {self.update_rate} games I've got an average score of", avgscore/self.update_rate, "Played in total", self.episodes, "games.",
                f"Last {self.update_rate} games the move distribution was:", printlist)
                print("Current exploration rate:", round(self.agent._exploration_rate(), 3))  
                self.moves = [0]*self.action_space
                avgscore = 0
                torch.save(self.agent.trainer.model, "checkpoints/" + str(self.episodes) + "_" + str(self.max_episodes))

        torch.save(self.agent.trainer.model, 'model_'+ datetime.now().strftime("%m_%d_%Y%H_%M_%S"))
        torch.save(self.agent.trainer.model, 'previous_model')

        input("Ready? ")
        self.agent.testing = True
        from graphics_module import Graphics
        self.graphics: Graphics = None
        #That's how much I'm gonna bother watching
        for _ in range(25):
            if self.graphics is None:
                self.graphics = Graphics(self.game.mapsize)
            self.run()
            self.game.reset()


hyperparams = Hyperparams()
hyperparams.set_load_path("previous_model")

trainer = Trainer(hyperparams = hyperparams)
trainer.main()


input("Ready again? ")
from ReplayMemoryGraphic import ReplayGraphics
graphic = ReplayGraphics(trainer.game.mapsize, trainer.agent.bank, trainer.agent.trainer.model)
