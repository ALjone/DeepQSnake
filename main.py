import matplotlib.pyplot as plt
from typing import List
from agent import DQAgent
import torch
from game import Game
from datetime import datetime
from checkpoint_visualizer import Visualizer
from hyperparams import Hyperparams

class Trainer:
    def __init__(self, hyperparams: Hyperparams) -> None:
        # params
        self.max_episodes: int = hyperparams.max_episodes
        self.episodes: int = 0 
        self.game: Game = hyperparams.game
        
        self.agent: DQAgent = DQAgent(hyperparams)
        
        self.update_rate: int = hyperparams.update_rate
        self.test_games: int = hyperparams.test_games

        #For plotting
        self.scores = [] 

        #For visualizing
        self.visualizer = Visualizer()
        self.hyperparams = Hyperparams

    def plot(self):
        #TODO Expand and crispen up
        plt.plot(self.scores)

    def test(self):
        #Having a function to print is bad, should be fixed
        #Also make it plot or something
        score = 0
        self.agent.testing = True
        max_score = 0
        for _ in range(self.test_games):
            temp_score = 0
            while(True):
                move = self.agent.get_move(self.game)
                self.game.do_action(move)
                if self.game.ate_last_turn: 
                    score += 1
                    temp_score += 1
                if(self.game.final_state):
                    self.game.reset()
                    break
            if temp_score > max_score:
                max_score = temp_score
        print(f"   Average over {self.test_games} test games is {round(score/self.test_games, 2)} apples. Max apples were {max_score}")
        self.agent.testing = False
        self.scores.append(score)

    def play_episode(self):
        while(True):
            move = self.agent.get_move(self.game)

            self.agent.make_memory(self.game, move)
            self.game.do_action(move)


            if(self.game.final_state):
                self.episodes += 1
                self.agent.make_memory(self.game, None)
                self.game.reset()
                return

    def main(self):
        print(f"GPU available: {torch.cuda.is_available()}")

        while (self.episodes < self.max_episodes):
            self.play_episode()
            self.agent.train()

            #TODO save models to a checkpoint folder, and make a script that easily visualizes it
            if self.episodes%self.update_rate == 0 and self.episodes != 0:
                print(f"Trained another {self.update_rate} games with exploration rate {round(self.agent._exploration_rate(), 3)}. At {int(self.episodes/1000)}k/{int(self.max_episodes/1000)}k games played.")
                self.test()
                torch.save(self.agent.trainer.model, "checkpoints/" + str(self.episodes) + "_" + str(self.max_episodes))

        torch.save(self.agent.trainer.model, "checkpoints/last") #For easily getting it
        torch.save(self.agent.trainer.model, 'model_'+ datetime.now().strftime("%m_%d_%Y%H_%M_%S"))
        torch.save(self.agent.trainer.model, 'previous_model')

        self.plot()

        input("Ready? ")
        self.visualizer.load_game("last", self.hyperparams)
        self.visualizer.visualize()

        input("Ready again? ")
        from ReplayMemoryGraphic import ReplayGraphics
        ReplayGraphics(hyperparams.game.mapsize, self.agent.bank, self.agent.trainer.model)
       


hyperparams = Hyperparams()
hyperparams.set_load_path("previous_model")

trainer = Trainer(hyperparams = hyperparams)
trainer.main()



