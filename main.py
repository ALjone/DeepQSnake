import matplotlib.pyplot as plt
from typing import List
from agent import DQAgent
import torch
from game import Game
from datetime import datetime
from checkpoint_visualizer import Visualizer
from hyperparams import Hyperparams
import time

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
        self.average_scores = [] 
        self.max_scores = []
        self.average_moves = []
        self.apple_less_games = []
        self.last_game_apples = []

        #For visualizing
        self.visualizer = Visualizer()
        self.hyperparams = hyperparams

    def plot(self):
        #TODO Expand and crispen up
        _, axs = plt.subplots((2, 3))
        axs[0].plot(self.average_scores)
        axs[0].set_title("Average amount of apples eaten in a game on average")
        axs[0].set_xlabel("Number of games (in thousands)")
        axs[0].set_ylabel("Average apples eaten")

        axs[1].plot(self.max_scores)
        axs[1].set_title(f"Max amount of apples eaten in {self.test_games} test games")
        axs[1].set_xlabel("Number of games (in thousands)")
        axs[1].set_ylabel("Max apples eaten")

        axs[2].plot(self.average_moves)
        axs[2].set_title("Average amount of moves made in a game on average before dying")
        axs[2].set_xlabel("Number of games (in thousands)")
        axs[2].set_ylabel("Average moves made")

        axs[3].plot(self.apple_less_games)
        axs[3].set_title(f"Amount of games where no apple was eaten in {self.test_games} test games")
        axs[3].set_xlabel("Number of games (in thousands)")
        axs[3].set_ylabel("Games without any apples")

        axs[4].hist(self.last_game_apples)
        axs[4].set_title(f"Distribution of apples eaten in {self.test_games} test games")

        plt.show()
        plt.pause(0.0001)

    def test(self):
        #Having a function to print is bad, should be fixed
        #Also make it plot or something 
        #This is very hacky...
        score = 0
        self.agent.testing = True
        max_score = 0
        moves = 0
        num_bad = 0
        self.last_game_apples = []
        for _ in range(self.test_games):
            temp_score = 0
            state = self.game.reset()
            while(True):
                move = self.agent.get_move(state)
                state, reward, done = self.game.do_action(move)
                moves += 1
                if self.game.ate_last_turn: 
                    score += 1
                    temp_score += 1
                if(done):
                    break
            if temp_score > max_score:
                max_score = temp_score
            if temp_score == 0:
                num_bad += 1
            self.last_game_apples.append(temp_score)
        print(f"   Average over {self.test_games} test games is {round(score/self.test_games, 2)} apples and {int(moves/self.test_games)} moves. Max apples were {max_score}, and number of games without any apples was {num_bad}.")
        self.agent.testing = False
        self.average_scores.append(score)
        self.max_scores.append(max_score)
        self.average_moves.append(moves/self.test_games)
        self.apple_less_games.append(num_bad)
        self.plot()

    def play_episode(self):
        next_state = self.game.reset()
        done = False
        while(not done):
            action = self.agent.get_move(next_state)
            state = next_state
            next_state, reward, done = self.game.do_action(action)
            self.agent.make_memory(action, state, next_state, reward, done)


    def formate_time(self, seconds):
        #https://stackoverflow.com/a/775075
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h == 0: 
            return f'{int(m)} minutes and {int(s)} seconds'
        else:
            return f'{int(h)} hours, {int(m)} minutes and {int(s)} seconds'

    def save(self):
        torch.save(self.agent.trainer.model, "checkpoints/last") #For easily getting it
        torch.save(self.agent.trainer.model, 'models/model_'+ datetime.now().strftime("%m_%d_%Y%H_%M_%S"))

    def main(self):
        start_time = time.time() 
        prev_time = start_time
        print(f"GPU available: {torch.cuda.is_available()}")

        while (self.episodes < self.max_episodes):
            self.play_episode()
            self.episodes += 1
            self.agent.game_is_done()

            #TODO maybe make this only care about the last epoch (or last X epochs) in order to make it more accurate as the snake survives longer
            if self.episodes%self.update_rate == 0 and self.episodes != 0:
                time_left = (time.time()-prev_time)*((self.max_episodes-self.episodes)/self.update_rate)
                prev_time = time.time() 
                print(f"Trained another {self.update_rate} games with exploration rate {round(self.agent._exploration_rate(), 3)}. At {int(self.episodes/1000)}k/{int(self.max_episodes/1000)}k games played. ETA: {self.formate_time(int(time_left))}")
                self.test()
                torch.save(self.agent.trainer.model, "checkpoints/last_checkpoint_in_case_of_crash")
        self.save()

        self.plot()
        print(f"Finished training. Took {self.formate_time(int(time.time()-start_time))}.")
        input("Ready? ")
        self.visualizer.load_game("last", self.hyperparams)
        self.visualizer.visualize()


hyperparams = Hyperparams()
#hyperparams.set_load_path("previous_model")

trainer = Trainer(hyperparams = hyperparams)
trainer.main()