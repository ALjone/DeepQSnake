#import matplotlib.pyplot as plt
from typing import List
from unicodedata import decimal
from agent import DQAgent
import torch
from game import Game
from datetime import datetime
from checkpoint_visualizer import Visualizer
from hyperparams import Hyperparams
import time
from matplotlib import pyplot as plt

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
        _, axs = plt.subplots(2, 3)
        axs[0, 0].plot(self.average_scores)
        axs[0, 0].set_title("Average amount of apples eaten in a game on average")
        axs[0, 0].set_xlabel("Number of games (in thousands)")
        axs[0, 0].set_ylabel("Average apples eaten")

        axs[0, 1].plot(self.max_scores)
        axs[0, 1].set_title(f"Max amount of apples eaten in {self.test_games} test games")
        axs[0, 1].set_xlabel("Number of games (in thousands)")
        axs[0, 1].set_ylabel("Max apples eaten")

        axs[0, 2].plot(self.average_moves)
        axs[0, 2].set_title("Average amount of moves made in a game on average before dying")
        axs[0, 2].set_xlabel("Number of games (in thousands)")
        axs[0, 2].set_ylabel("Average moves made")

        axs[1, 0].plot(self.apple_less_games)
        axs[1, 0].set_title(f"Amount of games where no apple was eaten in {self.test_games} test games")
        axs[1, 0].set_xlabel("Number of games (in thousands)")
        axs[1, 0].set_ylabel("Games without any apples")

        axs[1, 1].hist(self.last_game_apples)
        axs[1, 1].set_title(f"Distribution of apples eaten in {self.test_games} test games")

        plt.show()
        #plt.pause(0.0001)

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
        actions = [0, 0, 0, 0]
        for _ in range(self.test_games):
            temp_score = 0
            state = self.game.reset(True)
            done = False
            while(not done):
                move = self.agent.get_move(state, self.game.valid_moves())
                actions[move] += 1
                state, reward, done = self.game.do_action(move)
                moves += 1
                if self.game.ate_last_turn: 
                    score += 1
                    temp_score += 1
            if temp_score > max_score:
                max_score = temp_score
            if temp_score == 0:
                num_bad += 1
            self.last_game_apples.append(temp_score)
        #print(f"   Average over {self.test_games} test games is {round(score/self.test_games, 2)} apples and {int(moves/self.test_games)} moves. Max apples were {max_score}, and number of games without any apples was {round((num_bad/self.test_games)*100, 2)}%.")
        actions = [round((a/sum(actions))*100, 2) for a in actions]
        print(f"\tAverage apples: {round(score/self.test_games, 2)}\n\tAverage moves: {int(moves/self.test_games)} moves\n\tMax apples: {max_score}\n\tGames without apples: {round((num_bad/self.test_games)*100, 2)}%\n\tAction distribution: {actions}")
        self.agent.testing = False
        self.average_scores.append(score/self.test_games)
        self.max_scores.append(max_score)
        self.average_moves.append(moves/self.test_games)
        self.apple_less_games.append((num_bad/self.test_games)*100)
        #self.plot()

    def get_benchmark(self):
        score = 0
        s = time.time()
        sims = 1000
        score = 0
        self.agent.testing = True
        max_score = 0
        moves = 0
        num_bad = 0
        self.last_game_apples = []
        actions = [0, 0, 0, 0]
        for _ in range(sims):
            temp_score = 0
            self.game.reset(True)
            done = False
            while(not done):
                move = self.agent._get_random(torch.tensor(self.game.valid_moves()))
                actions[move] += 1
                _, _, done = self.game.do_action(move)
                moves += 1
                if self.game.ate_last_turn: 
                    score += 1
                    temp_score += 1
            if temp_score > max_score:
                max_score = temp_score
            if temp_score == 0:
                num_bad += 1
        
        actions = [round((a/sum(actions))*100, 2) for a in actions]
        print(f"Benchmark: \n\tAverage apples: {round(score/sims, 2)}\n\tAverage moves: {int(moves/sims)} moves\n\tMax apples: {max_score}\n\tGames without apples: {round((num_bad/sims)*100, 2)}%\n\tPlayed {round(sims/(time.time()-s), 1)} g/s\n\tAction distribution: {actions}")
        self.agent.testing = False

    def play_episode(self):
        next_state = self.game.reset()
        done = False
        while(not done):
            action = self.agent.get_move(next_state, self.game.valid_moves())
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
        #TODO TQDM  
        start_time = time.time() 
        print(f"GPU available: {torch.cuda.is_available()}")
        self.get_benchmark()

        prev_time = start_time
        while (self.episodes < self.max_episodes):
            self.play_episode()
            self.episodes += 1
            self.agent.game_is_done()    

            #TODO maybe make this only care about the last epoch (or last X epochs) in order to make it more accurate as the snake survives longer
            if self.episodes%self.update_rate == 0 and self.episodes != 0:
                time_left = (time.time()-prev_time)*((self.max_episodes-self.episodes)/self.update_rate)
                print(f"\nAt {round(self.episodes/1000, 1)}k/{int(self.max_episodes/1000)}k games played. Exploration rate {round(self.agent._exploration_rate(), 3)}. ETA: {self.formate_time(int(time_left))}.",
                f"Playing {round(self.update_rate/(time.time()-prev_time), 1)} g/s")
                prev_time = time.time() 
                self.test()
            if self.episodes % self.hyperparams.model_save_rate == 0:
                torch.save(self.agent.trainer.model, "checkpoints/last_checkpoint_in_case_of_crash")
        self.save()

        print(f"Finished training. Took {self.formate_time(int(time.time()-start_time))}.")
        self.plot()
        print("Expected value at start:", torch.round(self.agent.trainer.model(torch.tensor(self.game.reset())), decimals = 2))
        #input("Ready? ")
        #self.visualizer.load_game("last", self.hyperparams)
        #self.visualizer.visualize()


hyperparams = Hyperparams()
hyperparams.set_load_path("checkpoints\\almost_perfect")

trainer = Trainer(hyperparams = hyperparams)
trainer.main()