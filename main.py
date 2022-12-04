from agent import DQAgent
import torch
from game import Game
from datetime import datetime
import numpy as np
from hyperparams import Hyperparams
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, hyperparams: Hyperparams) -> None:
        # params
        self.max_episodes: int = hyperparams.max_episodes
        self.game: Game = hyperparams.game
        
        self.agent: DQAgent = DQAgent(hyperparams)
        
        self.update_rate: int = hyperparams.update_rate
        self.test_games: int = hyperparams.test_games

        self.hyperparams = hyperparams

        self.writer = SummaryWriter()
        self.moves_made = 0

    def update_writer(self, games_per_second, episodes, noise_level):
        self.writer.add_scalar("Other/Games per second", games_per_second, episodes//self.update_rate)
        self.writer.add_scalar("Other/Noise level", noise_level, episodes//self.update_rate)


    def test(self, episodes):
        #Having a function to print is bad, should be fixed
        #Also make it plot or something 
        #This is very hacky...
        apples = []
        self.agent.testing = True
        max_score = 0
        moves = 0
        num_bad = 0
        last_game_apples = []
        actions = [0, 0, 0, 0]
        V = []
        As = [[], [], [], []]
        for i in tqdm(range(self.test_games), leave=False):
            temp_score = 0
            state = self.game.reset(True)
            done = False
            while(not done):
                move = self.agent.get_move(state, self.game.valid_moves())
                actions[move] += 1
                state, _, done = self.game.do_action(move)
                moves += 1
                if i == self.test_games-1: #Only do this for last game
                    _, value, action_values = self.agent.trainer.model(state, return_separate = True)
                    V.append(value.item())
                    for A, value in zip(As, action_values.squeeze()):
                        A.append(value.item())
                if self.game.ate_last_turn:
                    temp_score += 1
            apples.append(temp_score)
            if temp_score > max_score:
                max_score = temp_score
            if temp_score == 0:
                num_bad += 1
            last_game_apples.append(temp_score)
        actions = [round((a/sum(actions))*100, 2) for a in actions]
        std = np.std(apples)
        print(f"\tAverage apples: {round(sum(apples)/self.test_games, 2)} (std: {round(std, 5)})\n\tAverage moves: {int(moves/self.test_games)} moves\n\tMax apples: {max_score}\n\tGames without apples: {round((num_bad/self.test_games)*100, 2)}%\n\tAction distribution: {actions}")
        self.agent.testing = False

        self.writer.add_scalar("Average/Average apples", sum(apples)/self.test_games, episodes//self.update_rate)

        self.writer.add_scalar("Average/Average moves", moves/self.test_games, episodes//self.update_rate)
        
        self.writer.add_scalar("Apple num/Max apples", max_score, episodes//self.update_rate)

        self.writer.add_scalar("Apple num/No apples", (num_bad/self.test_games)*100, episodes//self.update_rate)

        self.writer.add_scalar("Other/std of average apple estimate", std, episodes//self.update_rate)

        self.writer.add_histogram("Apple Distribution", torch.tensor(last_game_apples))

        plt.plot(V)
        self.writer.add_figure("V and A/V as function of time", plt.gcf(), episodes//self.update_rate)
        
        highest = max(map(max, As))
        lowest = min(map(min, As))
        for A, dir in zip(As, ["Left", "Right", "Down", "Up"]):
            plt.plot(A)
            plt.ylim((lowest*1.1, highest*1.1))
            plt.legend([dir])
            self.writer.add_figure(f"V and A/{dir}-value as function of time", plt.gcf(), episodes//self.update_rate)

    def get_benchmark(self):
        score = 0
        s = time.time()
        sims = 1000
        score = 0
        self.agent.testing = True
        max_score = 0
        moves = 0
        num_bad = 0
        actions = [0, 0, 0, 0]
        for _ in tqdm(range(sims), leave=False):
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
        print(f"Benchmark: \n\tAverage apples: {round(score/sims, 2)}\n\tAverage moves: {int(moves/sims)} moves\n\tMax apples: {max_score}\n\tGames without apples: {round((num_bad/sims)*100, 2)}%\n\tPlayed {round(sims/(time.time()-s), 2)} g/s\n\tAction distribution: {actions}")
        self.agent.testing = False

    def play_episode(self):
        next_state = self.game.reset()
        done = False
        while(not done):
            action = self.agent.get_move(next_state, self.game.valid_moves())
            state = next_state
            next_state, reward, done = self.game.do_action(action)
            self.agent.make_memory(action, state, next_state, reward, done)
            self.moves_made += 1

            if self.moves_made%self.hyperparams.train_rate == 0:
                self.agent.train()


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

    def run(self, benchmark = False):
        start_time = time.time() 
        print(f"GPU available: {torch.cuda.is_available()}")
        if benchmark:
            self.get_benchmark()
        episodes = 0
        prev_time = start_time
        while (episodes < self.max_episodes):
            for i in tqdm(range(self.update_rate), leave=False):
                self.play_episode()
                episodes += 1
                self.agent.train()    
                if episodes % self.hyperparams.model_save_rate == 0:
                    torch.save(self.agent.trainer.model, "checkpoints/last_checkpoint_in_case_of_crash")

            #if episodes%self.update_rate == 0 and episodes != 0:
            time_left = (time.time()-prev_time)*((self.max_episodes-episodes)/self.update_rate)
            print(f"\nAt {round(episodes/1000, 1)}k/{round(self.max_episodes/1000, 1)}k games played. Noise level for exploration: {round(self.agent.trainer.model.get_noise_level(), 4)}. ETA: {self.formate_time(int(time_left))}.",
            f"Playing {round(self.update_rate/(time.time()-prev_time), 2)} g/s")
            self.test(episodes)
            self.update_writer(games_per_second=self.update_rate/(time.time()-prev_time), episodes=episodes, noise_level = self.agent.trainer.model.get_noise_level())
            prev_time = time.time() 
        self.save()

        print(f"Finished training. Took {self.formate_time(int(time.time()-start_time))}.")
        print("Expected value at start:", torch.round(self.agent.trainer.model(torch.tensor(self.game.reset())), decimals = 2))



if __name__ == "__main__":

    hyperparams = Hyperparams() 
    hyperparams.set_load_path("checkpoints\\last_checkpoint_in_case_of_crash")

    trainer = Trainer(hyperparams = hyperparams)
    trainer.run()