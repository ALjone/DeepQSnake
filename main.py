from multiprocessing import Pool
from agent import DQAgent
import torch
from env import snake_env
#from snake import SnakeGame as snake_env
from simple_network import simple_network
from datetime import datetime
import numpy as np
from hyperparams import Hyperparams
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import gym



class Trainer:
    def __init__(self, hyperparams: Hyperparams) -> None:
        # params
        self.max_frames: int = hyperparams.frames
        self.frames = 0

        
        self.env: gym.vector.AsyncVectorEnv = gym.vector.SyncVectorEnv([lambda: gym.wrappers.FrameStack(snake_env(hyperparams.size, hyperparams.lifespan, hyperparams.device), hyperparams.frame_stack) for _ in range(hyperparams.workers)])
        self.test_env = gym.wrappers.FrameStack(snake_env(hyperparams.size, hyperparams.lifespan, hyperparams.device), hyperparams.frame_stack)

        self.agent: DQAgent = DQAgent(hyperparams)
        
        self.update_rate: int = 1#hyperparams.batch_times
        self.test_games: int = hyperparams.test_games

        self.hyperparams = hyperparams

        self.writer = SummaryWriter()

    def update_writer(self, games_per_second, actions_per_second):
        self.writer.add_scalar("Other/Games per second", games_per_second, self.frames)
        self.writer.add_scalar("Other/Actions per second", actions_per_second, self.frames)


    def test(self):
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
        for i in tqdm(range(self.test_games), leave=False, desc="Testing"):
            temp_score = 0
            state = np.array(self.test_env.reset())
            done = False
            while(not done):
                move = int(self.agent.get_move(state, self.test_env.valid_moves()).item())
                actions[move] += 1
                state, reward, done, _ = self.test_env.step(move)
                state = np.array(state)
                moves += 1
                if i == self.test_games-1: #Only do this for last game
                    s = torch.tensor(state).flatten(0, 1).to(self.agent.trainer.device)
                    _, value, action_values = self.agent.trainer.model(s, return_separate = True)
                    V.append(value.item())
                    for A, value in zip(As, action_values.squeeze()):
                        A.append(value.item())
                if reward == 1:
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

        self.writer.add_scalar("Average/Average moves", moves/self.test_games, self.frames)
        self.writer.add_scalar("Average/Average apples", sum(apples)/self.test_games, self.frames)
        self.writer.add_scalar("Apple num/Max apples", max_score, self.frames)
        self.writer.add_scalar("Apple num/No apples", (num_bad/self.test_games)*100, self.frames)
        self.writer.add_scalar("Other/std of average apple estimate", std, self.frames)
        self.writer.add_histogram("Apple Distribution", torch.tensor(last_game_apples))

        plt.plot(V)
        self.writer.add_figure("V and A/V as function of time", plt.gcf(), self.frames)
        
        highest = max(map(max, As))
        lowest = min(map(min, As))
        for A, dir in zip(As, ["Left", "Right", "Down", "Up"]):
            plt.plot(A)
            plt.ylim((lowest*1.1, highest*1.1))
            plt.legend([dir])
            self.writer.add_figure(f"V and A/{dir}-value as function of time", plt.gcf(), self.frames)

    def get_benchmark(self):
        score = 0
        start_time = time.time()
        sims = 1000
        score = 0
        self.agent.testing = True
        max_score = 0
        moves = 0
        num_bad = 0
        actions = [0, 0, 0, 0]
        for _ in tqdm(range(sims), leave=False):
            temp_score = 0
            self.test_env.reset()
            done = False
            while(not done):
                move = self.agent._get_random(self.test_env.valid_moves())
                actions[move] += 1
                _, reward, done = self.test_env.step(move)
                moves += 1
                if reward == 1: 
                    score += 1
                    temp_score += 1
            if temp_score > max_score:
                max_score = temp_score
            if temp_score == 0:
                num_bad += 1
        
        actions = [round((a/sum(actions))*100, 2) for a in actions]
        print(f"Benchmark: \n\tAverage apples: {round(score/sims, 2)}\n\tAverage moves: {int(moves/sims)} moves\n\tMax apples: {max_score}\n\tGames without apples: {round((num_bad/sims)*100, 2)}%\n\tPlayed {round(sims/(time.time()-start_time), 2)} g/s\n\tAction distribution: {actions}")
        self.agent.testing = False

    def formate_time(self, seconds):
        #https://stackoverflow.com/a/775075
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if m == 0:
            return f'{int(s)} seconds' 
        if h == 0: 
            return f'{int(m)} minutes and {int(s)} seconds' 
        else:
            return f'{int(h)} hours, {int(m)} minutes and {int(s)} seconds'

    def save(self):
        torch.save(self.agent.trainer.model, "checkpoints/last") #For easily getting it
        torch.save(self.agent.trainer.model, 'models/model_'+ datetime.now().strftime("%m_%d_%Y%H_%M_%S"))

    def _train(self, memories):
        train_time = time.time()
        self.agent.train((memories//self.hyperparams.batch_size)*self.hyperparams.train_per_memory)
        return time.time()-train_time

    def play_episodes(self):
        s = time.time()
        reward = 0
        next_states = torch.tensor(self.env.reset()).to(self.hyperparams.device)
        next_states = self.env.reset()
        done = np.zeros(self.hyperparams.workers, dtype=np.bool8)
        games_played = 0
        #TODO: Not elegant
        frames_processed = 0
        for _ in tqdm(range(self.hyperparams.frames_per_iteration_per_game), leave = False, desc="Experiencing and training"):
            frames_processed += self.hyperparams.workers
            states = next_states
            actions = self.agent.get_move(states, [1 for _ in range(self.hyperparams.action_space)])
            next_states, reward, done, _ = self.env.step(actions)
            next_states = torch.tensor(next_states).to(self.hyperparams.device)
            self.agent.make_memory((actions, states, next_states, reward, done))
            games_played += np.sum(done)
            if frames_processed > self.hyperparams.batch_size:
                self.agent.train(self.hyperparams.train_per_memory)
                frames_processed = 0

        return self.hyperparams.frames_per_iteration_per_game*self.hyperparams.workers, time.time()-s, games_played


    def train(self, benchmark = False):
        start_time = time.time() 
        print(f"GPU available: {torch.cuda.is_available()}")
        if benchmark:
            self.get_benchmark()
        prev_time = start_time

        total_memories = 0
        total_games = 0
        #Main loop
        while (self.frames < self.max_frames):
            memories, exp_time, games_played = self.play_episodes()

            #train_time = self._train(memories)

            total_memories += memories
            total_games += games_played

            torch.save(self.agent.trainer.model, "checkpoints/last_checkpoint_in_case_of_crash")

            self.frames += memories

            #time_left = (time.time()-prev_time)*(self.max_frames-self.frames)
            print(f"\nAt {int(total_games/1000)}k games played.", #/{round(self.max_episodes/1000, 1)}k games played.",# ETA: {self.formate_time(int(time_left))}.",
            f"Playing {round(games_played/(time.time()-prev_time), 2)} g/s and doing {round(memories/(time.time()-prev_time), 2)} a/s.", 
            f"Spent {self.formate_time(exp_time)} experiencing and training.",#f"Spent {self.formate_time(exp_time)} experiencing and {self.formate_time(train_time)} training.", 
            f"Trained on {int(self.agent.trained_times/1000)}k samples so far, done {int(total_memories/1000)}k actions. {self.formate_time(time.time()-start_time)} elapsed")

            self.update_writer(games_per_second=games_played/(time.time()-prev_time), actions_per_second=memories/(time.time()-prev_time))
            self.test()
            prev_time = time.time() 

        self.save()

        print(f"Finished training. Took {self.formate_time(int(time.time()-start_time))}.")



if __name__ == "__main__":
    hyperparams = Hyperparams() 
    #hyperparams.set_load_path("checkpoints\\last_checkpoint_in_case_of_crash")
    #hyperparams.set_load_path("checkpoints\\very_good_7x7")

    trainer = Trainer(hyperparams = hyperparams)
    trainer.train(False)