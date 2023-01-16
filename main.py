from multiprocessing import Pool
from agent import DQAgent
import torch
from game import snake_env
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

def play_episodes(episodes_to_play, model: simple_network, hyperparams: Hyperparams, env: snake_env):
    memories = []
    #env = snake_env(hyperparams.size, hyperparams.size, hyperparams.lifespan)
    reward = 0
    for _ in range(episodes_to_play):
        queue = deque([], maxlen=hyperparams.frame_stack)
        next_state = env.reset()
        for _ in range(hyperparams.frame_stack):
            queue.append(next_state)
        done = False
        while(not done):
            state = torch.concatenate(tuple(queue), 0)
            action = model.predict(state, env.valid_moves())
            next_state, reward, done = env.step(action)
            queue.append(next_state)
            memories.append((action, state, torch.concatenate(tuple(queue), 0), reward, done))
    return memories

class Trainer:
    def __init__(self, hyperparams: Hyperparams) -> None:
        # params
        self.max_episodes: int = hyperparams.max_episodes
        self.game: snake_env = snake_env(hyperparams.size, hyperparams.size, hyperparams.lifespan, hyperparams.device)

        self.agent: DQAgent = DQAgent(hyperparams)
        
        self.update_rate: int = 1#hyperparams.batch_times
        self.test_games: int = hyperparams.test_games

        self.hyperparams = hyperparams

        self.writer = SummaryWriter()

    def update_writer(self, games_per_second, actions_per_second, episodes):
        self.writer.add_scalar("Other/Games per second", games_per_second, episodes//self.update_rate)
        self.writer.add_scalar("Other/Actions per second", actions_per_second, episodes//self.update_rate)


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
        queue = deque([], maxlen=self.hyperparams.frame_stack)
        for i in tqdm(range(self.test_games), leave=False):
            temp_score = 0
            state = self.game.reset()
            for _ in range(self.hyperparams.frame_stack):
                queue.append(state)
            done = False
            while(not done):
                state = torch.concatenate(tuple(queue), 0)
                move = self.agent.get_move(state, self.game.valid_moves())
                actions[move] += 1
                state, reward, done = self.game.step(move)
                queue.append(torch.tensor(state) if type(state) == np.ndarray else state)
                moves += 1
                if i == self.test_games-1: #Only do this for last game
                    state = torch.concatenate(tuple(queue), 0)
                    state = torch.tensor(state) if type(state) == np.ndarray else state
                    _, value, action_values = self.agent.trainer.model(state.to(self.agent.trainer.device), return_separate = True)
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
            self.game.reset()
            done = False
            while(not done):
                move = self.agent._get_random(self.game.valid_moves())
                actions[move] += 1
                _, reward, done = self.game.step(move)
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

    def experience(self, models, hyperparams, envs, num_per_core):
        memories = 0
        exp_time = time.time()
        for model in models:
            model.reload_model(self.agent.trainer.model.state_dict())
        with Pool(processes=self.hyperparams.workers) as pool:
            multiple_results = [pool.apply_async(play_episodes, (n, m, h, e)) for n, m, h, e in zip(num_per_core, models, hyperparams, envs)]
            results = [res.get() for res in multiple_results]

        for res in results:
            memories += len(res)
            self.agent.make_memory(res)

        return memories, time.time()-exp_time

    def _train(self, memories):
        train_time = time.time()
        self.agent.train((memories//self.hyperparams.batch_size)*self.hyperparams.train_per_memory)
        return time.time()-train_time

    def train(self, benchmark = False):
        start_time = time.time() 
        print(f"GPU available: {torch.cuda.is_available()}")
        if benchmark:
            self.get_benchmark()
        episodes = 0
        prev_time = start_time
        
        #Initialize distribution
        num_per_core = [self.hyperparams.game_batch_size for _ in range(self.hyperparams.workers)]
        hyperparams = [self.hyperparams for _ in range(self.hyperparams.workers)]
        models = [simple_network(self.hyperparams.size, self.hyperparams.action_space, self.hyperparams.action_masking, self.hyperparams.frame_stack, self.agent.trainer.model.state_dict(), self.hyperparams.device) for _ in range(self.hyperparams.workers)]
        envs = [snake_env(self.hyperparams.size, self.hyperparams.size, self.hyperparams.lifespan, self.hyperparams.device) for _ in range(self.hyperparams.workers)]


        total_memories = 0
        #Main loop
        while (episodes < self.max_episodes):
            memories, exp_time = self.experience(models, hyperparams, envs, num_per_core)

            train_time = self._train(memories)

            total_memories += memories

            torch.save(self.agent.trainer.model, "checkpoints/last_checkpoint_in_case_of_crash")

            eps = sum(num_per_core)
            episodes += eps

            time_left = (time.time()-prev_time)*(self.max_episodes-episodes)
            print(f"\nAt {round(episodes/1000, 1)}k games played.", #/{round(self.max_episodes/1000, 1)}k games played.",# ETA: {self.formate_time(int(time_left))}.",
            f"Playing {round(eps/(time.time()-prev_time), 2)} g/s and doing {round(memories/(time.time()-prev_time), 2)} a/s. Spent {self.formate_time(exp_time)} experiencing and {self.formate_time(train_time)} training.", 
            f"Trained on {int(self.agent.trained_times/1000)}k samples so far, done {int(total_memories/1000)}k actions")

            self.update_writer(games_per_second=eps/(time.time()-prev_time), actions_per_second=memories/(time.time()-prev_time), episodes=episodes)
            self.test(episodes)
            prev_time = time.time() 

        self.save()

        print(f"Finished training. Took {self.formate_time(int(time.time()-start_time))}.")



if __name__ == "__main__":
    hyperparams = Hyperparams() 
    hyperparams.set_load_path("checkpoints\\last_checkpoint_in_case_of_crash")
    #hyperparams.set_load_path("checkpoints\\very_good_7x7")

    trainer = Trainer(hyperparams = hyperparams)
    trainer.train(False)