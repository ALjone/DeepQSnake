from agent import DQAgent
import torch
from game_map import Game
from datetime import datetime
import time
class Trainer:
    def __init__(self, load_warmstart_model: bool = False, load_model: bool = False) -> None:
        # params
        size: int = 10
        lifespan: int = 25
        memory_bank_size = 10000
        self.max_episodes: int = 2000
        self.episodes = 0
        self.game: Game = Game(size, lifespan)
        
        if load_warmstart_model:
            print("Loading warm-start model")
            self.agent = DQAgent(self.max_episodes, bank_size = memory_bank_size, load_path="warm_start_model")
        else:
            self.agent = DQAgent(self.max_episodes, bank_size = memory_bank_size)
            
        if load_model: 
            self.agent.trainer.model = torch.load("previous_model")
        self.warm_start = 1
        
    def run(self):
        reward = 0
        while(True):
            if self.graphics is not None: 
                self.graphics.updateWin(self.game, reward)

            move = self.agent.get_move(self.game)
            self.game.do_action(move)
            reward += self.agent._get_reward(self.game)

            if(self.game.dead):
                self.agent.get_move(self.game)
                break    

    def play_episode(self):
        while(True):
            move = self.agent.get_move(self.game)
            self.agent.make_memory(self.game, move)
            self.game.do_action(move)

            if(self.game.dead):
                score = self.game.score
                self.episodes += 1
                self.agent.make_memory(self.game, 5)
                self.game.reset(warm_start=self.warm_start)
                start = time.time()
                self.agent.train()
                stop = time.time()

                return score

    def main(self):
        avgscore = 0
        fourth = False
        half = False
        threefourths = False

        print(f"GPU available: {torch.cuda.is_available()}")

        while (self.episodes < self.max_episodes):
            avgscore += self.play_episode()


            if self.episodes%100 == 0 and self.episodes != 0:
                print("Over the last 100 games I've got an average score of", avgscore/100, "Played in total", self.episodes, "games")
                avgscore = 0
            
            
            if self.episodes > self.max_episodes/4 and not fourth:
                print("One fourth is done. Increasing the warm start range of the snake, this will make it harder.")
                fourth = True
                self.warm_start = 2
            
            if self.episodes > self.max_episodes/2 and not half:
                print("Half is done. Increasing the warm start range of the snake, this will make it harder.")
                half = True
                self.warm_start = 3
            
            if self.episodes > (3*self.max_episodes)/4 and not threefourths:
                print("Three fourths are done. Increasing the warm start range of the snake, this will make it harder.")
                self.warm_start = 0
                threefourths = True


        input("Ready? ")
        self.agent.testing = True
        from graphics_module import Graphics
        self.graphics: Graphics = None
        for _ in range(100):
            if self.graphics is None:
                self.graphics = Graphics(self.game.mapsize)
            self.run()
            self.game.reset(warm_start=self.warm_start)

        torch.save(self.agent.trainer.model, 'model_'+ datetime.now().strftime("%m_%d_%Y%H_%M_%S"))
        torch.save(self.agent.trainer.model, 'previous_model')

trainer = Trainer(load_model = False)
trainer.main()