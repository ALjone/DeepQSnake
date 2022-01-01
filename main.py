from agent import DQAgent
import torch
from game_map import Game
from graphics_module import Graphics
from datetime import datetime

class Trainer:
    def __init__(self, load_warmstart_model: bool = False, load_model: bool = False) -> None:
        size: int = 10
        lifespan: int = 30
        range: int = 1
        self.graphics: Graphics = None
        self.max_game_steps: int = 1000000
        self.game_steps: int = 0

        self.game: Game = Game(size, lifespan, range)
        
        if load_warmstart_model:
            print("Loading warm-start model")
            self.agent = DQAgent(self.max_game_steps, load_path="warm_start_model")
        else:
            self.agent = DQAgent(self.max_game_steps)
            
        if load_model: 
            self.agent.trainer.model = torch.load("previous_model")
        self.warm_start = 1
        
    def run(self):
        while(True):
            self.game_steps +=1
            if self.graphics is not None: 
                self.graphics.updateWin(self.game)
                #print(self.agent.trainer.model(self.agent._get_features(self.game.game_map), self.game.moves_since_ate/self.game.lifespan))

            move = self.agent.get_move(self.game)
            self.game.do_action(move)

            if(self.game.dead or self.game_steps == self.max_game_steps):
                self.agent.get_move(self.game)
                break    

    def main(self):
        games = 0
        avgscore = 0
        new_game = True
        fourth = False
        half = False
        threefourths = False
        while (self.game_steps < self.max_game_steps):
            self.game_steps += 1
            move = self.agent.get_move(self.game)
            self.game.do_action(move)

            if(self.game.dead or self.game_steps == self.max_game_steps):
                avgscore += self.game.score
                games += 1
                self.agent.get_move(self.game)
                self.game.reset(warm_start=self.warm_start)
                new_game = True

            if self.game_steps%100 == 0:
                self.agent.train()


            if games%10 == 0 and games != 0 and new_game:
                new_game = False
                print("Over the last 10 games I've got an average score of", avgscore/10, "Played in total", games, "games")
                avgscore = 0
            
            
            if self.game_steps > self.max_game_steps/4 and not fourth:
                print("One fourth is done. Increasing the spawn range of the snake, this will make it harder.")
                self.game.range = 2
                fourth = True
                self.warm_start = 2
            
            if self.game_steps > self.max_game_steps/2 and not half:
                print("Half is done. Increasing the spawn range of the snake, this will make it harder.")
                self.game.range = 3
                half = True
                self.warm_start = 3
            
            if self.game_steps > (3*self.max_game_steps)/4 and not threefourths:
                print("Three fourths are done. Increasing the spawn range of the snake, this will make it harder.")
                self.game.range = 4
                #Why was this here?
                self.warm_start = 0
                threefourths = True


        input("Ready? ")
        self.agent.testing = True
        for _ in range(1000):
            if self.graphics is None:
                self.graphics = Graphics(self.game.mapsize)
            self.run()
            self.game.reset(warm_start=self.warm_start)

        torch.save(self.agent.trainer.model, 'model_'+ datetime.now().strftime("%m_%d_%Y%H_%M_%S"))
        torch.save(self.agent.trainer.model, 'previous_model')

trainer = Trainer(load_model = False, load_warmstart_model = True)
trainer.main()