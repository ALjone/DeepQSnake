from agent import DQAgent
import torch
from game import Game
from datetime import datetime
import time

class Trainer:
    def __init__(self, load_warmstart_model: bool = False, load_model: bool = False) -> None:
        # params
        size: int = 10
        lifespan: int = 50
        memory_bank_size = 10000
        self.max_episodes: int = 20000
        self.episodes = 0
        self.game: Game = Game(size, lifespan)
        
        if load_warmstart_model:
            print("Loading warm-start model")
            self.agent = DQAgent(self.max_episodes, bank_size = memory_bank_size, load_path="warm_start_model")
        else:
            self.agent = DQAgent(self.max_episodes, bank_size = memory_bank_size)
            
        if load_model:
            self.agent.trainer.model = torch.load("previous_model")

        self.moves = [0, 0, 0, 0]
    
    def run(self):
        reward = 0
        while(True):
            if self.graphics is not None: 
                self.graphics.updateWin(self.game, reward)
            
            model_made, move = self.agent.get_move(self.game)
            self.game.do_action(move)
            reward += self.agent._get_reward(self.game)

            if(self.game.final_state):
                model_made, move = self.agent.get_move(self.game)
                self.graphics.updateWin(self.game, reward)
                break

    def play_episode(self):
        while(True):
            model_made, move = self.agent.get_move(self.game)

            self.agent.make_memory(self.game, move)
            self.game.do_action(move)

            if model_made: self.moves[move] += 1

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


            if self.episodes%1000 == 0 and self.episodes != 0:
                printlist = [round(num*self.agent._exploration_rate(), 0) for num in self.moves]
                print("Over the last 1000 games I've got an average score of", avgscore/1000, "Played in total", self.episodes, "games.",
                "Last 1000 games the model made moves were:", printlist)
                print("Current exploration rate:", round(self.agent._exploration_rate(), 2))    
                self.moves = [0, 0, 0, 0]
                avgscore = 0


        torch.save(self.agent.trainer.model, 'model_'+ datetime.now().strftime("%m_%d_%Y%H_%M_%S"))
        torch.save(self.agent.trainer.model, 'previous_model')

        input("Ready? ")
        self.agent.testing = True
        from graphics_module import Graphics
        self.graphics: Graphics = None
        for _ in range(100):
            if self.graphics is None:
                self.graphics = Graphics(self.game.mapsize)
            self.run()
            self.game.reset()

trainer = Trainer(load_model = False, load_warmstart_model = False)
trainer.main()

pos = 0
neg = 0
for memory in trainer.agent.bank.memory:
    if memory.reward == 1.0:
        pos += 1
    if memory.reward == -1.0:
        neg += 1

print("Percentage of apple memories to non apple memories:", (pos/(10000))*100)
print("Percentage of death memories to non death memories:", (neg/(10000))*100)