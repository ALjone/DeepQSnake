import torch

class Game_map:
    def __init__(self, mapsize) -> None:
        self.mapsize = mapsize

        self.__game_map = torch.zeros(3, self.mapsize, self.mapsize)

    def has_tail(self, x, y):
        return self.__game_map[1, x, y] == 1

    def update(self, head, tail, apple_x, apple_y):
        if torch.sum(self.__game_map[2, :, :] > 1):
            print("More than one apple????")
        """Resets and updates the position of all the objectives on the map"""
        #Reset map
        self.__game_map: torch.Tensor = torch.zeros(3, self.mapsize, self.mapsize)

        #Add head
        self.__game_map[0, head.x_pos, head.y_pos] = 1

        #Add tail
        while (tail.next != None):
            self.__game_map[1, tail.x_pos, tail.y_pos] = 1
            tail = tail.next

        #Add apple
        if apple_x is not None and apple_y is not None:
            self.__game_map[2, apple_x, apple_y] = 1

    def get_map(self):
        return self.__game_map