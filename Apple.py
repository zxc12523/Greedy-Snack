import random

import pygame


def get_random_position(window_width, window_height):
    random_x = random.sample([i for i in range(10, window_width+10, 20)], k=1)[0]
    random_y = random.sample([i for i in range(10, window_height+10, 20)], k=1)[0]

    return [random_x, random_y]


class Apple(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.width = 20
        self.pos = [210, 210]



