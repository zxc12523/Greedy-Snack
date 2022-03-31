import sys
import Snack
import Apple
import Render
import Update
import cv2

import numpy as np
import pygame
from pygame.locals import *

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400
WHITE = (255, 255, 255)
IMAGEWIDTH = 40
IMAGEHEIGHT = 40
FPS = 60


snack = Snack.User_Snack()
apple = Apple.Apple()
ai = Snack.AI_Snack()

n = 2
if n == 1:
    pygame.init()
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))  # 設定視窗大小

    pygame.display.set_caption('Greed Snack')  # 設定標題
    while 1:
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()

        pressed_key = pygame.key.get_pressed()
        snack.move(pressed_key)
        Update.update(window, snack, apple)
        Render.draw(window, snack, apple)

elif n == 2:
    for episode in range(20000):
        apple = Apple.Apple()
        ai = Snack.AI_Snack()

        done = False
        # pygame.image.save(window, "image.jpg")
        state = np.zeros((1, 400, 400, 3))
        state = Update.reset(ai, apple, state)
        cv2.imwrite('./image folder/color_img' + str(00) + '.jpg', np.array(state)[0])

        running_reward = 0

        for i in range(2000):
            if not done:
                action = ai.policy(state)
                n_state, reward, done = Update.ai_step(
                    ai, apple, state, action)
                ai.store_gameplay_experience(
                    state, n_state, reward, action, done)
                state = n_state
                cv2.imwrite('./image folder/color_img' + str(i) + '.jpg', np.array(state)[0])
                running_reward += reward
                if i % 100 == 0:
                    loss = ai.train()

        print('Episode: {0}, reward: {1}, body_len: {2}'.format(episode+1, running_reward, len(ai.body)))
        ai.update_variables()
