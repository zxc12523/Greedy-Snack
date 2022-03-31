import random
from collections import deque
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, InputLayer, BatchNormalization
import pygame
from pygame.locals import *

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400


class User_Snack(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.head_pos = [110, 210]
        self.dir = 0
        self.body = [[self.head_pos[0] - i, self.head_pos[1]] for i in range(0, 40, 20)]
        self.width = 20

    def move(self, pressed_key):
        b = pressed_key[K_LEFT] + pressed_key[K_RIGHT] + pressed_key[K_UP] + pressed_key[K_DOWN]
        l = pressed_key[K_LEFT] and self.dir == 0
        r = pressed_key[K_RIGHT] and self.dir == 2
        u = pressed_key[K_UP] and self.dir == 3
        d = pressed_key[K_DOWN] and self.dir == 1
        if b == 0 or b > 1 or l or r or u or d:
            if self.dir == 2:
                self.head_pos[0] = (self.head_pos[0] - self.width + WINDOW_WIDTH) % WINDOW_WIDTH
            if self.dir == 0:
                self.head_pos[0] = (self.head_pos[0] + self.width + WINDOW_WIDTH) % WINDOW_WIDTH
            if self.dir == 1:
                self.head_pos[1] = (self.head_pos[1] - self.width + WINDOW_HEIGHT) % WINDOW_HEIGHT
            if self.dir == 3:
                self.head_pos[1] = (self.head_pos[1] + self.width + WINDOW_HEIGHT) % WINDOW_HEIGHT
        else:
            if pressed_key[K_LEFT]:
                self.dir = 2
                self.head_pos[0] = (self.head_pos[0] - self.width + WINDOW_WIDTH) % WINDOW_WIDTH
            if pressed_key[K_RIGHT]:
                self.dir = 0
                self.head_pos[0] = (self.head_pos[0] + self.width + WINDOW_WIDTH) % WINDOW_WIDTH
            if pressed_key[K_UP]:
                self.dir = 1
                self.head_pos[1] = (self.head_pos[1] - self.width + WINDOW_HEIGHT) % WINDOW_HEIGHT
            if pressed_key[K_DOWN]:
                self.dir = 3
                self.head_pos[1] = (self.head_pos[1] + self.width + WINDOW_HEIGHT) % WINDOW_HEIGHT


class AI_Snack(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.head_pos = [110, 210]
        self.dir = 0
        self.body = [[self.head_pos[0] - i, self.head_pos[1]] for i in range(0, 80, 20)]
        self.width = 20
        self.q_network = self.build_model()
        self.q_target = self.build_model()
        self.gameplay_experience = deque(maxlen=2000)
        self.batch_size = 64
        self.gamma = 0.9
        self.epsilon = 0.2
        self.decrease = 0.00001

    @staticmethod
    def build_model():
        model = tf.keras.Sequential([InputLayer(input_shape=(WINDOW_WIDTH, WINDOW_HEIGHT, 3)),
                                     BatchNormalization(),
                                     Conv2D(filters=64,
                                            kernel_size=20, strides=20, activation='relu'),
                                     MaxPool2D(),
                                     Flatten(),
                                     Dense(512, activation='relu'),
                                     Dropout(0.2),
                                     Dense(512, activation='relu'),
                                     Dropout(0.2),
                                     Dense(512, activation='relu'),
                                     Dropout(0.2),
                                     Dense(4, activation='linear')
                                     ])
        # model.summary()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    @staticmethod
    def input_preprocessing(img):
        img = np.array(cv2.imread(img) / 255)
        img = np.expand_dims(img, axis=0)
        return img

    def policy(self, state_input):
        if np.random.random() < self.epsilon:
            self.epsilon -= self.decrease
            return random.randint(0, 3)

        q_action = self.q_network(state_input).numpy()[0]
        action = np.argmax(q_action)
        return action

    def store_gameplay_experience(self, state, n_state, reward, action, done):
        self.gameplay_experience.append([state[0], n_state[0], reward, action, done])

    def sample_gameplay_batch(self):
        batch = random.sample(self.gameplay_experience, k=self.batch_size) \
            if len(self.gameplay_experience) > self.batch_size \
            else self.gameplay_experience

        batch = np.array(batch).transpose()

        return (np.array([batch[i][j] for j in range(len(batch[0]))]) for i in range(5))

    def train(self):
        batch = self.sample_gameplay_batch()
        state_batch, n_state_batch, reward_batch, action_natch, done_batch = batch
        current_action_q = self.q_network(state_batch).numpy()
        n_state_action_q = self.q_network(n_state_batch).numpy()
        # print(current_action_q, n_state_action_q)
        q1_action = np.argmax(n_state_action_q, axis=1)
        n_state_action_q = self.q_target(n_state_batch).numpy()
        for i in range(len(state_batch)):
            if done_batch[i]:
                current_action_q[i][action_natch[i]] = reward_batch[i]
            else:
                # print(current_action_q, action_natch[i], reward_batch[i])
                # print(n_state_action_q, q1_action[i])
                current_action_q[i][action_natch[i]] = reward_batch[i] + self.gamma * n_state_action_q[i][q1_action[i]]
        history = self.q_network.fit(state_batch, current_action_q, verbose=0)
        loss = history.history['loss']
        return loss[0]

    def update_variables(self):
        self.q_target.set_weights(self.q_network.get_weights())

    def move(self, action):
        l = action == 2 and self.dir == 0
        r = action == 0 and self.dir == 2
        u = action == 1 and self.dir == 3
        d = action == 3 and self.dir == 1
        if l or r or u or d:
            if self.dir == 2:
                self.head_pos[0] = (self.head_pos[0] - self.width + WINDOW_WIDTH) % WINDOW_WIDTH
            if self.dir == 0:
                self.head_pos[0] = (self.head_pos[0] + self.width + WINDOW_WIDTH) % WINDOW_WIDTH
            if self.dir == 1:
                self.head_pos[1] = (self.head_pos[1] - self.width + WINDOW_HEIGHT) % WINDOW_HEIGHT
            if self.dir == 3:
                self.head_pos[1] = (self.head_pos[1] + self.width + WINDOW_HEIGHT) % WINDOW_HEIGHT
        else:
            if action == 2:
                self.dir = 2
                self.head_pos[0] = (self.head_pos[0] - self.width + WINDOW_WIDTH) % WINDOW_WIDTH
            if action == 0:
                self.dir = 0
                self.head_pos[0] = (self.head_pos[0] + self.width + WINDOW_WIDTH) % WINDOW_WIDTH
            if action == 1:
                self.dir = 1
                self.head_pos[1] = (self.head_pos[1] - self.width + WINDOW_HEIGHT) % WINDOW_HEIGHT
            if action == 3:
                self.dir = 3
                self.head_pos[1] = (self.head_pos[1] + self.width + WINDOW_HEIGHT) % WINDOW_HEIGHT

