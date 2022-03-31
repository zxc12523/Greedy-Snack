import math
import sys
import random
import tensorflow as tf
import pygame

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400


def detect_collision(snack):
    return snack.head_pos in snack.body


def detect_eat(snack, apple):
    return snack.head_pos == apple.pos


def get_random_position(window_width, window_height):
    random_x = random.sample([i for i in range(10, window_width+10, 20)], k=1)[0]
    random_y = random.sample([i for i in range(10, window_height+10, 20)], k=1)[0]

    return [random_x, random_y]


def update_snack(eat, snack):
    if not eat:
        snack.body.pop()
        snack.body.insert(0, [snack.head_pos[0], snack.head_pos[1]])
    else:
        snack.body.insert(0, [snack.head_pos[0], snack.head_pos[1]])


def update_apple(snack, apple):
    while apple.pos in snack.body:
        apple.pos = get_random_position(WINDOW_WIDTH, WINDOW_HEIGHT)


def update_block(x, y, width, color, state):
    color = tf.convert_to_tensor(color)
    w = int(width / 2)
    for i in range(x-w, x+w):
        for j in range(y-w, y+w):
            state[0][i][j] = color

    return state


def ai_step(ai_snack, apple, state, action):
    ai_snack.move(action)
    reward = 0
    done = False
    if detect_collision(ai_snack):
        reward = -100
        done = True
        return state, reward, done

    if detect_eat(ai_snack, apple):
        reward = 100
        update_snack(1, ai_snack)
        state = update_block(ai_snack.head_pos[0], ai_snack.head_pos[1], ai_snack.width, [0, 255, 0], state)
        state = update_block(ai_snack.body[1][0], ai_snack.body[1][1], ai_snack.width, [0, 0, 255], state)
        update_apple(ai_snack, apple)
        state = update_block(apple.pos[0], apple.pos[1], apple.width, [255, 0, 0], state)
    else:
        state = update_block(ai_snack.body[len(ai_snack.body)-1][0], ai_snack.body[len(ai_snack.body)-1][1], ai_snack.width, [0, 0, 0], state)
        update_snack(0, ai_snack)
        state = update_block(ai_snack.head_pos[0], ai_snack.head_pos[1], ai_snack.width, [0, 255, 0], state)
        state = update_block(ai_snack.body[1][0], ai_snack.body[1][1], ai_snack.width, [0, 0, 255], state)
        reward = math.sqrt((apple.pos[0] - ai_snack.head_pos[0])**2 + (apple.pos[1] - ai_snack.head_pos[1])**2)
        reward /= -1000

    return state, reward, done


def update(window, snack, apple):
    if detect_collision(snack):
        print("loss")
        pygame.image.save(window, "image.jpg")
        sys.exit()

    if detect_eat(snack, apple):
        update_snack(1, snack)
        update_apple(snack, apple)
    else:
        update_snack(0, snack)


def reset(ai_snack, apple, state):
    state = update_block(apple.pos[0], apple.pos[1], apple.width, [255, 0, 0], state)
    for i in ai_snack.body:
        state = update_block(i[0], i[1], ai_snack.width, [0, 0, 255], state)
    state = update_block(ai_snack.head_pos[0], ai_snack.head_pos[1], ai_snack.width, [0, 255, 0], state)
    return state