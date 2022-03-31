import pygame

FPS = 60
main_clock = pygame.time.Clock()


def draw_snack(window, agent):
    for pos in agent.body:
        pygame.draw.rect(window, (255, 255, 255), [pos[0] - 10, pos[1] - 10, agent.width, agent.width], 0)


def draw_apple(window, apple):
    pygame.draw.circle(window, (255, 0, 0), apple.pos, 10)


def draw(window, agent, apple):
    # 背景顏色，清除畫面
    window.fill((0, 0, 0))

    # 畫蛇
    draw_snack(window, agent)
    # 畫蘋果
    draw_apple(window, apple)

    pygame.display.update()
    # 控制遊戲迴圈迭代速率
    main_clock.tick(FPS / 3)



