import pygame
import random
import math

# from pygame import mixer

pygame.init()

screen = pygame.display.set_mode((800, 600))

pygame.display.set_caption("Captain Nowhere")

running = True

icon = pygame.image.load("okay.jpg")

# background sound
# mixer.music.load("whatever.wav")
# mixer.music.play(-1)

player_img = pygame.image.load("superhero.png")
player_X = 370
player_Y = 480
pxc = 0

noe = 5
enemy_img = []
enemy_X = []
enemy_Y = []
exc = []
for i in range(noe):
    enemy_img.append(pygame.image.load("monster.png"))
    enemy_X.append(random.randint(0, 735))
    enemy_Y.append(random.randint(40, 140))
    exc.append(0.3)

sword = pygame.image.load("sword.png")
sword_X = 0
sword_Y = 480
sxc = 0
syc = .3
sword_state = "ready"

score_value = 0
font = pygame.font.Font("am.ttf", 32)
text_X = 10
text_Y = 10

over_font = pygame.font.Font('freesansbold.ttf', 64)


def show_score(x, y):
    score = font.render("SCORE : " + str(score_value), True, (255, 255, 255))
    screen.blit(score, (x, y))


def show_game_over():
    game_over = over_font.render("GAME OVER", True, (255, 255, 255))
    screen.blit(game_over, (200,250))


def player(x, y):
    player = screen.blit(player_img, (x, y))


def enemy(x, y, i):
    enemy = screen.blit(enemy_img[i], (x, y))


def throw(x, y):
    global sword_state
    sword_state = "fire"
    screen.blit(sword, (x + 16, y + 10))


def collision(x1, y1, x2, y2):
    distance = math.sqrt((math.pow(x1 - x2, 2)) + (math.pow(y1 - y2, 2)))
    if distance < 27:
        return True
    else:
        return False


pygame.display.set_icon(icon)
while running:
    screen.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                pxc = -0.4
            if event.key == pygame.K_d:
                pxc = 0.4
            if event.key == pygame.K_SPACE:
                if sword_state == "ready":
                    # sword_sound = mixer.sound()
                    # sword_sound.play()
                    sword_X = player_X
                    throw(sword_X, sword_Y)

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_a or event.key == pygame.K_d:
                pxc = 0

    for i in range(noe):
        if enemy_Y[i] > 440:
            for j in range(noe):
                enemy_Y[j] = 2000
            show_game_over()
            break

        enemy(enemy_X[i], enemy_Y[i], i)

        is_collision = collision(enemy_X[i], sword_Y, sword_X, enemy_Y[i])
        if is_collision:
            # kill = mixer.sound()
            # kill.play()
            sword_state = "ready"
            sword_Y = 480
            score_value += 1
            enemy_X[i] = random.randint(0, 735)
            enemy_Y[i] = random.randint(40, 140)

        if enemy_X[i] >= 736:
            exc[i] = -.3
            enemy_Y[i] += 40
        elif enemy_X[i] <= 0:
            exc[i] = .3
            enemy_Y[i] += 40
        if enemy_Y[i] >= 416:
            enemy_Y[i] = 480
        enemy_X[i] += exc[i]

    if player_X <= 0:
        player_X = 0
    elif player_X >= 736:
        player_X = 736
    player_X += pxc

    if sword_Y <= 0:
        sword_Y = 480
        sword_state = "ready"
    if sword_state == "fire":
        throw(sword_X, sword_Y)
        sword_Y -= syc
    show_score(text_X, text_Y)
    player(player_X, player_Y)
    pygame.display.update()
