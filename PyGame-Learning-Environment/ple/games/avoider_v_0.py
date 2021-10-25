import sys
import pygame
from .utils import percent_round_int

from ple.games import base
from pygame.constants import K_a, K_d
import numpy as np


class Paddle(pygame.sprite.Sprite):

    def __init__(self, width, height, SCREEN_WIDTH, SCREEN_HEIGHT):
        self.width = width

        self.SCREEN_WIDTH = SCREEN_WIDTH

        pygame.sprite.Sprite.__init__(self)

        image = pygame.Surface((width, height))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))

        pygame.draw.rect(
            image,
            (255, 255, 255),
            (0, 0, width, height),
            0
        )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = (
            SCREEN_WIDTH / 2 - self.width / 2,
            SCREEN_HEIGHT - height - 3)

    def update(self, dx, dt):

        x_init, y = self.rect.center
        x = x_init + self.width // 2

        buckets = self.SCREEN_WIDTH // 3
        if x <= buckets:  # lower third
            if dx > 0:
                x = buckets * 1.5  # move to middle third
        elif x >= buckets * 2:  # upper third
            if dx < 0:
                x = buckets * 1.5  # move to middle third
        else:  # middle third
            if dx > 0:  # move to upper third
                x = buckets * 2.5  # move to middle third
            elif dx < 0:  # move to lower third
                x = buckets * 0.5  # move to lower third

        self.rect.center = (x - self.width // 2, y)

    def draw(self, screen):
        screen.blit(self.image, self.rect.center)


class Stone(pygame.sprite.Sprite):

    def __init__(self, speed, size, SCREEN_WIDTH, SCREEN_HEIGHT, rng):
        self.speed = speed
        self.size = size

        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        self.rng = rng

        pygame.sprite.Sprite.__init__(self)

        image = pygame.Surface((size, size))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))

        # (255, 120, 120),
        pygame.draw.rect(
            image,
            (128, 128, 128),
            (0, 0, size, size),
            0
        )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = (-30, -30)

    def update(self, dt):
        x, y = self.rect.center

        if y > 0:
            n_y = y + self.SCREEN_HEIGHT // 4
        else:
            n_y = y + 1  # self.speed * dt

        if n_y > self.SCREEN_HEIGHT and y != self.SCREEN_HEIGHT - self.size:
            n_y = self.SCREEN_HEIGHT - self.size

        self.rect.center = (x, n_y)

    def reset(self):
        # x = self.rng.choice(
        #     range(
        #         self.size *
        #         2,
        #         self.SCREEN_WIDTH -
        #         self.size *
        #         2,
        #         self.size))

        buckets = self.SCREEN_WIDTH // 3

        x = self.rng.choice([buckets * 0.5, buckets * 1.5, buckets * 2.5]) - self.size // 2
        y = 0
        # y = self.rng.choice(
        #     range(
        #         self.size,
        #         int(self.SCREEN_HEIGHT / 2),
        #         self.size))

        self.rect.center = (x, -1 * y)

        pygame.draw.rect(
            self.image,
            (128, 128, 128),
            (0, 0, self.size, self.size),
            0
        )

    def draw(self, screen):
        screen.blit(self.image, self.rect.center)

    def draw_collision(self, screen):
        pygame.draw.rect(
            self.image,
            (255, 120, 120),  # (120, 255, 120)
            (0, 0, self.size, self.size),
            0
        )
        screen.blit(self.image, self.rect.center)


class Avoider_v_0(base.PyGameWrapper):
    """
    Based on `Eder Santana`_'s game idea.

    .. _`Eder Santana`: https://github.com/EderSantana

    Parameters
    ----------
    width : int
        Screen width.

    height : int
        Screen height, recommended to be same dimension as width.

    init_lives : int (default: 3)
        The number lives the agent has.

    """

    def __init__(self, width=64, height=64, init_lives=3):

        actions = {
            "left": K_a,
            "right": K_d
        }

        base.PyGameWrapper.__init__(self, width, height, actions=actions)

        self.fruit_size = percent_round_int(height, 0.06)
        self.fruit_fall_speed = 0.00095 * height

        self.player_speed = 0.021 * width
        self.paddle_width = percent_round_int(width, 0.3)  # 0.2
        self.paddle_height = percent_round_int(height, 0.04)

        self.dx = 0.0
        self.init_lives = init_lives

    def _handle_player_events(self):
        self.dx = 0.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                if key == self.actions['left']:
                    self.dx = -1  # -= self.player_speed

                if key == self.actions['right']:
                    self.dx = 1  # += self.player_speed

    def init(self):
        self.score = 0
        self.lives = self.init_lives

        self.player = Paddle(self.paddle_width,
                             self.paddle_height, self.width, self.height)

        self.stone_1 = Stone(self.fruit_fall_speed, self.fruit_size,
                             self.width, self.height, self.rng)
        self.stone_2 = Stone(self.fruit_fall_speed, self.fruit_size,
                             self.width, self.height, self.rng)
        # self.stone_3 = Stone(self.fruit_fall_speed, self.fruit_size,
        #                      self.width, self.height, self.rng)

        self.stone_1.reset()
        self.stone_2.reset()
        # self.stone_3.reset()

    def getGameState(self):
        """
        Gets a non-visual state representation of the game.

        Returns
        -------

        dict
            * player x position.
            * players velocity.
            * fruits x position.
            * fruits y position.

            See code for structure.

        """

        player_x = self.player.rect.center[0] + self.player.width // 2
        if player_x == 256:
            player_x = 255
        stone_1_x = self.stone_1.rect.center[0] + self.stone_1.size // 2
        stone_2_x = self.stone_2.rect.center[0] + self.stone_2.size // 2
        # stone_3_x = self.stone_3.rect.center[0] + self.stone_3.size//2

        stone_1_y = np.maximum(1, self.stone_1.rect.center[1])
        stone_2_y = np.maximum(1, self.stone_2.rect.center[1])
        # stone_3_y = np.maximum(1,self.stone_3.rect.center[1])

        # Filtering
        player_x = np.clip(player_x - 255, -1, 1)

        stone_1_x = np.clip(stone_1_x - 255, -1, 1)
        stone_2_x = np.clip(stone_2_x - 255, -1, 1)
        # stone_3_x = np.clip(stone_3_x - 255 , -1, 1)

        line_1 = line_2 = line_3 = -1

        stone_1_y = (481 - stone_1_y) / 481
        stone_2_y = (481 - stone_2_y) / 481


        if stone_1_x == -1:
            line_1 = stone_1_y
        if stone_1_x == 0:
            line_2 = stone_1_y
        if stone_1_x == 1:
            line_3 = stone_1_y

        if stone_2_x == -1:
            line_1 = stone_2_y
        if stone_2_x == 0:
            line_2 = stone_2_y
        if stone_2_x == 1:
            line_3 = stone_2_y

        state = {
            "player_x": player_x,
            "line_1": line_1,  # fruit_x,
            "line_2": line_2,  # fruit_y
            "line_3": line_3,
        }

        return state

    def getScore(self):
        return self.score

    def game_over(self):
        return self.lives <= 0

    def step(self, dt):
        self.screen.fill((0, 0, 0))
        self._handle_player_events()

        self.score += self.rewards["tick"]

        if self.stone_1.rect.center[1] >= self.height:
            self.stone_1.draw_collision(self.screen)
            self.score += self.rewards["positive"]
            self.stone_1.reset()
            self.stone_2.reset()

        if pygame.sprite.collide_rect(self.player, self.stone_1):
            self.score += self.rewards["negative"]
            self.lives -= 1
            self.stone_1.reset()
            self.stone_2.reset()

        if self.stone_1.rect.center[0] - self.stone_2.rect.center[0] > 0:
            if self.stone_2.rect.center[1] >= self.height:
                self.stone_2.draw_collision(self.screen)
                self.score += self.rewards["positive"]
                self.stone_2.reset()

            if pygame.sprite.collide_rect(self.player, self.stone_2):
                self.score += self.rewards["negative"]
                self.lives -= 1
                self.stone_2.reset()

        # if self.stone_1.rect.center[1] != self.stone_3.rect.center[1] and self.stone_2.rect.center[1] != self.stone_3.rect.center[1]:
        #
        #     if self.stone_3.rect.center[1] >= self.height:
        #         self.stone_3.draw_collision(self.screen)
        #         self.score += self.rewards["positive"]
        #         self.stone_3.reset()
        #
        #     if pygame.sprite.collide_rect(self.player, self.stone_3):
        #         self.score += self.rewards["negative"]
        #         self.lives -= 1
        #         self.stone_3.reset()

        self.stone_1.update(dt)
        self.stone_2.update(dt)
        # self.stone_3.update(dt)

        self.player.update(self.dx, dt)

        if self.lives == 0:
            self.score += self.rewards["loss"]

        buckets = self.width // 3
        pygame.draw.line(self.screen, (255, 255, 255), (buckets, 0), (buckets, self.height), 1)
        pygame.draw.line(self.screen, (255, 255, 255), (buckets * 2, 0), (buckets * 2, self.height), 1)

        self.player.draw(self.screen)
        self.stone_1.draw(self.screen)
        self.stone_2.draw(self.screen)
        # self.stone_3.draw(self.screen)
