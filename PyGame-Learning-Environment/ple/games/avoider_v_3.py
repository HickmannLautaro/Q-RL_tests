import sys
import pygame
from .utils import percent_round_int

from ple.games import base
from pygame.constants import K_a, K_d
import numpy as np


class Paddle(pygame.sprite.Sprite):

    def __init__(self, width, height, SCREEN_WIDTH, SCREEN_HEIGHT):
        self.width = width
        self.height = height

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

    def __init__(self, speed_min, speed_max, speed_step, speed_div, size, SCREEN_WIDTH, SCREEN_HEIGHT, rng):
        self.speed = 0
        self.size = size
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.speed_step = speed_step
        self.speed_div = speed_div

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

        # if y > 0:
        #     n_y = y + self.SCREEN_HEIGHT // 4
        # else:
        n_y = y + self.speed * dt

        # if n_y > self.SCREEN_HEIGHT and y != self.SCREEN_HEIGHT - self.size:
        #     n_y = self.SCREEN_HEIGHT - self.size

        self.rect.center = (x, n_y)

    def reset(self):
        self.speed = self.rng.choice(range(self.speed_min, self.speed_max, self.speed_step)) / self.speed_div

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

        y = self.rng.choice(
            range(
                self.size,
                int(self.SCREEN_HEIGHT / 2),
                self.size))

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


class Avoider_v_3(base.PyGameWrapper):
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

    obstacles : int (default: 2)
        Number of possible obstacles

    """

    def __init__(self, width=64, height=64, init_lives=3, obstacles=3):

        actions = {
            "left": K_a,
            "right": K_d
        }

        base.PyGameWrapper.__init__(self, width, height, actions=actions)

        self.stone_size = percent_round_int(height, 0.06)
        self.speed_min = 95 * height
        self.speed_max = 950 * height
        self.speed_step = (self.speed_max - self.speed_min) // 5
        self.speed_div = 100000

        self.player_speed = 0.021 * width
        self.paddle_width = percent_round_int(width, 0.3)  # 0.2
        self.paddle_height = percent_round_int(height, 0.04)

        self.dx = 0.0
        self.init_lives = init_lives

        self.obstacles = obstacles
        self.list_of_obstacles = []
        self.list_avoided = []
        self.old_score = 0

        self.rewards = {
            "positive": 1.0,
            "negative": -5.0,
            "tick": 0,
            "loss": -10.0,
            "win": 5.0
        }

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
        self.old_score = self.score
        self.score = 0
        self.lives = self.init_lives

        self.player = Paddle(self.paddle_width,
                             self.paddle_height, self.width, self.height)

        self.list_of_obstacles = []
        self.list_avoided = []
        for obstacle in range(self.obstacles):
            self.list_of_obstacles.append(Stone(self.speed_min, self.speed_max, self.speed_step, self.speed_div, self.stone_size, self.width, self.height, self.rng))
            self.list_avoided.append(False)

        for obstacle in self.list_of_obstacles:
            obstacle.reset()

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

        obstacles_x = []
        obstacles_y = []
        obstacles_vel = []
        for obstacle in self.list_of_obstacles:
            obstacles_x.append(obstacle.rect.center[0] + obstacle.size // 2)
            obstacles_y.append(np.maximum(1, obstacle.rect.center[1]))
            obstacles_vel.append(obstacle.speed)

        # Filtering
        player_x = np.clip(player_x - 255, -1, 1)

        obstacles_x = [np.clip(obs - 255, -1, 1) for obs in obstacles_x]
        obstacles_y = [(481 - obs) / 481 for obs in obstacles_y]

        env_obs = np.ones((3, 2)) * -1

        for i in range(len(obstacles_x)):
            aux_1 = env_obs[obstacles_x[i] + 1, 0]
            aux_2 = obstacles_y[i]
            if env_obs[obstacles_x[i] + 1, 0] == -1 or env_obs[obstacles_x[i] + 1, 0] < 0 or env_obs[obstacles_x[i] + 1, 0] > obstacles_y[i]:
                if obstacles_y[i] < 0:
                    env_obs[obstacles_x[i] + 1, 0] = - 1
                    env_obs[obstacles_x[i] + 1, 1] = -1
                else:
                    env_obs[obstacles_x[i] + 1, 0] = obstacles_y[i]
                    env_obs[obstacles_x[i] + 1, 1] = obstacles_vel[i] / (self.speed_max / self.speed_div)

        env_obs = env_obs.flatten()
        state = {
            "player_x": player_x,
            "line_1_y": env_obs[0],
            "line_1_vel": env_obs[1],
            "line_2_y": env_obs[2],
            "line_2_vel": env_obs[3],
            "line_3_y": env_obs[4],
            "line_3_vel": env_obs[5],
        }

        return state

    def draw_epoch_custom(self, epoch):

        pygame.font.init()  # you have to call this at the start,
        # if you want to use this module.
        myfont = pygame.font.Font(None, 30)
        textsurface1 = myfont.render(f"Epoch {epoch}", False, (255, 255, 255))
        textsurface2 = myfont.render(f"previous epoch score {self.old_score}", False, (255, 255, 255))

        text_rect = textsurface1.get_rect(center=(self.width / 2, self.height / 2))
        text_rect2 = textsurface2.get_rect(center=(self.width / 2, self.height / 2))

        self.screen.fill((0, 0, 0))

        self.screen.blit(textsurface1, text_rect)
        self.screen.blit(textsurface2, (text_rect2[0],text_rect2[1]+25))


    def getScore(self):
        return self.score

    def game_over(self):
        return self.lives <= 0

    def step(self, dt):
        sizes = []
        # for obstacle in self.list_of_obstacles:
        #     sizes.append(obstacle.size)
        # 486400
        # 152900

        self.speed_max = (self.player.height + self.stone_size) / dt - 0.001  # max_speed = (agent_height + obstacle_height)/dt - 1


        self.screen.fill((0, 0, 0))
        self._handle_player_events()

        self.score += self.rewards["tick"]

        for obstacle in self.list_of_obstacles:
            if obstacle.speed > self.speed_max:
                obstacle.speed = self.speed_max

            if pygame.sprite.collide_rect(self.player, obstacle):
                self.score += self.rewards["negative"]
                self.lives -= 1
                obstacle.draw_collision(self.screen)

                for obstacle in self.list_of_obstacles:
                    obstacle.reset()
                break

            if self.lives == 0:
                self.score += self.rewards["loss"]

        for i in range(len(self.list_of_obstacles)):  # , avoided in zip(self.list_of_obstacles, self.list_avoided):
            if self.list_of_obstacles[i].rect.center[1] >= self.height and not self.list_avoided[i]:
                self.score += self.rewards["positive"]
                self.list_avoided[i] = True

            self.list_of_obstacles[i].update(dt)

        if all(x for x in self.list_avoided):
            self.lives -= 1
            # for i in range(len(self.list_of_obstacles)):
            #     self.list_of_obstacles[i].reset()
            #     self.list_avoided[i] = False

        self.player.update(self.dx, dt)

        # if self.lives == 0:
        #     self.score += self.rewards["loss"]

        buckets = self.width // 3
        pygame.draw.line(self.screen, (255, 255, 255), (buckets, 0), (buckets, self.height), 1)
        pygame.draw.line(self.screen, (255, 255, 255), (buckets * 2, 0), (buckets * 2, self.height), 1)

        self.player.draw(self.screen)

        for obstacle in self.list_of_obstacles:
            obstacle.draw(self.screen)
        self.speed_max = int(self.speed_max * self.speed_div)
