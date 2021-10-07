import sys
import pygame
from .utils import percent_round_int

from ple.games import base
from pygame.constants import K_a, K_d
import numpy as np

class Paddle(pygame.sprite.Sprite):

    def __init__(self,  width, height, SCREEN_WIDTH, SCREEN_HEIGHT):
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
        x = x_init + self.width//2

        buckets = self.SCREEN_WIDTH//3
        if x <= buckets: #lower third
            if dx > 0:
                x = buckets*1.5 # move to middle third
        elif x>= buckets*2: #upper third
            if dx <0 :
                x = buckets*1.5 # move to middle third
        else: # middle third
            if dx > 0: # move to upper third
                x = buckets * 2.5  # move to middle third
            elif dx <0 : #move to lower third
                x = buckets * 0.5  # move to lower third


        self.rect.center = (x - self.width//2, y)

    def draw(self, screen):
        screen.blit(self.image, self.rect.center)


class Fruit(pygame.sprite.Sprite):

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
            (255, 120, 120),
            (0, 0, size, size),
            0
        )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = (-30, -30)

    def update(self, dt):
        x, y = self.rect.center

        if y>0:
            n_y = y + self.SCREEN_HEIGHT//4
        else:
            n_y = y + 1# self.speed * dt

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

        buckets = self.SCREEN_WIDTH//3

        x = self.rng.choice([buckets*0.5,buckets*1.5,buckets*2.5]) - self.size//2
        y = 0
        # y = self.rng.choice(
        #     range(
        #         self.size,
        #         int(self.SCREEN_HEIGHT / 2),
        #         self.size))

        self.rect.center = (x, -1 * y)

        pygame.draw.rect(
            self.image,
            (255, 120, 120),
            (0, 0, self.size, self.size),
            0
        )

    def draw(self, screen):
        screen.blit(self.image, self.rect.center)

    def draw_collision(self, screen):
        pygame.draw.rect(
            self.image,
            (120, 255, 120),
            (0, 0, self.size, self.size),
            0
        )
        screen.blit(self.image, self.rect.center)


class Catcher(base.PyGameWrapper):
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
        self.paddle_width = percent_round_int(width, 0.3) #0.2
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
                    self.dx = -1 # -= self.player_speed

                if key == self.actions['right']:
                    self.dx = 1 # += self.player_speed

    def init(self):
        self.score = 0
        self.lives = self.init_lives

        self.player = Paddle(self.paddle_width,
                             self.paddle_height, self.width, self.height)

        self.fruit = Fruit(self.fruit_fall_speed, self.fruit_size,
                           self.width, self.height, self.rng)

        self.fruit.reset()

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

        player_x = self.player.rect.center[0] + self.player.width//2
        if player_x == 256:
            player_x = 255
        fruit_x = self.fruit.rect.center[0] + self.fruit.size//2

        fruit_y = np.maximum(1,self.fruit.rect.center[1])
        # Filtering
        player_x = np.clip(player_x -255, -1, 1)
        fruit_x = np.clip(fruit_x - 255 , -1, 1)
        fruit_y = (481 - fruit_y)/481

        state = {
            "player_x": player_x,
            "fruit_x": fruit_x,
            "fruit_y": fruit_y
        }

        return state

    def getScore(self):
        return self.score

    def game_over(self):
        return self.lives == 0

    def step(self, dt):
        self.screen.fill((0, 0, 0))
        self._handle_player_events()

        self.score += self.rewards["tick"]

        if self.fruit.rect.center[1] >= self.height:
            self.score += self.rewards["negative"]
            self.lives -= 1
            self.fruit.reset()

        if pygame.sprite.collide_rect(self.player, self.fruit):
            self.fruit.draw_collision(self.screen)
            self.score += self.rewards["positive"]
            self.fruit.reset()

        self.player.update(self.dx, dt)
        self.fruit.update(dt)




        if self.lives == 0:
            self.score += self.rewards["loss"]

        buckets = self.width // 3
        pygame.draw.line(self.screen, (255,255,255), (buckets,0), (buckets,self.height), 1 )
        pygame.draw.line(self.screen, (255,255,255), (buckets*2,0), (buckets*2,self.height), 1 )

        self.player.draw(self.screen)
        self.fruit.draw(self.screen)

if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = Catcher(width=256, height=256)
    game.rng = np.random.RandomState(24)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.init()

    while True:
        dt = game.clock.tick_busy_loop(30)
        if game.game_over():
            game.reset()

        game.step(dt)
        pygame.display.update()
