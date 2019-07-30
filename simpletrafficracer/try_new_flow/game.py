import pygame
import random
import sys
import os
from pygame.locals import *
from matplotlib.pyplot import imshow
from PIL import Image
from io import StringIO
from collections import deque
import cv2
import math
import resource
from enum import Enum

# set SDL to use the dummy NULL video driver,
#   so it doesn't need a windowing system.
#os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np


class FrontBack(Enum):
    FORWARD = 1
    NEUTRAL = 0
    BACKWARD = -1


class LeftRight(Enum):
    LEFT = 1
    NEUTRAL = 0
    RIGHT = -1


class GameDing:
    TOPSCORE_FILE = "data/save.dat"

    def __init__(self, debug_print=False):
        self.debug_print = debug_print
        # "Settings"
        self.WINDOWWIDTH = 800
        self.WINDOWHEIGHT = 600
        self.TEXTCOLOR = (255, 255, 255)
        self.BACKGROUNDCOLOR = (0, 0, 0)
        self.FPS = 20
        self.BADDIEMINSPEED = 8
        self.BADDIEMAXSPEED = 8
        self.ADDNEWBADDIERATE = 6
        self.PLAYERMOVERATE = 5
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.FINAL_EPSILON = 0.05  # final value of epsilon
        self.INITIAL_EPSILON = 1.0  # starting value of epsilon
        self.GAMMA = 0.99  # decay rate of past observations
        # Dimensions of our position board
        self.scale_down_factor = 10
        self.max_diff_x = math.floor(
            self.WINDOWHEIGHT / self.scale_down_factor)+self.scale_down_factor
        self.max_diff_y = math.floor(
            self.WINDOWWIDTH / self.scale_down_factor)+self.scale_down_factor
        # Images
        self.playerImage = pygame.image.load('image/car1.png')
        self.car3 = pygame.image.load('image/car3.png')
        self.car4 = pygame.image.load('image/car4.png')
        self.playerRect = self.playerImage.get_rect()
        self.baddieImage = pygame.image.load('image/car2.png')
        self.sample = [self.car3, self.car4, self.baddieImage]
        self.wallLeft = pygame.image.load('image/left.png')
        self.wallRight = pygame.image.load('image/right.png')
        # Score
        if not os.path.exists(self.TOPSCORE_FILE):
            f = open(self.TOPSCORE_FILE, 'w')
            f.write(str(0))
            f.close()
        v = open(self.TOPSCORE_FILE, 'r')
        self.topScore = int(v.readline())
        v.close()
        # set up pygame, the window, and the mouse cursor
        pygame.init()
        self.mainClock = pygame.time.Clock()
        self.windowSurface = pygame.display.set_mode(
            (self.WINDOWWIDTH, self.WINDOWHEIGHT))
        pygame.display.set_caption('car race')
        pygame.mouse.set_visible(False)
        # fonts
        self.font = pygame.font.SysFont(None, 30)
        # V
        self.t = 0
        # Set all resetable states
        self.reset()

    # Reset state for a new round + let it run a bit to init field
    def reset(self):
        self.baddies = []
        # model.save(f"models/model_t{t:08}.h5")
        self.score = 0
        left = self.WINDOWWIDTH / 2 - 120
        top = self.WINDOWHEIGHT - 90
        left_range = 80
        left_random = random.randint(-left_range, left_range)
        top_range = 40
        top_random = random.randint(-top_range, top_range)
        self.playerRect.topleft = (left + left_random, top + top_random)
        self.moveLeft = self.moveRight = self.moveUp = self.moveDown = False
        self.baddieAddCounter = 0
        self.terminal = False
        self.observation = {
            "mini_image": None,
            "discrete_image": None,
            "own_pos": None,
            "action": (LeftRight.NEUTRAL, FrontBack.NEUTRAL),
            "dead": False,
            "score": 0,
        }
        self.position_board = np.zeros(
            (self.max_diff_x*2, self.max_diff_y*2), np.bool)
        self.position_list = np.zeros((40,2), dtype=np.int32)
        for _ in range(7):
            self.next_round(LeftRight.NEUTRAL, FrontBack.NEUTRAL)

    # game termination
    def _terminate(self):
        pygame.quit()
        sys.exit()

    # check collision
    def _playerHasHitBaddie(self):
        for b in self.baddies:
            if self.playerRect.colliderect(b['rect']):
                return True
        return False

    def _drawText(self, text, font, surface, x, y):
        textobj = font.render(text, 1, self.TEXTCOLOR)
        textrect = textobj.get_rect()
        textrect.topleft = (x, y)
        surface.blit(textobj, textrect)

    # one step forward
    def next_round(self, direction: LeftRight, gas: FrontBack, realtime=False):
        self.score += 1  # increase score
        if realtime:
            self.mainClock.tick(self.FPS)
        else:
            self.mainClock.tick()

        if direction == LeftRight.LEFT:
            self.moveLeft = True
            self.moveRight = False
        elif direction == LeftRight.RIGHT:
            self.moveLeft = False
            self.moveRight = True
        elif direction == LeftRight.NEUTRAL:
            self.moveLeft = False
            self.moveRight = False
        else:
            raise ValueError()

        if gas == FrontBack.FORWARD:
            self.moveUp = True
            self.moveDown = False
        elif gas == FrontBack.BACKWARD:
            self.moveUp = False
            self.moveDown = True
        elif gas == FrontBack.NEUTRAL:
            self.moveUp = False
            self.moveDown = False
        else:
            raise ValueError()

        for event in pygame.event.get():
            if event.type == QUIT:
                self._terminate()
            if event.type == KEYUP:
                if event.key == K_ESCAPE:
                    self._terminate()

        # Add new baddies at the top of the screen
        self.baddieAddCounter += 1
        if self.baddieAddCounter == self.ADDNEWBADDIERATE:
            self.baddieAddCounter = 0
            baddieSize = 30
            newBaddie = {'rect': pygame.Rect(random.randint(140, 485), 0 - baddieSize, 23, 47),
                         'speed': random.randint(self.BADDIEMINSPEED, self.BADDIEMAXSPEED),
                         'surface': pygame.transform.scale(random.choice(self.sample), (23, 47)),
                         }
            self.baddies.append(newBaddie)
            sideLeft = {'rect': pygame.Rect(0, 0, 126, 600),
                        'speed': random.randint(self.BADDIEMINSPEED, self.BADDIEMAXSPEED),
                        'surface': pygame.transform.scale(self.wallLeft, (126, 599)),
                        }
            self.baddies.append(sideLeft)
            sideRight = {'rect': pygame.Rect(497, 0, 303, 600),
                         'speed': random.randint(self.BADDIEMINSPEED, self.BADDIEMAXSPEED),
                         'surface': pygame.transform.scale(self.wallRight, (303, 599)),
                         }
            self.baddies.append(sideRight)

        # Move the player around.
        if self.moveLeft and self.playerRect.left > 0:
            self.playerRect.move_ip(-1 * self.PLAYERMOVERATE, 0)
        if self.moveRight and self.playerRect.right < self.WINDOWWIDTH:
            self.playerRect.move_ip(self.PLAYERMOVERATE, 0)
        if self.moveUp and self.playerRect.top > 0:
            self.playerRect.move_ip(0, -1 * self.PLAYERMOVERATE)
        if self.moveDown and self.playerRect.bottom < self.WINDOWHEIGHT:
            self.playerRect.move_ip(0, self.PLAYERMOVERATE)

        self.position_board.fill(False)
        self.position_list.fill(0)
        for i,b in enumerate(self.baddies):
            b['rect'].move_ip(0, b['speed'])
            relative_position = (
                self.playerRect.center[0] - b["rect"].center[0], self.playerRect.center[1] - b["rect"].center[1])
            self.position_list[i] = relative_position
            relative_position = (math.floor(
                relative_position[0]/self.scale_down_factor), math.floor(relative_position[1]/self.scale_down_factor))
            if abs(relative_position[0]) > self.max_diff_x:
                self.max_diff_x = abs(relative_position[0])
                print("New max diff:", self.max_diff_x, self.max_diff_y)
                sys.exit(1)
            if abs(relative_position[1]) > self.max_diff_y:
                self.max_diff_y = abs(relative_position[1])
                print("New max diff:", self.max_diff_x, self.max_diff_y)
                sys.exit(1)
            self.position_board[relative_position[0] + self.max_diff_x,
                                relative_position[1] + self.max_diff_y] = True

        for b in self.baddies[:]:
            if b['rect'].top > self.WINDOWHEIGHT:
                self.baddies.remove(b)

        # Draw the game world on the window.
        self.windowSurface.fill(self.BACKGROUNDCOLOR)

        # Draw the score and top score.
        self._drawText('Score: %s' % (self.score),
                       self.font, self.windowSurface, 128, 0)
        self._drawText('Top Score: %s' % (self.topScore),
                       self.font, self.windowSurface, 128, 20)
        #drawText('Rest Life: %s' % (count), font, windowSurface, 128, 40)

        self.windowSurface.blit(self.playerImage, self.playerRect)

        # data = pygame.image.tostring(windowSurface, 'RGBA')
        # img = Image.frombytes('RGBA', (100, 200), data)
        # imshow(data)
        for b in self.baddies:
            self.windowSurface.blit(b['surface'], b['rect'])

        pygame.display.update()

        # Check if any of the car have hit the player.
        if self._playerHasHitBaddie():
            self.terminal = True
            if self.score > self.topScore:
                g = open(self.TOPSCORE_FILE, 'w')
                g.write(str(self.score))
                g.close()
                self.topScore = self.score
        else:
            self.terminal = False

        own_pos = (self.playerRect.x, self.playerRect.y)

        # Divide with max value. Bool ==> 1
        im = 255 * (self.position_board / 1)
        w, h = im.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
        surf = pygame.surfarray.make_surface(ret)
        self.windowSurface.blit(surf, (0, 0))
        pygame.display.update()

        if self.debug_print:
            print("own pos:", own_pos, "Len:", len(self.baddies),
                "terminal:", self.terminal, "score:", self.score)
        if self.score != 0:
            self.observation = {
                "mini_image": im,
                "discrete_image": np.array(self.position_board, dtype=np.uint8),
                "own_pos": own_pos,
                "action": (direction, gas),
                "dead": self.terminal,
                "score": self.score,
                "position_vectors": self.position_list
            }

        self.t += 1

        if len(self.baddies) >= 40:
            print("More then 40 other cars:", len(self.baddies))

        return self.terminal

    def get_last_memory_image(self):
        return self.observation["mini_image"]
    
    def get_last_logic_state(self):
        return self.observation["discrete_image"]
