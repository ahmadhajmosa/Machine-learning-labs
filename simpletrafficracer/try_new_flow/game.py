import pygame, random, sys ,os,time
from pygame.locals import *
from matplotlib.pyplot import imshow
from PIL import Image
from io import StringIO
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from collections import deque
import tensorflow as tf
import cv2
import math
import resource
from enum import Enum

# set SDL to use the dummy NULL video driver, 
#   so it doesn't need a windowing system.
#os.environ["SDL_VIDEODRIVER"] = "dummy"


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, \
    Conv2D, MaxPooling2D, BatchNormalization
from tensorflow import keras
import numpy as np



# reinforce
#def replay(batch_size):
#        minibatch = random.sample(memory, batch_size)
#        for state, own_pos, action, done, points in minibatch:
#            target = points
#            reshaped_state = np.reshape(state, (1, max_diff_x*2, max_diff_y*2, 1))
#            target_f = model.predict(reshaped_state)
#            target_f[0][action] = target
#            model.fit(reshaped_state, target_f, epochs=1, verbose=0)
#
#def calc_points(state, done):
#    target = 0
#    found_points = 0
#    objects = np.nonzero(state)
#    for i in range(len(objects[0])):
#        idx_x = objects[0][i]
#        idx_y = objects[1][i]
#        current = state[idx_x, idx_y]
#        if current == True:
#            x = idx_x-max_diff_x
#            y = idx_y-max_diff_y
#            distance = (np.sqrt(x**2+y**2)-1)/10
#            print(distance)
#            target -= 1/distance
#            found_points += 1
#    if done == True:
#        target = -2
#        return target
#    if found_points == 0:
#        target = 0
#        return target
#    target /= found_points
#    return target

# build convnet
#def build_model():
#    # Neural Net for Deep-Q learning Model
#
#    #np.random.seed(1000)
#
#    # (2) Get Data
#
#    # (3) Create a sequential model
#    model = Sequential()
#
#    # 1st Convolutional Layer
#    model.add(Conv2D(filters=32, input_shape=(max_diff_x*2, max_diff_y*2, 1), kernel_size=(8, 8), \
#                     strides=(4, 4), padding='valid'))
#    model.add(Activation('relu'))
#    # Pooling
#    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
#    # Batch Normalisation before passing it to the next layer
#    model.add(BatchNormalization())
#
#    # 2nd Convolutional Layer
#    model.add(Conv2D(filters=100, kernel_size=(8, 8), strides=(1, 1), padding='valid'))
#    model.add(Activation('relu'))
#    # Pooling
#    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
#    # Batch Normalisation
#    model.add(BatchNormalization())
#
#    # Passing it to a dense layer
#    model.add(Flatten())
#    # 1st Dense Layer
#    model.add(Dense(1000))
#    model.add(Activation('relu'))
#    # Add Dropout to prevent overfitting
#    model.add(Dropout(0.4))
#    # Batch Normalisation
#    model.add(BatchNormalization())
#
#
#    # Output Layer
#    model.add(Dense(ACTIONS))
#
#    model.compile(loss='mse',
#                  optimizer=Adam(lr=0.001))
#    keras.utils.plot_model(model, to_file="model.png")
#    keras.utils.plot_model(model, to_file="model_with_shapes.png", show_shapes=True)
#    return model

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
    def __init__(self):
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
        self.REPLAY_MEMORY=10000 # Images to save for replay
        self.OBSERVE = 5000. # timesteps to observe before training
        self.EXPLORE = 5000. # frames over which to anneal epsilon
        self.ACTIONS=4
        self.memory = deque(maxlen=self.REPLAY_MEMORY)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.FINAL_EPSILON = 0.05 # final value of epsilon
        self.INITIAL_EPSILON = 1.0 # starting value of epsilon
        self.GAMMA = 0.99 # decay rate of past observations
        # Dimensions of our position board
        self.scale_down_factor = 10
        self.max_diff_x = math.floor(self.WINDOWHEIGHT / self.scale_down_factor)+self.scale_down_factor
        self.max_diff_y = math.floor(self.WINDOWWIDTH / self.scale_down_factor)+self.scale_down_factor
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
            f=open(self.TOPSCORE_FILE,'w')
            f.write(str(0))
            f.close()   
        v = open(self.TOPSCORE_FILE,'r')
        self.topScore = int(v.readline())
        v.close()
        # set up pygame, the window, and the mouse cursor
        pygame.init()
        self.mainClock = pygame.time.Clock()
        self.windowSurface = pygame.display.set_mode((self.WINDOWWIDTH, self.WINDOWHEIGHT))
        pygame.display.set_caption('car race')
        pygame.mouse.set_visible(False)
        # fonts
        self.font = pygame.font.SysFont(None, 30)
        # V
        self.t=0
        # Set all resetable states
        self.reset()

    # Reset state for a new round
    def reset(self):
        self.baddies = []
        #model.save(f"models/model_t{t:08}.h5")
        self.score = 0
        self.playerRect.topleft = (self.WINDOWWIDTH / 2, self.WINDOWHEIGHT - 50)
        self.moveLeft = self.moveRight = self.moveUp = self.moveDown = False
        self.baddieAddCounter = 0
        self.terminal = False
        self.position_board = np.zeros((self.max_diff_x*2, self.max_diff_y*2), np.bool)
    
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
        self.score += 1 # increase score
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
        for b in self.baddies:
            b['rect'].move_ip(0, b['speed'])
            relative_position = (self.playerRect.center[0] - b["rect"].center[0], self.playerRect.center[1] - b["rect"].center[1])
            relative_position = (math.floor(relative_position[0]/self.scale_down_factor), math.floor(relative_position[1]/self.scale_down_factor))
            if abs(relative_position[0]) > self.max_diff_x:
                self.max_diff_x = abs(relative_position[0])
                print("New max diff:", self.max_diff_x, self.max_diff_y)
                sys.exit(1)
            if abs(relative_position[1]) > self.max_diff_y:
                self.max_diff_y = abs(relative_position[1])
                print("New max diff:", self.max_diff_x, self.max_diff_y)
                sys.exit(1)
            self.position_board[relative_position[0] + self.max_diff_x, relative_position[1] + self.max_diff_y] = True

        for b in self.baddies[:]:
            if b['rect'].top > self.WINDOWHEIGHT:
                self.baddies.remove(b)

        # Draw the game world on the window.
        self.windowSurface.fill(self.BACKGROUNDCOLOR)

        # Draw the score and top score.
        self._drawText('Score: %s' % (self.score), self.font, self.windowSurface, 128, 0)
        self._drawText('Top Score: %s' % (self.topScore), self.font, self.windowSurface, 128, 20)
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

        im = 255 * (self.position_board / 1) # Divide with max value. Bool => 1
        w, h = im.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
        surf = pygame.surfarray.make_surface(ret)
        self.windowSurface.blit(surf, (0,0))
        pygame.display.update()

        print("own pos:", own_pos, "Len:", len(self.baddies), "terminal:", self.terminal, "score:", self.score)
        if self.score != 0:
            self.memory.append((im, own_pos, (direction, gas), self.terminal, self.score))

        while len(self.memory) > self.REPLAY_MEMORY:
            self.memory.popleft()
        
        self.t+=1

        if len(self.baddies) >= 40:
            print("More then 40 other cars:",len(self.baddies))
        
        return self.terminal
    
    def get_last_memory_image(self):
        return self.memory[-1][0]