import pygame, random, sys ,os,time
from pygame.locals import *
from matplotlib.pyplot import imshow
from PIL import Image
from io import StringIO
import keras
from keras.layers import Dense, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam
from collections import deque
import tensorflow as tf
import cv2
import math
import resource

# set SDL to use the dummy NULL video driver, 
#   so it doesn't need a windowing system.
#os.environ["SDL_VIDEODRIVER"] = "dummy"


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, \
    Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np

# Safty first!
# Set RAM limit
#max_ram = 10000*1e3 # in byte
#resource.setrlimit(resource.RLIMIT_AS, (max_ram,max_ram))
#print(f"Set RAM limit to max {max_ram} bytes. Equals {max_ram/1e3}MB.")

WINDOWWIDTH = 800
WINDOWHEIGHT = 600
TEXTCOLOR = (255, 255, 255)
BACKGROUNDCOLOR = (0, 0, 0)
FPS = 10
BADDIEMINSPEED = 8
BADDIEMAXSPEED = 8
ADDNEWBADDIERATE = 6
PLAYERMOVERATE = 5
count=3
REPLAY_MEMORY=10000
OBSERVE = 5000. # timesteps to observe before training
EXPLORE = 5000. # frames over which to anneal epsilon
ACTIONS=4
memory = deque(maxlen=2000)
gamma = 0.95  # discount rate
epsilon = 1.0  # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
GAMMA = 0.99 # decay rate of past observations

# Dimensions of our position board
scale_down_factor = 10
max_diff_x = math.floor(600/scale_down_factor)
max_diff_y = math.floor(700/scale_down_factor)

# reinforce
def replay(batch_size):
        minibatch = random.sample(memory, batch_size)
        for state, own_pos, action, done, points in minibatch:
            target = points
            reshaped_state = np.reshape(state, (1, max_diff_x*2, max_diff_y*2, 1))
            target_f = model.predict(reshaped_state)
            target_f[0][action] = target
            model.fit(reshaped_state, target_f, epochs=1, verbose=0)

def calc_points(state, done):
    target = 0
    found_points = 0
    objects = np.nonzero(state)
    for i in range(len(objects[0])):
        idx_x = objects[0][i]
        idx_y = objects[1][i]
        current = state[idx_x, idx_y]
        if current == True:
            x = idx_x-max_diff_x
            y = idx_y-max_diff_y
            distance = (np.sqrt(x**2+y**2)-1)/10
            print(distance)
            target -= 1/distance
            found_points += 1
    if done == True:
        target = -1
        return target
    if found_points == 0:
        target = 0
        return target
    target /= found_points
    return target

# build convnet

def build_model():
    # Neural Net for Deep-Q learning Model

    #np.random.seed(1000)

    # (2) Get Data

    # (3) Create a sequential model
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=32, input_shape=(max_diff_x*2, max_diff_y*2, 1), kernel_size=(8, 8), \
                     strides=(4, 4), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=100, kernel_size=(8, 8), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())


    # Output Layer
    model.add(Dense(ACTIONS))

    model.compile(loss='mse',
                  optimizer=Adam(lr=0.001))
    keras.utils.plot_model(model, to_file="model.png")
    keras.utils.plot_model(model, to_file="model_with_shapes.png", show_shapes=True)
    return model

model = build_model()
# game termination

def terminate():
    pygame.quit()
    sys.exit()

def waitForPlayerToPressKey():
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                terminate()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE: #escape quits
                    terminate()
                return
# check collision
def playerHasHitBaddie(playerRect, baddies):
    for b in baddies:
        if playerRect.colliderect(b['rect']):
            return True
    return False

def drawText(text, font, surface, x, y):
    textobj = font.render(text, 1, TEXTCOLOR)
    textrect = textobj.get_rect()
    textrect.topleft = (x, y)
    surface.blit(textobj, textrect)

# set up pygame, the window, and the mouse cursor
pygame.init()
mainClock = pygame.time.Clock()
windowSurface = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
pygame.display.set_caption('car race')
pygame.mouse.set_visible(False)

# fonts
font = pygame.font.SysFont(None, 30)

# sounds
#gameOverSound = pygame.mixer.Sound('music/crash.wav')
#pygame.mixer.music.load('music/car.wav')
#laugh = pygame.mixer.Sound('music/laugh.wav')


# images
playerImage = pygame.image.load('image/car1.png')
car3 = pygame.image.load('image/car3.png')
car4 = pygame.image.load('image/car4.png')
playerRect = playerImage.get_rect()
baddieImage = pygame.image.load('image/car2.png')
sample = [car3,car4,baddieImage]
wallLeft = pygame.image.load('image/left.png')
wallRight = pygame.image.load('image/right.png')


zero=0
if not os.path.exists("data/save.dat"):
    f=open("data/save.dat",'w')
    f.write(str(zero))
    f.close()   
v=open("data/save.dat",'r')
topScore = int(v.readline())
v.close()

t=0
# main game loop
while (count>0):
    # start of the game
    baddies = []
    model.save(f"models/model_t{t:08}.h5")
    score = 0
    playerRect.topleft = (WINDOWWIDTH / 2, WINDOWHEIGHT - 50)
    moveLeft = moveRight = moveUp = moveDown = False
    baddieAddCounter = 0
    terminal = False
    position_board = np.zeros((max_diff_x*2, max_diff_y*2), np.bool)
    while True: # the game loop
        score += 1 # increase score


        mainClock.tick(FPS) # Can set here the max FPS

        a_t = np.zeros([ACTIONS])
        action_index = 0
        if random.random() <= epsilon or t <= OBSERVE:
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            reshaped_state = np.reshape(position_board, (1, max_diff_x*2, max_diff_y*2, 1))
            readout_t = model.predict(reshaped_state)
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        if action_index ==0:
            moveRight = False
            moveLeft = True
        elif action_index ==1:
            moveLeft = False
            moveRight = True
        elif action_index ==2:
            moveDown = False
            moveUp = True
        else:
            moveUp = False
            moveDown = True

        for event in pygame.event.get():

            if event.type == QUIT:
                terminate()


            # Brought to you by code-projects.org
            if event.type == KEYUP:
                if event.key == K_ESCAPE:
                    terminate()


        # Add new baddies at the top of the screen
        baddieAddCounter += 1
        if baddieAddCounter == ADDNEWBADDIERATE:
            baddieAddCounter = 0
            baddieSize = 30
            newBaddie = {'rect': pygame.Rect(random.randint(140, 485), 0 - baddieSize, 23, 47),
                         'speed': random.randint(BADDIEMINSPEED, BADDIEMAXSPEED),
                         'surface': pygame.transform.scale(random.choice(sample), (23, 47)),
                         }
            baddies.append(newBaddie)
            sideLeft = {'rect': pygame.Rect(0, 0, 126, 600),
                        'speed': random.randint(BADDIEMINSPEED, BADDIEMAXSPEED),
                        'surface': pygame.transform.scale(wallLeft, (126, 599)),
                        }
            baddies.append(sideLeft)
            sideRight = {'rect': pygame.Rect(497, 0, 303, 600),
                         'speed': random.randint(BADDIEMINSPEED, BADDIEMAXSPEED),
                         'surface': pygame.transform.scale(wallRight, (303, 599)),
                         }
            baddies.append(sideRight)

        # Move the player around.
        if moveLeft and playerRect.left > 0:
            playerRect.move_ip(-1 * PLAYERMOVERATE, 0)
        if moveRight and playerRect.right < WINDOWWIDTH:
            playerRect.move_ip(PLAYERMOVERATE, 0)
        if moveUp and playerRect.top > 0:
            playerRect.move_ip(0, -1 * PLAYERMOVERATE)
        if moveDown and playerRect.bottom < WINDOWHEIGHT:
            playerRect.move_ip(0, PLAYERMOVERATE)

        position_board.fill(False)
        for b in baddies:
            b['rect'].move_ip(0, b['speed'])
            relative_position = (playerRect.center[0] - b["rect"].center[0], playerRect.center[1] - b["rect"].center[1])
            relative_position = (math.floor(relative_position[0]/scale_down_factor), math.floor(relative_position[1]/scale_down_factor))
            if abs(relative_position[0]) > max_diff_x:
                max_diff_x = abs(relative_position[0])
                print("New max diff:", max_diff_x,max_diff_y)
                sys.exit(1)
            if abs(relative_position[1]) > max_diff_y:
                max_diff_y = abs(relative_position[1])
                print("New max diff:", max_diff_x,max_diff_y)
                sys.exit(1)
            position_board[relative_position[0] + max_diff_x, relative_position[1] + max_diff_y] = True

        for b in baddies[:]:
            if b['rect'].top > WINDOWHEIGHT:
                baddies.remove(b)

        # Draw the game world on the window.
        windowSurface.fill(BACKGROUNDCOLOR)

        # Draw the score and top score.
        drawText('Score: %s' % (score), font, windowSurface, 128, 0)
        drawText('Top Score: %s' % (topScore), font, windowSurface, 128, 20)
        #drawText('Rest Life: %s' % (count), font, windowSurface, 128, 40)

        windowSurface.blit(playerImage, playerRect)

        # data = pygame.image.tostring(windowSurface, 'RGBA')
        # img = Image.frombytes('RGBA', (100, 200), data)
        # imshow(data)
        for b in baddies:
            windowSurface.blit(b['surface'], b['rect'])

        pygame.display.update()

        # Check if any of the car have hit the player.
        if playerHasHitBaddie(playerRect, baddies):
            terminal = True
            if score > topScore:
                g = open("data/save.dat", 'w')
                g.write(str(score))
                g.close()
                topScore = score
        else:
            terminal = False

        own_pos = (playerRect.x, playerRect.y)
        points = calc_points(position_board, terminal)

        im = 255 * (position_board / position_board.max())
        w, h = im.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
        surf = pygame.surfarray.make_surface(ret)
        windowSurface.blit(surf, (0,0))
        pygame.display.update()

        print("own pos:", own_pos, "Len:", len(baddies), "points:", points, "terminal:", terminal, "score:", score)
        if points != 0:
            memory.append((position_board, own_pos, action_index, terminal, points))

        if terminal == True:
            break

        while len(memory) > REPLAY_MEMORY:
            memory.popleft()

        if t > OBSERVE:
            replay(64)
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
        t+=1

        if len(baddies) >= 40:
            print("More then 40 other cars:",len(baddies))
        
        if t%80 == 0:
            print("Frame:", t)
    # "Game Over" screen.
    #count=count-1
    time.sleep(0.1)
    #if (count==0):
     #laugh.play()
     #drawText('Game over', font, windowSurface, (WINDOWWIDTH / 3), (WINDOWHEIGHT / 3))
     #drawText('Press any key to play again.', font, windowSurface, (WINDOWWIDTH / 3) - 80, (WINDOWHEIGHT / 3) + 30)
     #pygame.display.update()
     #time.sleep(2)
     #waitForPlayerToPressKey()
     #count=3
