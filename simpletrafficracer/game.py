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
import cv2


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, \
    Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
WINDOWWIDTH = 800
WINDOWHEIGHT = 600
TEXTCOLOR = (255, 255, 255)
BACKGROUNDCOLOR = (0, 0, 0)
FPS = 40
BADDIEMINSIZE = 10
BADDIEMAXSIZE = 40
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
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))




# reinforce
def replay(batch_size):
        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + gamma * \
                       np.amax(model.predict(np.reshape(next_state,(1,80,80,12)))[0])

            target_f = model.predict(np.reshape(state,(1,80,80,12)))
            target_f[0][action] = target
            model.fit(np.reshape(state,(1,80,80,12)), target_f, epochs=1, verbose=0)


# build convnet

def build_model():
    # Neural Net for Deep-Q learning Model
    # (1) Importing dependency
    import keras

    np.random.seed(1000)

    # (2) Get Data

    # (3) Create a sequential model
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=32, input_shape=(80, 80, 4*3), kernel_size=(8, 8), \
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


# "Start" screen
drawText('Press any key to start the game.', font, windowSurface, (WINDOWWIDTH / 3) - 30, (WINDOWHEIGHT / 3))
drawText('And Enjoy', font, windowSurface, (WINDOWWIDTH / 3), (WINDOWHEIGHT / 3)+30)
pygame.display.update()
waitForPlayerToPressKey()
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
    score = 0
    playerRect.topleft = (WINDOWWIDTH / 2, WINDOWHEIGHT - 50)
    moveLeft = moveRight = moveUp = moveDown = False
    reverseCheat = slowCheat = False
    baddieAddCounter = 0
    terminal = False
    while True: # the game loop
        score += 1 # increase score


        mainClock.tick(FPS)

        x_t = pygame.surfarray.array3d(pygame.display.get_surface())
        x_t = x_t.reshape((600, 800, 3))
        x_t = cv2.resize(x_t, (80, 80))
        if t==0:
            s_t = np.stack((x_t, x_t, x_t, x_t), axis=2).reshape(80,80,12)
            s_t1 = s_t
        else:
            x_t = np.reshape(x_t,(80,80,3))
            s_t1 = np.append(x_t, s_t[:,:,0:9], axis = 2).reshape(80,80,12)


        a_t = np.zeros([ACTIONS])
        action_index = 0
        if random.random() <= epsilon or t <= OBSERVE:
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            readout_t = model.predict(np.reshape(s_t,(1,80,80,12)))
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
        if not reverseCheat and not slowCheat:
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

        for b in baddies:
            if not reverseCheat and not slowCheat:
                b['rect'].move_ip(0, b['speed'])
            elif reverseCheat:
                b['rect'].move_ip(0, -5)
            elif slowCheat:
                b['rect'].move_ip(0, 1)

        for b in baddies[:]:
            if b['rect'].top > WINDOWHEIGHT:
                baddies.remove(b)

        # Draw the game world on the window.
        windowSurface.fill(BACKGROUNDCOLOR)

        # Draw the score and top score.
        drawText('Score: %s' % (score), font, windowSurface, 128, 0)
        drawText('Top Score: %s' % (topScore), font, windowSurface, 128, 20)
        drawText('Rest Life: %s' % (count), font, windowSurface, 128, 40)

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
            r_t = -1
            if score > topScore:
                g = open("data/save.dat", 'w')
                g.write(str(score))
                g.close()
                topScore = score
            break
        else:
            terminal = False

            r_t = 0


        memory.append((s_t, action_index, r_t, s_t1, terminal))

        if len(memory) > REPLAY_MEMORY:
            memory.popleft()

        if t > OBSERVE:
            replay(64)
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            #D.append((s_t, a_t, r_t, s_t1, terminal))
        t+=1

        print(t)
    # "Game Over" screen.
    #count=count-1
    time.sleep(1)
    #if (count==0):
     #laugh.play()
     #drawText('Game over', font, windowSurface, (WINDOWWIDTH / 3), (WINDOWHEIGHT / 3))
     #drawText('Press any key to play again.', font, windowSurface, (WINDOWWIDTH / 3) - 80, (WINDOWHEIGHT / 3) + 30)
     #pygame.display.update()
     #time.sleep(2)
     #waitForPlayerToPressKey()
     #count=3
