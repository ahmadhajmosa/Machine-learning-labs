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

import pickle

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, \
    Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
WINDOWWIDTH = 800
WINDOWHEIGHT = 600
TEXTCOLOR = (255, 255, 255)
BACKGROUNDCOLOR = (0, 0, 0)
FPS = 8
BADDIEMINSIZE = 10
BADDIEMAXSIZE = 40
BADDIEMINSPEED = 8
BADDIEMAXSPEED = 8
ADDNEWBADDIERATE = 6
PLAYERMOVERATE = 5
count=3
REPLAY_MEMORY=10000
OBSERVE = 5000. # timesteps to observe before training
EXPLORE = 3000. # frames over which to anneal epsilon
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


train =  0# frames assigend for training the model
#If train = 0, the game will start trainign using the data. Set a value larger
# than 0 to paly or set 0 for training
batchY = deque(maxlen=10000) # actions performed by player
batchX = deque(maxlen=10000) # capture of frame
#leftAvoidFra = deque(maxlen=4000)
#leftAvoidAct = deque(maxlen=4000)
AvoidFra = deque(maxlen=4000)
AvoidAct = deque(maxlen=4000)
x_frames = 1 # Frequency of frame/action append. Append each x_frames
trainedModel = False

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


def build_model2():
    # Neural Net for Deep-Q learning Model
    # (1) Importing dependency
    import keras

    np.random.seed(1000)


    # (3) Create a sequential model
    model2 = Sequential()

    # 1st Convolutional Layer
    model2.add(Conv2D(filters=32, input_shape=(100, 100, 3), kernel_size=(8, 8), \
                     strides=(4, 4), padding='valid'))
    model2.add(Activation('relu'))
    model2.add(BatchNormalization())

    # 2nd Convolutional Layer
    model2.add(Conv2D(filters=100, kernel_size=(4, 4), strides=(1, 1), padding='valid'))
    model2.add(Activation('relu'))
    model2.add(BatchNormalization())
    
    #3rd Convolutional Layer
    model2.add(Conv2D(filters=60, kernel_size=(2, 2), strides=(1, 1), padding='valid'))
    model2.add(Activation('relu'))
    model2.add(BatchNormalization())
    
    #4th Convolutional Layer
#    model2.add(Conv2D(filters=60, kernel_size=(1, 1), strides=(1, 1), padding='valid'))
#    model2.add(Activation('relu'))
#    model2.add(BatchNormalization())

    # Passing to a dense layer
    model2.add(Flatten())
    # 1st Dense Layer
    model2.add(Dense(1000))
    model2.add(Activation('relu'))

    # 2nd Dense Layer
#    model2.add(Dense(100))
#    model2.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model2.add(Dropout(0.4))
    # Batch Normalisation
    model2.add(BatchNormalization())


    # Output Layer
    model2.add(Dense(5)) # 5 possible actions
    #model.add(Dense(ACTIONS))
    model2.add(Activation('softmax'))

    model2.compile(loss='mse', optimizer=Adam(lr=0.001))
    return model2

model2 = build_model2()

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
    
def appendEachX(x_frames, t): #will append the frame each x frames
    if t%x_frames == 0:
        #batchX.append(stackFrame1)
        batchX.append(frame)
        batchY.append(actionVector)
        
#def fitConvNet():
#    model2.fit(np.array(batchX), np.array(batchY), epochs=10, verbose=1)
#def replay(batch_size):
#        minibatch = random.sample(memory, batch_size)
#        for state, action, reward, next_state, done in minibatch:
#            target = reward
#            if not done:
#              target = reward + gamma * \
#                       np.amax(model.predict(np.reshape(next_state,(1,80,80,12)))[0])
#
#            target_f = model.predict(np.reshape(state,(1,80,80,12)))
#            target_f[0][action] = target
#            model.fit(np.reshape(state,(1,80,80,12)), target_f, epochs=1, verbose=0)   
    

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
        actionWas = 0 # keep track of movement of player
        score += 1 # increase score

        mainClock.tick(FPS)        
        

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

# Lowers the existing baddies
        for b in baddies:
            if not reverseCheat and not slowCheat:
                b['rect'].move_ip(0, b['speed'])
            elif reverseCheat:
                b['rect'].move_ip(0, -5)
            elif slowCheat:
                b['rect'].move_ip(0, 1)
            
# Delete the baddies out of screen
        for b in baddies[:]:
            if b['rect'].top > WINDOWHEIGHT:
                baddies.remove(b)
            
# Get the frames for training/predicting
        frame = pygame.surfarray.array3d(pygame.display.get_surface())#800*600*3
        frame = frame[110:514, 60:600] # crop frame to RoI 404*540*3  
        #frame = frame.reshape((540, 404, 3))
        frame = cv2.resize(frame, (100, 100)) # downscale frame to 100*100        
       
# Draw the game world on the window.
        windowSurface.fill(BACKGROUNDCOLOR)



## Keyboard data aquisition during the first "train" (5000) frames        
        if t < train: # get data for training CNN                   
            ## keyboard input reader
            for event in pygame.event.get():
                # Manual move the player around.
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT: 
                        actionWas = 1
                    elif event.key == pygame.K_RIGHT:
                        actionWas = 2
                    elif event.key == pygame.K_UP:                    
                        actionWas = 3
                    elif event.key == pygame.K_DOWN:
                        actionWas = 4                        
                # Manual close game
                if event.type == KEYUP:
                    if event.key == K_ESCAPE:
                        terminate()
                if event.type == QUIT:
                    terminate()
                    
            ## Car manager                  
            # Get the action performes/Move car with manual input
            if actionWas == 1:
                actionVector = [1,0,0,0,0]
                playerRect.move_ip(-1 * PLAYERMOVERATE, 0) # left
                AvoidFra.append(frame)
                AvoidAct.append(actionVector)
            elif actionWas == 2:
                actionVector = [0,1,0,0,0]
                playerRect.move_ip(PLAYERMOVERATE, 0) # right
                AvoidFra.append(frame)
                AvoidAct.append(actionVector)
            elif actionWas == 3:
                actionVector = [0,0,1,0,0]
                playerRect.move_ip(0, -1 * PLAYERMOVERATE) # up
            elif actionWas == 4:
                actionVector = [0,0,0,1,0]
                playerRect.move_ip(0, PLAYERMOVERATE) # down
            else :#actionWas == 0
                actionVector = [0,0,0,0,1] # no move                
            
#            ## Data manager
#            if len(batchX) > batchX_mem:
#                batchX.popleft()
#            if len(batchY) > batchY_mem:
#                batchY.popleft()
            # gather the data
#            appendEachX(x_frames, t) # appends frame and action every X frames (2)
            
#def appendEachX(x_frames, t): #will append the frame each x frames
#    if t%x_frames == 0:
#        batchX.append(stackFrame1)
#        batchY.append(actionVector)
            
# Draw the score while manual contrl
            drawText('Manual Score: %s' % (score), font, windowSurface, 128, 0)
            drawText('Top Score: %s' % (topScore), font, windowSurface, 128, 20)
            drawText('Rest Life: %s' % (count), font, windowSurface, 128, 40)

            windowSurface.blit(playerImage, playerRect)

            for b in baddies:
                windowSurface.blit(b['surface'], b['rect'])

            pygame.display.update()            
            
            

        else: # data gathered, train and auto play            
            if trainedModel == False:          

                #export training data
#                outputX = open('export_batchX.pkl', 'wb')
#                pickle.dump(batchX, outputX)
#                outputX.close()
#                outputY = open('export_batchY.pkl', 'wb')
#                pickle.dump(batchY, outputY)
#                outputY.close()

                #import data
                inputX = open('export_batchX.pkl', 'rb')
                batchX = pickle.load(inputX)
                inputX.close()
                inputY = open('export_batchY.pkl', 'rb')
                batchY = pickle.load(inputY)
                inputY.close()            
                
                model2.fit(np.array(batchX), np.array(batchY), epochs=15, verbose=1)
                #FPS = 40 # Set higher frame rate for auto play
                tt = 0
                trainedModel = True

            predictAction =  model2.predict(np.reshape(frame,(1,100,100,3)))
            predictAction = np.reshape(predictAction,(5,1))
            predictAction = np.argmax(predictAction)

            if predictAction == 0:
                playerRect.move_ip(-1 * PLAYERMOVERATE, 0) # left
            elif predictAction == 1:
                playerRect.move_ip(PLAYERMOVERATE, 0) # right
            elif predictAction == 2:
                playerRect.move_ip(0, -1 * PLAYERMOVERATE) # up
            elif predictAction == 3:
                playerRect.move_ip(0, PLAYERMOVERATE) # down

            # Draw the score while auto control
            tt = tt+1
            drawText('Auto Score: %s' % (tt), font, windowSurface, 128, 0)
            drawText('Top Score: %s' % (topScore), font, windowSurface, 128, 20)
            drawText('Rest Life: %s' % (count), font, windowSurface, 128, 40)

            windowSurface.blit(playerImage, playerRect)

            for b in baddies:
                windowSurface.blit(b['surface'], b['rect'])

            pygame.display.update()     
            
            # Manual close game
            for event in pygame.event.get():                                      
                if event.type == KEYUP:
                    if event.key == K_ESCAPE:
                        terminate()
                if event.type == QUIT:
                    terminate()
            
            
## Collision manager
        # Check if any of the car have hit the player.
        if playerHasHitBaddie(playerRect, baddies):
            terminal = True
            tt = 0
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
        
        
#        if t > OBSERVE:
#            replay(64)
#            if epsilon > epsilon_min:
#                epsilon *= epsilon_decay

#        if len(memory) > REPLAY_MEMORY:
#            memory.popleft()
        




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
     
     #for x in range(0, 30):
     #     model.fit(np.array(batchX), np.array(batchY), epochs=10, verbose=1)

