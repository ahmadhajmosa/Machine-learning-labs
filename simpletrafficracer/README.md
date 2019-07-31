# Car simulatation
## by Michał Gilski

# Changes
## (diff_arch.py)
Attempt at different architecture of model, preset resizing and skipping the cnn part to make the model simpler and easier to train

## (record.py)
Record the game states during a human playthrough and save them to files as states with the corresponding actions

## (game.py)
 - Start using pretrained model either loaded off the disk or from the recorded games (imitation learning)
 - Change the reward funtion to make make the car stay more likely in the center and bottom
 - Regular RL to continue the learing
 - Saving the model and the results in files to train the same model 
