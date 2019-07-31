"""
Can not run at the moment
"""
import tensorflow as tf
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tensorflow.keras.layers as kl
import numpy as np
from tqdm import tqdm
from game import GameDing, LeftRight, FrontBack


print("Tensorflow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)


def build_model(shape, export_graphstructue=True):
    # shape0 x shape1 Gray images
    input_layer = tf.keras.Input(
        shape=(shape[0], shape[1]), dtype="float16", name="input_layer")
    reshaped = kl.Reshape(target_shape=(
        shape[0], shape[1], 1), name="reshaper")(input_layer)
    conv1 = kl.Conv2D(filters=32, kernel_size=(
        8, 8), data_format="channels_last", padding="valid")(reshaped)
    activator = kl.Activation("relu")(conv1)
    pool1 = kl.MaxPool2D()(activator)
    norm1 = kl.BatchNormalization()(pool1)
    flat = kl.Flatten()(norm1)

    # from here on split. Each path get its own dense layer
    left_right_dense = kl.Dense(
        128, activation="relu", name="left_right_dense")(flat)
    forward_backward_dense = kl.Dense(
        128, activation="relu", name="forward_backward_dense")(flat)

    # Here we convert to (unscaled) Propabilities. Called "Logits"
    steer_actions = 3  # left, right, strait
    gas_actions = 3  # forward, backwards, neutral
    left_right_logits = kl.Dense(
        steer_actions, name="left_right_logits")(left_right_dense)
    forward_backward_logits = kl.Dense(
        gas_actions, name="forward_backward_logits")(forward_backward_dense)

    # Convert to category
    # Note: It is important to sample from this distribution as taking the argmax of the distribution can easily get the model stuck in a loop.
    # TODO: Implement, or leaf it at `action_value`-function?

    # create a model from our layers
    model = tf.keras.Model(inputs=[input_layer], outputs=[
                           left_right_logits, forward_backward_logits])

    # Setup optimizer and loss. Here we could set weigths for the outputs or use differert loss functions.
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    if export_graphstructue:
        # Plot architecture
        tf.keras.utils.plot_model(model, "model_plot.png", show_shapes=True)

    return model


def predict_action(model: tf.keras.Model, inpt) -> (LeftRight, FrontBack):
    ret = model.predict(inpt)
    left_right_logits = ret[0]
    front_back_logits = ret[1]
    a = tf.squeeze(tf.random.categorical(left_right_logits, 1), axis=-1)
    b = tf.squeeze(tf.random.categorical(front_back_logits, 1), axis=-1)
    # Cast from tensorflow to numpy to our steering datatype
    return (LeftRight(a.numpy()[0]-1), FrontBack(b.numpy()[0]-1))

class A2CAgent:
    def __init__(self, model: tf.keras.Model, shape, gamma=0.99):
        self.model = model
        self.gamma = gamma  # discount factor
        self.shape = shape
    
    def train(self, environment: GameDing, use_tensorboard=False, batch_size=32, rounds=1000) -> [float]:
        action_space = 2 # left+right / front+back
        actions = np.empty((batch_size,) + (action_space,), dtype=np.int32)
        rewards, dones = np.empty((2, batch_size))
        observations = np.empty((batch_size,) + self.shape)
        ep_data = [0.0]

        environment.reset()
        for r in tqdm(range(rounds)):
            for b in range(batch_size):
                obs = environment.get_last_memory_image()
                observations[b] = obs.copy()

                (lr, fb) = predict_action(self.model, obs[None, :])
                actions[b] = (lr.value, fb.value)  # TODO: Translate here from [-1,1] to [0,2]?
                dead = environment.next_round(lr, fb, realtime=False)
                reward = 0
                if dead:
                    reward = -10
                else:
                    reward = 1
                rewards[b] = reward
                ep_data[-1] += reward
                dones[b] = dead
                if dead:
                    ep_data.append(0.0)
                    env.reset()
            discounted_returns = self._return_discount(rewards, dones)
            losses = self.model.train_on_batch(observations, [actions, discounted_returns])
        return ep_data
    
    def test(self, environment: GameDing) -> int:
        dead = False
        environment.reset()
        points = 0
        while not dead:
            obs = environment.get_last_memory_image()
            (lr, fb) = predict_action(self.model, obs[None, :])
            dead = environment.next_round(lr, fb, realtime=False)
            if not dead:
                points += 1
        return points
    
    def _return_discount(self, rewards, deads):
        start_value = [rewards[-1]]
        returns = np.append(np.zeros_like(rewards), start_value, axis=-1)
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t+1] * (1-deads[t])
        returns = returns[:-1] # reverse array
        return returns

env = GameDing()
shape = (env.max_diff_x*2, env.max_diff_y*2)
model = build_model(shape=shape,
                    export_graphstructue=True)
agent = A2CAgent(model, shape=shape)

if False: # test model without train
    while True:
        obs = env.get_last_memory_image()
        (lr, fb) = predict_action(model, obs[None, :])
        dead = env.next_round(lr, fb)
        if dead:
            env.reset()

if True: # train models
    rewards_history = agent.train(env)
    print("Finished training, testing...")

if False: # Test model
    while True:
        points = agent.test(env)
        print(f"got {points} points")
