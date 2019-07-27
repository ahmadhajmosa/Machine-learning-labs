from game import GameDing, LeftRight, FrontBack
import numpy as np
import tensorflow as tf
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tensorflow.keras.layers as kl
from tqdm import tqdm

print("Tensorflow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
    def __init__(self, num_actions, field_size):
        super().__init__('mlp_policy')
        # no tf.get_variable(), just simple Keras API
        # Input_size: tuple of integers, does not include the sample axis, e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".
        self.input_size = (field_size[0]*2, field_size[1]*2, 1)
        self._build_input_shape = self.input_size
        self.conv1 = kl.Conv2D(filters=32, input_shape=self.input_size, kernel_size=(8, 8), \
            strides=(4, 4), padding='valid', data_format='channels_last')
        self.activator1 = kl.Activation('relu')
        # Pooling
        self.pool1 = kl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        # Batch Normalisation before passing it to the next layer
        self.normliser1 = kl.BatchNormalization()
        self.flatter = kl.Flatten()
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation='relu')
        # logits are unnormalized log probabilities
        self.logits_lr = kl.Dense(num_actions[0], name='policy_logits') # Left - right
        self.logits_fr = kl.Dense(num_actions[1], name='policy_logits') # forward - backwards/reverse
        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        conved = self.conv1(inputs)
        activaded = self.activator1(conved)
        pooled = self.pool1(activaded)
        flatted = self.flatter(pooled)
        # separate hidden layers from the same input tensor
        hidden_logs = self.hidden1(flatted)
        hidden_vals = self.hidden2(flatted)
        return self.logits_lr(hidden_logs), self.logits_fr(hidden_vals)

    def action_value(self, obs):
        #print(np.shape(obs))
        #print(obs)
        reshaped = np.reshape(obs, (1,self.input_size[0], self.input_size[1], 1))
        # executes call() under the hood
        logits_lr, logits_fr = self.predict(reshaped)
        #logits_lr, logits_fr = self.call(obs)
        action_lr = self.dist.predict(logits_lr)
        action_fr = self.dist.predict(logits_fr)
        # a simpler option, will become clear later why we don't use it
        # action = tf.random.categorical(logits, 1)
        return (action_lr, action_fr)


class A2CAgent:
    def __init__(self, model):
        # hyperparameters for loss terms
        self.params = {'value': 0.5, 'entropy': 0.0001, 'gamma': 0.99}
        self.model = model
        self.model.compile(
            optimizer=ko.RMSprop(lr=0.0007),
            # define separate losses for policy logits and value estimate
            loss=[self._logits_loss, self._logits_loss]
        )
    
    def train(self, env, batch_sz=64, updates=1000):
        # storage helpers for a single batch of data
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + (env.max_diff_x*2,) + (env.max_diff_y*2,) + (1,))
        # training loop: collect samples, send to optimizer, repeat updates times
        ep_rews = [0.0]
        env.reset()
        env.next_round(LeftRight.NEUTRAL, FrontBack.NEUTRAL)
        next_obs = env.get_last_memory_image()
        for update in tqdm(range(updates)):
            for step in range(batch_sz):
                edit_ops = np.expand_dims(next_obs.copy(), axis=-1)
                observations[step] = edit_ops
                actions[step], values[step] = self.model.action_value(edit_ops[None, :])
                dones[step] = env.next_round(LeftRight(actions[step] - 1), FrontBack(values[step] - 1))
                #next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                next_obs = env.get_last_memory_image()
                rewards[step] = env.score

                ep_rews[-1] += rewards[step]
                if dones[step]:
                    ep_rews.append(0.0)
                    env.reset()
                    dones[step] = env.next_round(LeftRight(actions[step] - 1), FrontBack(values[step] - 1))
                    next_obs = env.get_last_memory_image()

            _, next_value = self.model.action_value(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # a trick to input actions and advantages through same API
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            # performs a full training step on the collected batch
            # note: no need to mess around with gradients, Keras API handles it
            print(observations.shape, acts_and_advs.shape, returns.shape)
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
        return ep_rews

    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t+1] * (1-dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages

    
    def test(self, env):
        env.reset()
        env.next_round(LeftRight.NEUTRAL, FrontBack.NEUTRAL)
        obs = env.get_last_memory_image()
        done = False
        ep_reward = 0
        while not done:
            action, value = self.model.action_value(obs[None, :])
            done = env.next_round(LeftRight(action - 1), FrontBack(value - 1))
            reward = 1 
            obs = env.get_last_memory_image()

            ep_reward += reward
        return ep_reward

    def _value_loss(self, returns, value):
        # value loss is typically MSE between value estimates and returns
        return self.params['value']*kls.mean_squared_error(returns, value)

    def _logits_loss(self, acts_and_advs, logits):
        # a trick to input actions and advantages through same API
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        # sparse categorical CE loss obj that supports sample_weight arg on call()
        # from_logits argument ensures transformation into normalized probabilities
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        # policy loss is defined by policy gradients, weighted by advantages
        # note: we only calculate the loss on the actions we've actually taken
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # entropy loss can be calculated via CE over itself
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        # here signs are flipped because optimizer minimizes
        return policy_loss - self.params['entropy']*entropy_loss


env = GameDing()
model = Model(num_actions=(len(LeftRight), len(FrontBack)), field_size=(env.max_diff_x, env.max_diff_y))

if False:
    obs = env.reset()
    env.next_round(LeftRight.NEUTRAL, FrontBack.NEUTRAL)
    # no feed_dict or tf.Session() needed at all
    inpt = [env.memory[-1][0]]
    ret = model.action_value(inpt)
    print(ret) # [1] [-0.00145713]
    print(len(ret[0]))

agent = A2CAgent(model)

rewards_history = agent.train(env)
print("Finished training, testing...")
while True:
    print("%d out of 200" % agent.test(env)) # 200 out of 200