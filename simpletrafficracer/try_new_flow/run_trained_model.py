import game_agent_driver
import tensorflow as tf
import tf_agents

model_path = "model/model_round_%05d.h5"
save_nr = 42000

env = game_agent_driver.RaceGameEnv()
saved_policy = tf.compat.v2.saved_model.load(model_path % save_nr)
get_initial_state_fn = saved_policy.signatures['get_initial_state']
action_fn = saved_policy.signatures['action']
policy_state_dict = get_initial_state_fn(batch_size=3)
time_step_dict = env.reset()
while True:
    time_step_state = dict(time_step_dict)
    time_step_state.update(policy_state_dict)
    policy_step_dict = action_fn(time_step_state)
    time_step_dict = env.step(action_dict)
  