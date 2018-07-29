from collections import deque;
import random
import scipy.signal;
import numpy as np;
import copy;


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1];

class BatchProvider(object):
    def __init__(self, *,
        reward_discount = 0.99,
        advantage_discount = 0.95,
        epochs = 10,
        horizon = 100):

        self._reward_discount = float(reward_discount);
        self._advantage_discount = advantage_discount;

        self._new = deque(maxlen = epochs * horizon);
        self._memory = deque(maxlen = epochs * horizon);
        self._buffer = deque(maxlen = horizon);

    def observe(self, state, reward, terminal, predicted_value, action):
        self._buffer.append((
            state,
            reward,
            terminal,
            predicted_value,
            action
            ));
        if terminal:
            self._process_buffer();
        if self._buffer.maxlen <= len(self._buffer):
            self._process_buffer();

    def _process_buffer(self):
        """Calculates the advantages and returns for the current buffer, adds the buffer to the memory and clears the buffer.
        """
        if len(self._buffer) <= 2:
            return;
        # Unpack buffer
        states, rewards, terminals, values, actions  = [list(l) for l in zip(*self._buffer)];

        # Calculate advantage (TODO: add paper)
        values = np.array(values);
        delta = rewards[:-1] + self._reward_discount * values[1:] - values[:-1]
        advantages = discount(delta, self._reward_discount*self._advantage_discount);

        T = len(self._buffer)-1;
        advantages = np.empty(T, 'float32');
        lastgae = 0;
        for t in reversed(range(T)):
            nonterminal = 1 - terminals[t+1];
            delta = rewards[t] + self._reward_discount * values[t+1]*nonterminal - values[t];
            advantages[t] = lastgae =  delta + self._reward_discount * self._advantage_discount * nonterminal * lastgae;




        # Calculate return
        returns = advantages + values[:-1];
        returns = discount(rewards[:-1], self._reward_discount);
        observations  =list(zip(states, actions, (advantages-advantages.mean())/advantages.std(), returns));
        self._new.extend(observations);
        self._memory.extend(observations);
        self._buffer.clear();
        self._buffer.append((states[-1], rewards[-1], terminals[-1], values[-1], actions[-1]));

    def get_batch(self, batch_size = 100):
        batch_size = min(len(self), batch_size);
        return [np.array(l) for l in zip(*random.sample(self._memory, batch_size))];

    def get_new(self):
        new = copy.deepcopy(self._new);
        self._new.clear();
        return [np.array(l) for l in zip(*new)]

    def iterate_batch(self, batch_size = 100):
        batch_size = min(len(self), batch_size);
        permuted = np.random.permutation(self._memory);
        for i in range(np.ceil(len(permuted)/batch_size).astype(int)):
            yield [np.array(l) for l in zip(*permuted[i*batch_size:(i+1)*batch_size])];




    def __len__(self):
        return len(self._memory);

    def __getitem__(self, key):
        return self._memory[key];
    @property
    def maxlen(self):
        return self._memory.maxlen;


from baselines.ppo1 import pposgd_simple, mlp_policy
import baselines.common.tf_util as U;
from baselines.common.mpi_adam import MpiAdam
from baselines.common import Dataset, explained_variance, fmt_row, zipsame

from tensorflow.python import debug as tf_debug
import tensorflow as tf;

class PPO(object):
    def __init__(self, env, policy_fn, batch,
        clip_param = 0.2,
        entropy_param = -0.01,
        value_param = 0.001,
        epochs = 4,
        batch_size = 64,
        adam_epsilon = 1e-5,
        observation_space = None,
        action_space = None,
        log_dir = None,
        writer = None ):

        # register properties
        self._env = env;
        if observation_space is None:
            self._ob_space = env.observation_space;
        else:
            self._ob_space = observation_space;
        if action_space is None:
            self._ac_space = env.action_space;
        else:
            self._ac_space = action_space;

        self.set_logdir(log_dir, writer)

        self._policy_fn = policy_fn;

        self._epochs = epochs;
        self._batch_size = batch_size;

        self._adam_epsilon = adam_epsilon;

        self._clip_param = tf.get_variable("clip_param",initializer=tf.constant(clip_param), trainable=False);
        self._entropy_param = tf.get_variable("entropy_param",initializer=tf.constant(entropy_param), trainable=False);
        self._value_param = tf.get_variable("value_param",initializer=tf.constant(value_param), trainable=False);

        self._multiply_placeholder = tf.placeholder(tf.float32, shape=() );
        self._multiply_entropy = tf.assign(self._entropy_param,tf.multiply(self._entropy_param, self._multiply_placeholder));


        # initialize empty structures
        self._tf_placeholders = dict();
        self._tf_results = dict();
        self._state = self._reward = self._terminal = self._action = self._predicted_value = self._prev_action = None;
        self._observation = 0;

        self._batch = batch;

        # perform setup
        session = U.single_threaded_session();
        #session = tf_debug.LocalCLIDebugWrapperSession(session);
        #session.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)
        self._session = session;

        self._setup();

    def _setup(self):
        # initialize obervation
        self._rp(shape = [None]+list(self._ob_space.shape), name="observations");

        # Initialize the policies
        self._policy = policy =  self._policy_fn("policy", self._ob_space, self._ac_space, summaries = self._summaries)
        self._old_policy = old_policy = self._policy_fn("reference_pol", self._ob_space, self._ac_space, summaries = self._summaries, should_act = False)

        # Setup loss
        with tf.name_scope("loss"):
            total_loss = self._setup_loss();

        # Setup optimizer
        self._setup_optimizer();

        # Sync old and new policy
        with tf.name_scope("sync"):
            self._update_old = U.function([],[], updates=[tf.assign(oldv, newv)
                for (oldv, newv) in zip(self._old_policy.get_variables(), self._policy.get_variables())])


        self._session.__enter__();
        U.initialize()
        self._optimizer.sync();
        self.setup_summary_op();


    def setup_summary_op(self):
        if self._summaries:
            self._writer.add_graph(self._session.graph);
            self._generate_summaries([], True, True, True);
            self._summary_op = tf.summary.merge_all();
        else:
            self._summary_op = None;

    def _setup_loss(self):

        # Create advantage placeholder
        advantage = self._rp(shape =[None], name="advantages");

        # Calculate the log probability
        actions  = self._rp(
            placeholder = self._policy.pdtype.sample_placeholder([None],
                name="actions"
            ),
            name = "actions"
        );
        returns = self._rp(shape=[None], name="returns");

        with tf.name_scope("policy_loss"):
            log_prob = self._rr(tf.clip_by_value(self._policy.pd.logp(actions),-1e7, 1e7), name="log_probability");
            log_prob_old = self._rr(tf.clip_by_value(self._old_policy.pd.logp(actions),-1e7,1e7), name="log_probability_old");

            log_prob_diff = self._rr(tf.clip_by_value(log_prob - log_prob_old, -1, 1), "log_probability_difference")
            prob_ratio = self._rr(tf.exp(log_prob_diff), "probability_ratio");


            # Calculate surrogate loss (L^CPI)
            surrogate_loss = self._rr(prob_ratio * advantage, "surrogate_loss");

            # Calculate clipped surrogate loss
            clip_param = self._clip_param;
            clipped_prob_ratio = self._rr(tf.clip_by_value(prob_ratio, 1-clip_param, 1+clip_param), "clipped_probability_ratio");
            clipped_surrogate_loss = self._rr(clipped_prob_ratio * advantage, "clipped_surrogate_loss");

            # Calculate policy loss (L^CLIP)
            policy_loss = self._rr(- tf.reduce_mean(tf.minimum(surrogate_loss, clipped_surrogate_loss)), "policy_loss");

        with tf.name_scope("value_loss"):
            # Calculate value loss (squared error)
            predicted_value = self._rr(self._policy.vpred, "predicted_value");
            value_difference = self._rr(predicted_value - returns, "value_difference")
            value_loss = self._rr(self._value_param * tf.reduce_mean(tf.square(value_difference)), "value_loss");

        with tf.name_scope("entropy_loss"):
            # Calculate entropy loss
            entropy = self._rr(tf.reduce_mean(self._policy.pd.entropy()), "entropy");
            entropy_loss = self._rr(-self._entropy_param * entropy, "entropy_loss");

        # Calculate total loss
        total_loss = self._rr(policy_loss + value_loss + entropy_loss, "total_loss");

        return total_loss;

    def _setup_optimizer(self):
        # Calculate gradient of current policy
        var_list = self._policy.get_trainable_variables();
        with tf.name_scope("gradient"):
            gradient = self._rr(U.flatgrad(self.tf_loss, var_list, 100), "gradient");
            self._gradient = U.function(self._get_placeholders(names =[
                "observations",
                "actions",
                "advantages",
                "returns"
            ]), gradient);
        with tf.name_scope("optimizer"):
            self._optimizer = MpiAdam(var_list, epsilon = self._adam_epsilon);

    def _generate_summaries(self, summaries = [], results = False, placeholders = False, weights = False):
        if not self._summaries:
            return;
        assert isinstance(summaries, list), "summaries should be a list of result names";
        if results:
            summaries += list(self._tf_results.keys());

        if placeholders:
            summaries += list(self._tf_placeholders.keys());

        for name in summaries:
            assert name in self._tf_results or name in self._tf_placeholders, "{} should be in results for summary".format(name);
            if name in self._tf_results:
                tensor = self._tf_results[name];
            else:
                tensor = self._tf_placeholders[name];
            if len(tensor.shape) == 0:
                tf.summary.scalar(tensor.name, tensor);
            elif len(tensor.shape) == 1:
                tf.summary.histogram(tensor.name, tensor);
            elif len(tensor.shape) == 2:
                tf.summary.histogram(tensor.name, tensor);
            else:
                print("No summary:", name, tensor, tensor.shape, len(tensor.shape));

        if weights:
            for tensor in self._policy.get_trainable_variables():
                tf.summary.histogram(tensor.name, tensor);
        self._debug_summaries = [

        ];
        self._debug_summaries = tf.summary.merge([
            tf.summary.scalar("debug/log_std", tf.reduce_max(self._policy.pd.logstd), collections = [])
        ]);





    def _gp(self,*, name):
        assert name in self._tf_placeholders, "{} is not a registered placeholder".format(name);
        return self._tf_placeholders[name];

    def _get_placeholders (self, *, names):
        assert hasattr(names, "__iter__"), "Names should be iterable to get placeholders";
        return [self._gp(name = name) for name in names];

    def _rp(self, **kwargs):
        return self._register_placeholder(**kwargs);

    def _register_placeholder(self, *,
        name = None,
        dtype = None,
        shape = None,
        placeholder = None):
        if placeholder is None:
            placeholder = U.get_placeholder(name = name, dtype=tf.float32, shape=shape);
        elif name is None:
            name = placeholder.name;
        if name in self._tf_placeholders:
            raise ValueError("Placeholder with name {} already exists".format(name));
        self._tf_placeholders[name] = placeholder;
        return placeholder

    def _rr(self, tensor, name):
        tensor = tf.identity(tensor,name= name);
        return self._register_result(name = name, tensor=tensor);

    def _register_result(self, *,
        name = None,
        tensor = None):
        assert name is not None;
        assert tensor is not None;
        assert name not in self._tf_results;
        self._tf_results[name] = tensor;
        return tensor;

    def _get_feeddict(self, batch):
        placeholders = self._get_placeholders(names =[
            "observations",
            "actions",
            "advantages",
            "returns"
        ]);
        feed_dict = dict();
        for key, value in zip(placeholders,batch):
            feed_dict[key] = value;
        return feed_dict

    def calculate_results(self, *, names, batch=None):
        assert hasattr(names, "__iter__"), "Names should be iterable to get placeholders";
        if batch is None:
            batch = self._batch.get_batch(self._batch_size);
        feed_dict = self._get_feeddict(batch);
        results = [self._tf_results[name] for name in names]
        return np.array(tf.get_default_session().run(
            fetches= results,
            feed_dict= feed_dict));

    def reset(self, state):
        self._state = state;
        self._action = None;

    def observe(self, state, reward =None, terminal = None, reset = False):
        self._state = state;
        self._reward = reward;
        self._terminal = terminal;
        self._batch.observe(
            self._state,
            self._reward,
            self._terminal,
            self._predicted_value,
            self._action
        );
        self._observation +=1;
        if self._observation % int(self._batch.maxlen/2) ==0 and self._observation > self._batch.maxlen:
            self._learn();

    def multiply_entropy(self, f):
        tf.get_default_session().run(self._multiply_entropy, {self._multiply_placeholder:f});


    def _learn(self):
        self._update_old();
        self._batch._process_buffer();
        #print(self._batch.get_batch(100)[0]);
        new = self._batch.get_new();
        #self._policy.ob_rms.update(self._batch.get_new()[0]);
        loss_names = ['total_loss', 'policy_loss', 'value_loss', 'entropy_loss'];
        #print(fmt_row(13, loss_names+['mean_advantage','mean_value','gradient norm']));
        for _ in range(self._epochs):
            #losses = [];
            for batch in self._batch.iterate_batch(self._batch_size):
                gradient = self._gradient(*batch);
                self._optimizer.update(gradient, 1e-6);
                #losses.append(list(self.calculate_results(names = loss_names, batch= batch))+[np.mean(batch[2]),np.mean(batch[3]), np.linalg.norm(gradient)]);
            #print(fmt_row(13, np.mean(losses, axis=0)))
        if self._summaries and self._observation - self._last_summary > 3000:
            summaries = self._session.run(self._summary_op, feed_dict =self._get_feeddict(new) );
            self._writer.add_summary(summaries, self._observation);
            self._last_summary = self._observation;



    def act(self):
        self._prev_action = self._action;
        self._action, self._predicted_value = self._policy.act(True, self._state);
        #summaries = self._session.run(self._debug_summaries,
        #    feed_dict = {
        #        self._tf_placeholders['observations']: [self._state]
        #    });
        #self._writer.add_summary(summaries, self._observation);
        return self._action, self._predicted_value;

    def close(self):
        self._session.close();

    @property
    def tf_loss(self):
        return self._tf_results['total_loss'];

    def set_logdir(self, logdir, writer = None):
        self._summaries = logdir is not None;
        if self._summaries :
            self._writer = tf.summary.FileWriter(logdir) if writer is None else writer;
            self._last_summary = -3000;
        self.setup_summary_op();







if __name__ == "__main__":
    import gym;
    # Create a standard interact loop
    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,hid_size=64, num_hid_layers=2)

    env = gym.make("CartPole-v0");
    batch = BatchProvider(epochs = 1, horizon = 256);
    ppo = PPO(env, policy_fn, batch);
    terminal = True;
    iteration = 0;
    episode_lengths = [0];
    while True:
        if terminal:
            ppo.reset(env.reset());
            if iteration >= 1000:
                break;
            iteration += 1;
            if iteration % 10 == 0:
                print(iteration,":",np.mean(episode_lengths), np.std(episode_lengths));
                episode_lengths = [];
            episode_lengths.append(0)

        action, _ = ppo.act();
        state, reward, terminal, _ = env.step(action);
        ppo.observe(state, reward, terminal);
        episode_lengths[-1] +=1;
    env.close();
    ppo.close();
