import numpy as np
import tensorflow as tf
import time
from multiprocessing import Lock, Value


try:
    from .network import ActorCritic
except (SystemError, ImportError):
    from network import ActorCritic
try:
    from .data_generator import DataGenerator
except (SystemError, ImportError):
    from data_generator import DataGenerator


class SamplerAgent():
    def __init__(self, sampler:ActorCritic, data_generator:DataGenerator, sync_lock, val_lock, val_counter, weight_rolling, skip_word_cost,
                 discount, read_word_cost, scale_pred, scale_actor, scale_critic, update_after, embed_partial, positive_when_wrong, mode=1):
        self.sampler = sampler
        self.data_generator = data_generator
        self.train_generator = data_generator.get_sample_train

        #later change to a semaphor protected variable shared across all agents.
        self.runned_batches = 0

        self.read_word_cost = read_word_cost
        self.skip_word_cost = skip_word_cost
        self.weight_rolling = weight_rolling
        # not currently used
        self.discount=discount
        self.embed_partial = embed_partial
        self.positive_when_wrong = positive_when_wrong

        self.i = 0

        #mode we are currently in, currently only have 2 modes 1 (full read) and 2 (partial read)
        self.mode = mode

        #to avoid race condition when syncing with consumer
        self.sync_lock = sync_lock

        #variables for allowing validation to be run on unchanging network that is locked
        self.val_lock = val_lock
        self.val_counter = val_counter
        self.val_mode = False

        #scaling const
        self.scale_pred = np.array(scale_pred, dtype=np.float32)
        self.scale_critic = np.array(scale_critic, dtype=np.float32)
        self.scale_actor = np.array(scale_actor, dtype=np.float32)

        #delayed updates
        self.update_after = update_after


    def changeMode(self,mode):
        self.mode=mode

    def rolling_reward(self, action_1, action_2, read_words, original_length, prediction_correct):
        #run over samples to construct the reward,
        reward_at_1 = np.zeros(shape=action_1.shape, dtype=np.float32)
        reward_at_2 = np.zeros(shape=action_2.shape, dtype=np.float32)
        reward_at_end = np.zeros(shape=action_1.shape[1], dtype=np.float32)
        filter = np.zeros(shape=action_1.shape, dtype=np.float32)
        max_len = action_1.shape[0]
        for (sample_i, (length,ori_length)) in enumerate(zip(read_words,original_length)):
            current_reward = 0
            current_correct = prediction_correct[sample_i]
            if self.positive_when_wrong:
                flip = current_correct == 0
            else:
                flip = 0
            for l in range(length):
                reward_at_1[l,sample_i] = current_reward
                #agent 1 acts and gain a reward (here negative as its a cost)
                if action_1[l, sample_i] == self.sampler.agent_1_skip:
                    current_reward = current_reward - self.skip_word_cost/ori_length + 2*self.skip_word_cost/ori_length*flip
                else: #self.sampler.agent_1_read
                    current_reward = current_reward - self.read_word_cost/ori_length + 2*self.read_word_cost/ori_length*flip
                #agent 2 acts
                reward_at_2[l,sample_i] = current_reward
                #update reward based on action, currently no reward associated with any of the choices
                current_reward = current_reward
            reward_at_end[sample_i] = current_reward
            filter[:length,sample_i] = 1

        reward_at_1 = reward_at_1 * self.weight_rolling
        reward_at_2 = reward_at_2 * self.weight_rolling
        reward_at_end = reward_at_end * self.weight_rolling
        return reward_at_1,reward_at_2, reward_at_end, filter


    def prediction_correct(self, preds, y):
        current_prop = np.zeros(len(y))
        for i in range(len(y)):
            current_prop[i] = preds[i,int(y[i])]
        predicted = np.argmax(preds,axis=1)
        correct = predicted == y
        correct = np.ndarray.astype(correct, dtype=np.float32)
        reward_end = np.max([correct,current_prop],axis=0)
        reward_end = np.ndarray.astype(reward_end, dtype=np.float32)
        return correct, reward_end

    def compute_total_reward(self,is_not_done, action_1, action_2, preds, y, original_length):
        prediction_correct, reward_end = self.prediction_correct(preds, y)

        batch_size = action_1.shape[1]
        is_not_done = np.append(is_not_done, np.zeros((1, batch_size), dtype=np.bool), axis=0)
        time_length_reduced_input = np.argmin(is_not_done, axis=0)
        max_len = np.max(time_length_reduced_input)

        reduced_x = np.zeros(shape=(max_len + 1, batch_size), dtype=np.int32)
        reduced_action_1 = np.zeros(shape=(max_len, batch_size), dtype=np.int32)
        reduced_action_2 = np.zeros(shape=(max_len, batch_size), dtype=np.int32)
        for (i, max) in enumerate(time_length_reduced_input):
            reduced_action_1[:max, i] = action_1[:max, i]
            reduced_action_2[:max, i] = action_2[:max, i]

        rolling_reward_agent_1, rolling_reward_agent_2, reward_at_end, is_not_done = self.rolling_reward(
            reduced_action_1, reduced_action_2, time_length_reduced_input, original_length,reward_end)

        final_reward = reward_at_end + reward_end
        return final_reward, prediction_correct


    def enqueue_op(self,queue):
        def _func():
            #lock for stopping training during validation
            if self.val_mode:
                #main thread have locked the following lock, release when val is done
                self.val_lock.acquire(), self.val_lock.release()

            #signal we enter area that modify network
            with self.val_counter.get_lock():
                self.val_counter.value += 1

            #get batch
            this_x, this_y, this_comma, this_punctuation, this_addMax, mask, seqs, q_a_words = self.train_generator()

            if self.mode == 1:
                return_mode = 1
                pred_loss, pred, entropy_loss = self.sampler.consume_sample_full_read(this_x, seqs, this_y, self.scale_pred, q_a_words)

                prediction_correct, reward_pred = self.prediction_correct(pred, this_y)

                #signal we leave area that modify network
                with self.val_counter.get_lock():
                    self.val_counter.value -= 1

                return pred_loss, prediction_correct, np.zeros(1,dtype=np.float32),np.zeros(1,dtype=np.float32),\
                       np.zeros(1,dtype=np.float32),entropy_loss,np.zeros(1,dtype=np.float32)

            elif self.mode == 2:
                return_mode = 2
                batch_size = this_x.shape[1]

                self.i += 1
                if self.i % self.update_after == 0:
                    with self.sync_lock:
                        self.sampler.sync()

                read_words, action_agent_1, value_agent_1, action_agent_2, value_agent_2, predictions, is_not_done,\
                    probs_agent_1, probs_agent_2= \
                    self.sampler.get_sample(this_x, seqs, this_comma, this_punctuation, q_a_words)


                #construct new input to the training network, conisisting only of the read words
                is_not_done = np.append(is_not_done, np.zeros((1,batch_size),dtype=np.bool),axis=0)
                time_length_reduced_input = np.argmin(is_not_done,axis=0)
                max_len = np.max(time_length_reduced_input)
                #make them 1 longer to avoid out of bounds errors when looping in lstm
                reduced_x = np.zeros(shape=(max_len+1,batch_size),dtype=np.int32)
                reduced_action_1 = np.zeros(shape=(max_len,batch_size),dtype=np.int32)
                reduced_value_1 = np.zeros(shape=(max_len,batch_size),dtype=np.int32)
                reduced_action_2 = np.zeros(shape=(max_len,batch_size),dtype=np.int32)
                reduced_value_2 = np.zeros(shape=(max_len,batch_size),dtype=np.int32)
                reduced_probs_1 = np.zeros(shape=(max_len,batch_size),dtype=np.float32)
                reduced_probs_2 = np.zeros(shape=(max_len,batch_size),dtype=np.float32)

                number_skips = np.zeros(shape=(batch_size), dtype=np.int)

                for (i,max) in enumerate(time_length_reduced_input):
                    reduced_x[:max,i] = this_x[read_words[:max,i],i]
                    reduced_action_1[:max,i] = action_agent_1[:max,i]
                    reduced_value_1[:max,i] =  value_agent_1[:max,i]
                    reduced_action_2[:max,i] =  action_agent_2[:max,i]
                    reduced_value_2[:max,i] = value_agent_1[:max,i]
                    reduced_probs_1[:max,i] = probs_agent_1[:max,i]
                    reduced_probs_2[:max,i] = probs_agent_2[:max,i]
                    number_skips[i] = np.sum(action_agent_1[:max,i] == ActorCritic.agent_1_skip())

                #compute the reward

                prediction_correct, reward_pred = self.prediction_correct(predictions, this_y)
                rolling_reward_agent_1, rolling_reward_agent_2, reward_at_end, is_not_done = self.rolling_reward(reduced_action_1, reduced_action_2,time_length_reduced_input,seqs,
                                                                                                                 prediction_correct)

                #t_flip = (((prediction_correct == 0) * -1) + (prediction_correct == 1))
                #rolling_reward_agent_1 = rolling_reward_agent_1 * t_flip
                #rolling_reward_agent_2 = rolling_reward_agent_2 * t_flip
                #reward_at_end = reward_at_end * t_flip
                #print(rolling_reward_agent_1.shape,rolling_reward_agent_2.shape, reward_at_end.shape)

                final_reward = reward_at_end+reward_pred

                #if prediction correct, use is_not_done to also remove the updates in the agent for these
                #print(is_not_done*prediction_correct)
                #to_add_tmp = prediction_correct==0 * 0.5
                #is_not_done = is_not_done*(prediction_correct+to_add_tmp)


                #compute actual advantage:
                sampled_advantage_1 = final_reward - rolling_reward_agent_1
                sampled_advantage_2 = final_reward - rolling_reward_agent_2
                pred_loss, actor_loss, critic_loss, entropy_loss = self.sampler.consume_sample(reduced_x, time_length_reduced_input, sampled_advantage_1, reduced_action_1,
                                        sampled_advantage_2, reduced_action_2, is_not_done, this_y, final_reward,
                                                         self.scale_pred, self.scale_critic, self.scale_actor, q_a_words,
                                                            reduced_probs_1, reduced_probs_2, embedding_train= self.embed_partial)

                #general logging of behavior
                self.runned_batches +=1
                reading_percentage = (time_length_reduced_input) / seqs

                #start with advantage based directly on reward

                #signal we leave area that modify network
                with self.val_counter.get_lock():
                    self.val_counter.value -= 1

                return pred_loss, prediction_correct, np.ndarray.astype(reading_percentage,np.float32), actor_loss,\
                       critic_loss, entropy_loss, final_reward

        run_sample = tf.py_func(_func,[],[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]) #[tf.int32,tf.int32, tf.float32, tf.int32, tf.float32, tf.float32])
        return queue.enqueue(run_sample)


def start_lock(val_lock, agents):
    val_lock.acquire()
    for agent in agents:
        agent.val_mode = True

def end_lock(val_lock, val_counter, agents):
    while 1:
        if val_counter.value == len(agents):
            break
        else:
            time.sleep(1)
    for agent in agents:
        agent.val_mode = False
    val_lock.release()

def change_agents_mode(agents, mode):
    for agent in agents:
        agent.changeMode(mode)



#simple code for trying to run the agent
if __name__ == "__main__":
    dataset_name = "dbpedia_medium"
    folder_data = "data/"
    folder_result = "results/testing_folder/"

    batch_size = 50
    is_Q_A = False
    sync_lock = Lock()
    val_lock = Lock()
    val_counter = Value("i",0)

    NUMBER_THREADS = 3

    tf.reset_default_graph()

    dg = DataGenerator(dataset_name, folder_data, folder_result, batch_size, is_Q_A, w2v=None)#, w2v=None #no embedding, as they are slow to use when just debugging
    ac_consumer = ActorCritic(dg.batch_size, dg.vocab_size, dg.number_targets, scope_name="consumer", device="/gpu:0")
    samplers = [ActorCritic(dg.batch_size, dg.vocab_size, dg.number_targets, consumer=ac_consumer, scope_name="sampler_" +str(i), device="/cpu:0") for i in range(NUMBER_THREADS)]
    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))#log_device_placement=True))
    sess.run(init_op)
    ac_consumer.initialize(sess,dg.w2v_for_data)
    for sampler in samplers:
        sampler.initialize(sess)
    agents = [SamplerAgent(sampler, dg, sync_lock, val_lock, val_counter) for sampler in samplers]

    queue = tf.FIFOQueue(capacity=NUMBER_THREADS, dtypes=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])#[tf.int32, tf.int32, tf.float32, tf.int32, tf.float32, tf.float32], )
    qr = tf.train.QueueRunner(queue, [agent.enqueue_op(queue) for agent in agents])
    tf.train.queue_runner.add_queue_runner(qr)
    sample = queue.dequeue()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord, start=True)





