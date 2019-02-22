import tensorflow as tf
import numpy as np
from multiprocessing import Lock, Value
import time
import pickle
import sys

try:
    from .network import ActorCritic
except (SystemError, ImportError):
    from network import ActorCritic
try:
    from .data_generator import DataGenerator
except (SystemError, ImportError):
    from data_generator import DataGenerator
try:
    from .network_agent import SamplerAgent
except (SystemError, ImportError):
    from network_agent import SamplerAgent


def start_lock(val_lock, agents):
    val_lock.acquire()
    for agent in agents:
        agent.val_mode = True

def end_lock(val_lock, val_counter, agents, threads,sess, op, extrastuff):
    for agent in agents:
        agent.val_mode = False
    val_lock.release()

def change_agents_mode(agents, mode):
    for agent in agents:
        agent.changeMode(mode)

def compute_reading_percentage(is_not_done, x, action_1):
    skipped_words = np.sum(((action_1+1) * is_not_done) == 1,axis=0)
    read_words = np.sum(is_not_done, axis=0)
    read_words_skipped = read_words - skipped_words
    total_words = np.sum(x!=0,axis=0)
    return read_words/total_words, read_words_skipped/total_words

def agent_summaries(is_not_done, agent_1,agent_2):
    agent_1 = (agent_1 + 1) * is_not_done
    agent_2 = (agent_2 + 1) * is_not_done * (agent_1 == ActorCritic.agent_1_read()+1)
    agent_1_0 = np.sum(agent_1 == 1, axis=0)
    agent_1_1 = np.sum(agent_1 == 2, axis=0)

    agent_2_0 = np.sum(agent_2 == 1, axis=0)
    agent_2_1 = np.sum(agent_2 == 2, axis=0)
    agent_2_2 = np.sum(agent_2 == 3, axis=0)
    agent_2_3 = np.sum(agent_2 == 4, axis=0)

    agent_1_stats = np.stack((agent_1_0,agent_1_1),axis=1)
    agent_2_stats = np.stack((agent_2_0,agent_2_1,agent_2_2,agent_2_3),axis=1)


    return agent_1_stats, agent_2_stats


def pretrain_agent_and_critic(data_generator, consumer, agents, iterations):
    train_generator = data_generator.get_sample_train
    for i in range(iterations):
        this_x, this_y, this_comma, this_punctuation, this_addMax, mask, seqs, q_a_words = train_generator()

        batch_size = this_x.shape[1]

        read_words, action_agent_1, value_agent_1, action_agent_2, value_agent_2, predictions, is_not_done, \
        probs_agent_1, probs_agent_2 = \
            consumer.get_sample(this_x, seqs, this_comma, this_punctuation, q_a_words)

        # construct new input to the training network, conisisting only of the read words
        is_not_done_added = np.append(is_not_done, np.zeros((1, batch_size), dtype=np.bool), axis=0)
        time_length_reduced_input = np.argmin(is_not_done_added, axis=0)
        max_len = np.max(time_length_reduced_input)
        # make them 1 longer to avoid out of bounds errors when looping in lstm
        reduced_x = np.zeros(shape=(max_len + 1, batch_size), dtype=np.int32)
        reduced_action_1 = np.zeros(shape=(max_len, batch_size), dtype=np.int32)
        reduced_action_2 = np.zeros(shape=(max_len, batch_size), dtype=np.int32)

        number_skips = np.zeros(shape=(batch_size), dtype=np.int)

        for (i, max) in enumerate(time_length_reduced_input):
            reduced_x[:max, i] = this_x[read_words[:max, i], i]
            reduced_action_1[:max, i] = action_agent_1[:max, i]
            reduced_action_2[:max, i] = action_agent_2[:max, i]
            number_skips[i] = np.sum(action_agent_1[:max, i] == ActorCritic.agent_1_skip())

        prediction_correct = agents[0].prediction_correct(predictions, this_y)
        rolling_reward_agent_1, rolling_reward_agent_2, reward_at_end, is_not_done = agents[0].rolling_reward(
            reduced_action_1, reduced_action_2, time_length_reduced_input, seqs,prediction_correct)

        final_reward = reward_at_end + prediction_correct

        # compute actual advantage:
        sampled_advantage_1 = final_reward - rolling_reward_agent_1
        sampled_advantage_2 = final_reward - rolling_reward_agent_2

        #call special consume, only 2 losses
        consumer.consume_sample_init(reduced_x, time_length_reduced_input, sampled_advantage_1, reduced_action_1,
                                        sampled_advantage_2, reduced_action_2, is_not_done, agents[0].scale_critic, q_a_words, embedding_train=False)


def as_matrix(config):
    return [[k, str(w)] for k, w in config.items()]

def main(d_runner, d_agent, d_network, dataset_name, folder_data, folder_dg, folder_results, batch_size, is_Q_A, NUMBER_THREADS, summary_period,
         val_period, val_without_increase_max, partial_read_until, learning_rate,max_iteration, scale_entropy, data_cutoff,consumer_sampling_method,
         producer_sampling_method, init_runs, val_size, decrease_learning, initial_dist_agent_1, initial_dist_agent_2,
         explore_dist_agent_1, explore_dist_agent_2, partial_learning_rate):
    sync_lock = Lock()
    val_lock = Lock()
    val_counter = Value("i", 0)

    dg = DataGenerator(dataset_name, folder_data, folder_dg, batch_size, is_Q_A, val_size)  # , w2v=None #no embedding, as they are slow to use when just debugging

    tf.reset_default_graph()

    #global vars
    resetable_counter = tf.Variable(0, trainable=False)
    increase_counter = resetable_counter.assign(resetable_counter+1)
    reset_counter = resetable_counter.assign(0)

    #learning_rate = tf.train.polynomial_decay(learning_rate, resetable_counter, max_iteration // 2,
    #                                          learning_rate * decrease_learning) #currently is constant
    learning_rate = tf.Variable(learning_rate, dtype=tf.float32, trainable=False)
    placeholder_learning_rate = tf.placeholder(tf.float32, [])
    update_learning_rate = learning_rate.assign(placeholder_learning_rate)


    scale_entropy = tf.train.polynomial_decay(scale_entropy, resetable_counter, (max_iteration-partial_read_until) // 2,
                                              scale_entropy * 0.05)


    #make tensorflow vars for initial_dist
    initial_dist_agent_1 = tf.Variable(initial_dist_agent_1,trainable=False, dtype=tf.float32)
    initial_dist_agent_2 = tf.Variable(initial_dist_agent_2, trainable=False, dtype=tf.float32)
    placeholder_dist_agent_1 = tf.placeholder(tf.float32, [2])
    placeholder_dist_agent_2 = tf.placeholder(tf.float32, [4])
    set_dist_agent_1 = initial_dist_agent_1.assign(placeholder_dist_agent_1)
    set_dist_agent_2 = initial_dist_agent_2.assign(placeholder_dist_agent_2)

    def set_dists(sess, set_dist_1, set_dist_2):
        sess.run([set_dist_agent_1, set_dist_agent_2], {placeholder_dist_agent_1: set_dist_1,
                                                        placeholder_dist_agent_2: set_dist_2})


    #make networks
    ac_consumer = ActorCritic(dg.batch_size, dg.vocab_size, dg.number_targets, scope_name="consumer",
                              device="/gpu:0", **d_network, is_Q_A=is_Q_A, learning_rate=learning_rate, scale_entropy=scale_entropy,
                              sampling_method=consumer_sampling_method, initial_dist_agent_1=initial_dist_agent_1, initial_dist_agent_2=initial_dist_agent_2)

    samplers = [ActorCritic(dg.batch_size, dg.vocab_size, dg.number_targets, consumer=ac_consumer, is_Q_A=is_Q_A,
                            scope_name="sampler_" + str(i), device="/cpu:0", **d_network, sampling_method=producer_sampling_method,
                            initial_dist_agent_1=initial_dist_agent_1, initial_dist_agent_2=initial_dist_agent_2) for i in range(NUMBER_THREADS)]


    #initialise agents
    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=20, intra_op_parallelism_threads=0))  # log_device_placement=True))
    sess.run(init_op)
    ac_consumer.initialize(sess, dg.w2v_for_data)
    for sampler in samplers:
        sampler.initialize(sess)
    agents = [SamplerAgent(sampler, dg, sync_lock, val_lock, val_counter, **d_agent) for sampler in samplers]


    #start queues
    queue = tf.FIFOQueue(capacity=NUMBER_THREADS,
                         dtypes=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

    qr = tf.train.QueueRunner(queue, [agent.enqueue_op(queue) for agent in agents])
    tf.train.queue_runner.add_queue_runner(qr)
    sample = queue.dequeue()
    coord = tf.train.Coordinator()

    (N_pred_loss, N_prediction_correct, N_reading_percentage, N_actor_loss, \
     N_critic_loss, N_entropy_loss, N_final_reward) = sample



    #means for summaries
    log_reading = tf.reduce_mean(N_reading_percentage)
    log_pred = tf.reduce_mean(N_pred_loss)
    log_actor = tf.reduce_mean(N_actor_loss)
    log_critic = tf.reduce_mean(N_critic_loss)
    log_entropy = tf.reduce_mean(N_entropy_loss)
    log_reward = tf.reduce_mean(N_final_reward)
    log_acc = tf.reduce_mean(N_prediction_correct)


    #start logging
    tf.summary.scalar("prediction_loss", log_pred)
    tf.summary.scalar("scale_entropy", scale_entropy)
    tf.summary.scalar("actor_loss", log_actor)
    tf.summary.scalar("critic_loss", log_critic)
    tf.summary.scalar("entropy_loss", log_entropy)
    tf.summary.scalar("accuracy", log_acc)
    tf.summary.scalar("learning_rate", learning_rate)
    tf.summary.scalar("reading_percentage", log_reading)
    tf.summary.scalar("reward", log_reward)
    tf.summary.scalar("resetable_counter", resetable_counter)
    summary_op = tf.summary.merge_all()

    config_runner_summary = tf.summary.text('TrainConfig_runner', tf.convert_to_tensor(as_matrix(d_runner)), collections=[])
    config_agent_summary = tf.summary.text('TrainConfig_agent', tf.convert_to_tensor(as_matrix(d_agent)), collections=[])
    config_network_summary = tf.summary.text('TrainConfig_network', tf.convert_to_tensor(as_matrix(d_network)), collections=[])
    summary_config_op = tf.summary.merge([config_runner_summary,config_agent_summary,config_network_summary])

    #need queue size for flushing between modes
    queue_size = queue.size()


    #initilisation of summary stuff
    summary_writer = tf.summary.FileWriter(folder_results + "overview/", sess.graph)
    summary_writer.add_summary(sess.run(summary_config_op))

    #savers
    saver = tf.train.Saver()#var_list=ac_consumer.train_vars)
    '''
    if "CN" in dataset_name:
        #pass
        print("###---- trying to load")
        fullread_folder = "C:\\Users\\casper\\Dropbox\\PHDwork-casper\\mypapers\\adaptiveLengthLSTM\\new\\realproject\\new_code\\casperVis\\results_46\\_CN_onetext_0.001_49000_128_2_0.2_1_32_0.0001_FULL_1\\"
        #new_saver = tf.train.import_meta_graph(folder_results + "fullread/")
        saver.restore(sess, fullread_folder+"fullread/best_val_model_full")  
        print("\n####---- loaded", dataset_name, "pretrained\n")
        pretrain_agent_and_critic(dg, ac_consumer, agents, init_runs)
        set_dists(sess, explore_dist_agent_1, explore_dist_agent_2)
        sess.run(update_learning_rate, {placeholder_learning_rate : partial_learning_rate})

        change_agents_mode(agents,2)
    '''

    if use_pretrained:
        print("###---- trying to load")
        if d_network['include_entropy_in_full']:
        	fullread_folder = "pretrained/"+dataset_name+"/w_entropy/best_val_model_full"
        else:
        	fullread_folder = "pretrained/"+dataset_name+"/no_entropy/best_val_model_full"
        saver.restore(sess, fullread_folder)  
        print("\n####---- loaded", dataset_name, "pretrained\n")

        #pretrain_agent_and_critic(dg, ac_consumer, agents, init_runs)
        #set_dists(sess, explore_dist_agent_1, explore_dist_agent_2)
        #sess.run(update_learning_rate, {placeholder_learning_rate : partial_learning_rate})
        #change_agents_mode(agents,2)


    #finalise graph
    sess.graph.finalize()


    threads = tf.train.start_queue_runners(sess=sess, coord=coord, start=True)


    #vars for looping
    i=0
    empty_mode_val = False
    empty_mode_switch = False
    full_read = True
    best_val_reward = -np.infty
    best_full_correct = -np.infty
    val_no_increases = 0
    results = []
    start_time = time.time()
    while i < max_iteration:
        i += 1

        if full_read:
            _s, summary_str = sess.run([sample, summary_op])
        else:
            _s, summary_str, _ = sess.run([sample, summary_op, increase_counter])
        #make summaries
        if i% summary_period == 0:
            summary_writer.add_summary(summary_str, i)

        if i==partial_read_until:
            if not empty_mode_val:
                start_lock(val_lock, agents)
            empty_mode_switch = True
            full_read = False

        if i%val_period==0:
            if not empty_mode_switch:
                start_lock(val_lock, agents)
            empty_mode_val = True

        if empty_mode_val:
            if  val_counter.value == 0:
                print(time.time()-start_time, "time")
                #start train mode
                ac_consumer.set_test(sess)
                reconstruct_all = []
                updated_best = False
                for t in ["val", "test"]:
                    gen = dg.get_batch(t)

                    running_reward = 0
                    running_correct = 0
                    running_reading_percentage = 0
                    running_reading_percentage_skipped = 0
                    running_agent_1_stats = np.zeros(2)
                    running_agent_2_stats = np.zeros(4)
                    to_divide = 0
                    reconstruct_sentences = []
                    while 1:
                        gen_sample = gen()
                        if gen_sample is not None:
                            this_x, this_y, this_comma, this_punctuation, this_addMax, mask, seqs, q_a_words = gen_sample
                            mask = mask == 1

                            if not full_read:
                                read_words, action_agent_1, value_agent_1, action_agent_2, value_agent_2, predictions, is_not_done,_,_ = \
                                    ac_consumer.get_sample(this_x, seqs, this_comma, this_punctuation, q_a_words)

                                reconstruct_sentences.append((this_x,read_words,action_agent_1,action_agent_2,mask))

                                total_reward,correct = agents[0].compute_total_reward(is_not_done, action_agent_1, value_agent_2, predictions, this_y, seqs)
                                reading_percentage, reading_percentage_skipped = compute_reading_percentage(is_not_done, this_x, action_agent_1)
                                agent_1_stats, agent_2_stats = agent_summaries(is_not_done, action_agent_1, action_agent_2)

                                running_reward += np.sum(total_reward[mask])
                                running_correct += np.sum(correct[mask])
                                running_reading_percentage += np.sum(reading_percentage[mask])
                                running_reading_percentage_skipped += np.sum(reading_percentage_skipped[mask])
                                running_agent_1_stats += np.sum(agent_1_stats[mask],axis=0)
                                running_agent_2_stats += np.sum(agent_2_stats[mask],axis=0)
                                to_divide += np.sum(mask)

                            else:
                                predictions = ac_consumer.full_read(this_x,seqs, q_a_words)
                                predictions = predictions[0]
                                correct, reward_pred = agents[0].prediction_correct(predictions,this_y)
                                running_correct += np.sum(correct[mask])
                                to_divide += np.sum(mask)
                        else:
                            running_reward /= to_divide
                            running_correct /= to_divide
                            running_reading_percentage /= to_divide
                            running_reading_percentage_skipped /= to_divide
                            running_agent_1_stats /= to_divide
                            running_agent_2_stats /= to_divide

                            toSave = (t, i, running_reward, running_correct, running_reading_percentage, running_reading_percentage_skipped, running_agent_1_stats,
                                  running_agent_2_stats)
                            print(toSave)
                            print("##",t,i)
                            results.append(toSave)
                            reconstruct_all.append((t,reconstruct_sentences))
                            if t == "val" and full_read:
                                if running_correct>best_full_correct:
                                    print(" --##-- updating best model full read")
                                    best_full_correct = running_correct
                                    saver.save(sess, folder_results + "fullread/best_val_model_full")
                                    print(" --##-- saved best model full read")
                            if t == "val" and not full_read:
                                if running_reward>best_val_reward:
                                    updated_best=True
                                    val_no_increases = 0
                                    print(" --##-- updating best model")
                                    best_val_reward = running_reward
                                    #saver.save(sess, folder_results + "partial/best_val_model")
                                    print(" --##-- saved best model")
                                else:
                                    val_no_increases += 1

                                if val_no_increases >= val_without_increase_max:
                                    i=max_iteration
                            break
                    pickle.dump(results, open(folder_results + "overview/results", "wb"))
                    start_time = time.time()

                if updated_best:
                    pickle.dump((i, reconstruct_all), open(folder_results + "reconstruct", "wb"))
                #end test mode
                ac_consumer.set_dropout(sess)
                #continue training MUST BE AT END OF IF
                if not empty_mode_switch:
                    end_lock(val_lock, val_counter, agents, threads, sess, queue_size, [empty_mode_switch, empty_mode_val, partial_read_until, i,full_read])
                empty_mode_val = False


        if empty_mode_switch:
            if val_counter.value == 0:
                #can only change mode after a validation currently
                pretrain_agent_and_critic(dg, ac_consumer, agents, init_runs)
                set_dists(sess, explore_dist_agent_1, explore_dist_agent_2)
                sess.run(update_learning_rate, {placeholder_learning_rate : partial_learning_rate})

                change_agents_mode(agents,2)
                print("changed to partial read")
                end_lock(val_lock, val_counter, agents, threads, sess, queue_size, [])
                empty_mode_switch = False

    coord.request_stop()
    coord.join(threads)

    dg.empty_train()


def run_dict():
    d_runner = {
        #"dataset_name" : "CN_onetext",
        "folder_data" : "data/",
        "folder_dg" : "split_folder/",
        #"folder_results" : "results/tmp/",
        "batch_size" : 100,
        "is_Q_A" : False,
        "NUMBER_THREADS" : 1,
        "summary_period": 1,
        "val_period" : 200,
        "val_without_increase_max" : 100,
        "learning_rate": 0.0005,
        "max_iteration": 31000,
        "scale_entropy": 0.1,
        "data_cutoff": 100,
        "consumer_sampling_method" : "greedy",
        "producer_sampling_method" : "normal",
        "init_runs" : 0,
        "val_size" : 0.15,
        "decrease_learning": 1,
        "initial_dist_agent_1" : [0.0, 1.0],
        "initial_dist_agent_2" : [1.0, 0.0, 0.0, 0.0],
        "explore_dist_agent_1" : [0.5, 0.5],
        "explore_dist_agent_2": [0.25, 0.25, 0.25, 0.25],
        "partial_learning_rate": 0.0005
    }
    d_agent = {
        #"weight_rolling" : 0.1,
        "skip_word_cost" : 0.5,
        "discount" : 1,
        "read_word_cost" : 1,
        "scale_pred" : 1,
        "scale_actor" : 10,
        "scale_critic" : 1,
        "update_after": 1,
        "embed_partial" : False,
        "positive_when_wrong": False
    }
    d_network = {
        "attention" : False,
        "grad_clip" : 0.1,
        "cell_size" : 128,
        "size_state_1" : 25,
        "size_state_2" : 25,
        "embedding_dropout" : 0.9,
        "rnn_dropout" : 0.9,
        "n_stacks" : 1,
        "trainable_embedding" : True,
        "off_policy": True,
        "name_optimizer": "RMSE",
        "skim_cells": 0,
        "include_entropy_in_full" : True,
    }

    return d_runner, d_agent, d_network


def is_data_Q_A(dataset_name):
    if "CN" in dataset_name or "NE" in dataset_name:
        return True
    else:
        return False


if __name__ == "__main__":

    dataset_name = "dbpedia_medium"
    partial_read_until = 30000
    scale_entropy = 0.1
    weight_rolling = 0.1
    use_pretrained = False

    d_runner, d_agent, d_network = run_dict()

    d_runner["folder_results"] = "results/" + "".join(["_" + str(t) for t in [dataset_name,partial_read_until,scale_entropy,weight_rolling]]) + "/"
    d_runner["partial_read_until"] = partial_read_until
    d_runner["scale_entropy"] = scale_entropy
    d_runner["dataset_name"] = dataset_name
    d_agent["weight_rolling"] = weight_rolling
    d_network["trainable_embedding"] = not is_data_Q_A(dataset_name)


    main(d_runner, d_agent, d_network,**d_runner)