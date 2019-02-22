import tensorflow as tf
import numpy as np

# Inspired heavily by https://github.com/hiwonjoon/tf-a3c-gpu, and code copied directly when possible

EPSILON = 1e-7


class ActorCritic():
    # produced the samples for consumer
    def _build_rnn_sampler(self, embedded, time_length, comma, punctuation, q_a_words_embedded):
        with tf.variable_scope(self.scope.name, reuse=tf.AUTO_REUSE) as scope:

            cell = self._make_rnn_cell()
            range_to_stack = tf.reshape(tf.range(self.batch_size), shape=(-1,))

            def loop_fn(time, cell_output, cell_state, loop_state):
                if cell_output is None:  # time == 0
                    cell_state = cell.zero_state(self.batch_size, tf.float32)
                    (_, cell_output) = cell_state[-1]

                    next_index = tf.zeros(self.batch_size, dtype=tf.int32)
                    next_elements_finished = next_index >= time_length

                    output = [cell_output[0]] + [tf.zeros(1, dtype=t) for t in [tf.int32, tf.bool,
                                                                                tf.int32, tf.float32, tf.int32,
                                                                                tf.float32,
                                                                                tf.float32, tf.float32]]

                    action_agent_1 = tf.constant(self.agent_1_read, dtype=tf.int32, shape=(self.batch_size,))
                    action_agent_2 = tf.constant(self.agent_2_word, dtype=tf.int32, shape=(self.batch_size,))
                else:
                    previus_index = loop_state[0]
                    previus_input = loop_state[1]
                    previus_cell_state = loop_state[2]
                    previus_output = loop_state[3]

                    # AGENT 1
                    # calculate value and policy
                    state_for_agent_1 = self._build_state_1(previus_cell_state, previus_input)
                    value_agent_1 = self._build_value_1(state_for_agent_1)
                    logits_agent_1 = self._build_policy_1(state_for_agent_1)

                    # get an actual action from policy, either 0 == skip, 1 == read
                    action_agent_1 = self.sample_fn(logits_agent_1)
                    prob_agent_1 = self._probability_of_action(logits_agent_1, action_agent_1, range_to_stack)

                    cell_state, cell_output = self.apply_action_1(action_agent_1, cell_state, previus_cell_state,
                                                                  previus_input)

                    # AGENT 2
                    state_for_agent_2 = self._build_state_2(cell_state)
                    value_for_agent_2 = self._build_value_2(state_for_agent_2)
                    logits_agent_2 = self._build_policy_2(state_for_agent_2)

                    # get action agent 2
                    action_agent_2 = self.sample_fn(logits_agent_2)
                    prob_agent_2 = self._probability_of_action(logits_agent_2, action_agent_2, range_to_stack)

                    # update steps based on agent 1 and 2, agent_2: 0->next, 1->comma, 2->punct, 3->end doc. If agent=0 agent 2 cant act
                    index_for_jumps = tf.stack([previus_index, range_to_stack], axis=1)

                    # if action is
                    action_agent_2 = tf.where(tf.equal(action_agent_1, self.agent_1_skip),
                                              tf.constant(self.agent_2_word,
                                                          dtype=tf.int32, shape=(self.batch_size,)), action_agent_2)

                    add_to_index = tf.ones(self.batch_size, dtype=tf.int32)  # self.agent_2_word
                    add_to_index = tf.where(tf.equal(action_agent_2, self.agent_2_comma),
                                            tf.squeeze(tf.gather_nd(comma, index_for_jumps)), add_to_index)
                    add_to_index = tf.where(tf.equal(action_agent_2, self.agent_2_punct),
                                            tf.squeeze(tf.gather_nd(punctuation, index_for_jumps)), add_to_index)
                    add_to_index = tf.where(tf.equal(action_agent_2, self.agent_2_end), time_length, add_to_index)
                    # add_to_index = tf.where(tf.equal(action_agent_1,self.agent_1_read), add_to_index, tf.ones(self.batch_size,dtype=tf.int32))

                    next_index = previus_index + add_to_index
                    next_index = tf.where(next_index > time_length, time_length, next_index)
                    next_elements_finished = next_index >= time_length

                    # construct output, is the word, and value/actions assoicaited with the given word
                    output = [cell_output, tf.expand_dims(previus_index, axis=1),
                              tf.expand_dims(previus_index < time_length, axis=1),
                              tf.expand_dims(action_agent_1, axis=1), value_agent_1,
                              tf.expand_dims(action_agent_2, axis=1), value_for_agent_2,
                              tf.expand_dims(prob_agent_1, axis=1), tf.expand_dims(prob_agent_2, axis=1)
                              ]

                index_range = tf.stack([next_index, range_to_stack], axis=1)
                next_input_word = tf.gather_nd(embedded, index_range)

                next_input = tf.concat([next_input_word, tf.one_hot(action_agent_1, self.number_actions_1),
                                        tf.one_hot(action_agent_2, self.number_actions_2)], axis=1)

                next_loop_state = (
                    next_index,
                    next_input,
                    cell_state,
                    cell_output
                )
                return (next_elements_finished, next_input, cell_state, output, next_loop_state)

            outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
            cell_outputs = outputs_ta[0].stack()
            read_words = outputs_ta[1].stack()
            is_not_done = outputs_ta[2].stack()
            action_agent_1 = outputs_ta[3].stack()
            value_agent_1 = outputs_ta[4].stack()
            action_agent_2 = outputs_ta[5].stack()
            value_agent_2 = outputs_ta[6].stack()
            probs_agent_1 = outputs_ta[7].stack()
            probs_agent_2 = outputs_ta[8].stack()

            rnn_state = self._cell_ouputs_to_state(cell_outputs, final_state, is_not_done, action_agent_1)

            output_logits = self._make_output_logits(rnn_state, q_a_words_embedded)

            probs = tf.nn.softmax(output_logits)

            return tf.squeeze(read_words), tf.squeeze(action_agent_1), tf.squeeze(value_agent_1), tf.squeeze(
                action_agent_2), \
                   tf.squeeze(value_agent_2), probs, tf.squeeze(is_not_done), tf.squeeze(probs_agent_1), tf.squeeze(
                probs_agent_2)

    # train using samples from sampler
    # try rewriting with tensor arrays to compare time
    def _build_rnn_consumer(self, embedded, time_length, advantage_1, action_1, advantage_2, action_2, is_not_done,
                            target, reward, scale_pred, scale_critic, scale_actor, scale_entropy, q_a_words_embedded,
                            sampler_probs_agent_1, sampler_probs_agent_2):
        with tf.variable_scope(self.scope.name, reuse=tf.AUTO_REUSE) as scope:
            cell = self._make_rnn_cell()
            range_to_stack = tf.reshape(tf.range(self.batch_size), shape=(-1,))
            max_time = tf.shape(embedded)[0] - 1  # embedded is 1 longer to avoid out of bounds

            action_1_ta = tf.TensorArray(dtype=tf.int32, size=max_time)
            action_1_ta = action_1_ta.unstack(action_1)

            action_2_ta = tf.TensorArray(dtype=tf.int32, size=max_time)
            action_2_ta = action_2_ta.unstack(action_2)

            def loop_fn(time, cell_output, cell_state, loop_state):
                if cell_output is None:  # time == 0
                    cell_state = cell.zero_state(self.batch_size, tf.float32)
                    (_, cell_output) = cell_state[-1]

                    next_index = tf.zeros(self.batch_size, dtype=tf.int32)
                    next_elements_finished = next_index >= time_length

                    action_agent_1 = tf.constant(self.agent_1_read, dtype=tf.int32, shape=(self.batch_size,))
                    action_agent_2 = tf.constant(self.agent_2_word, dtype=tf.int32, shape=(self.batch_size,))

                    output = [cell_output[0]] + [tf.zeros(s, dtype=t) for (s, t) in [(1, tf.float32),
                                                                                     (
                                                                                     self.number_actions_1, tf.float32),
                                                                                     (1, tf.float32),
                                                                                     (
                                                                                     self.number_actions_2, tf.float32),
                                                                                     (1, tf.float32), (1, tf.float32)]]
                else:
                    read_at = time - 1

                    previus_index = loop_state[0]
                    previus_input = loop_state[1]
                    previus_cell_state = loop_state[2]
                    previus_output = loop_state[3]

                    # agent 1
                    state_for_agent_1 = self._build_state_1(previus_cell_state, previus_input)
                    value_agent_1 = self._build_value_1(state_for_agent_1)
                    logits_agent_1 = self._build_policy_1(state_for_agent_1)
                    action_agent_1 = action_1_ta.read(read_at)
                    probs_agent_1 = self._probability_of_action(logits_agent_1, action_agent_1, range_to_stack)

                    # update cell state based on action
                    cell_state, cell_output = self.apply_action_1(action_agent_1, cell_state, previus_cell_state,
                                                                  previus_input)

                    # AGENT 2
                    state_for_agent_2 = self._build_state_2(cell_state)
                    value_for_agent_2 = self._build_value_2(state_for_agent_2)
                    logits_agent_2 = self._build_policy_2(state_for_agent_2)
                    action_agent_2 = action_2_ta.read(read_at)
                    probs_agent_2 = self._probability_of_action(logits_agent_2, action_agent_2, range_to_stack)

                    if self.off_policy:
                        probs_agent_1 = tf.stop_gradient(probs_agent_1)
                        probs_agent_2 = tf.stop_gradient(probs_agent_2)

                    next_index = previus_index + 1
                    next_elements_finished = next_index >= time_length

                    output = [cell_output, value_agent_1, logits_agent_1, value_for_agent_2, logits_agent_2,
                              tf.expand_dims(probs_agent_1, axis=-1),
                              tf.expand_dims(probs_agent_2, axis=-1)]

                index_range = tf.stack([next_index, range_to_stack], axis=1)
                next_input_word = tf.gather_nd(embedded, index_range)

                next_input = tf.concat([next_input_word, tf.one_hot(action_agent_1, self.number_actions_1),
                                        tf.one_hot(action_agent_2, self.number_actions_2)], axis=1)

                next_loop_state = (
                    next_index,
                    next_input,
                    cell_state,
                    cell_output
                )
                return (next_elements_finished, next_input, cell_state, output, next_loop_state)

            outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)

            cell_outputs = outputs_ta[0].stack()
            value_agent_1 = tf.squeeze(outputs_ta[1].stack())
            logits_agent_1 = outputs_ta[2].stack()
            value_agent_2 = tf.squeeze(outputs_ta[3].stack())
            logits_agent_2 = outputs_ta[4].stack()
            probs_agent_1 = tf.squeeze(outputs_ta[5].stack())
            probs_agent_2 = tf.squeeze(outputs_ta[6].stack())

            rnn_state = self._cell_ouputs_to_state(cell_outputs, final_state, tf.expand_dims(is_not_done, dim=-1),
                                                   tf.expand_dims(action_1, dim=-1))
            output_logits = self._make_output_logits(rnn_state, q_a_words_embedded)

            # losses

            # prediction
            pred_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=output_logits)
            pred_loss = tf.reduce_mean(pred_loss)
            pred = tf.nn.softmax(output_logits)

            # value losses
            filter_1 = is_not_done

            sum_filter_1 = tf.cast(tf.reduce_sum(filter_1), dtype=tf.float32)

            value_loss_1 = tf.losses.mean_squared_error(labels=advantage_1, predictions=value_agent_1,
                                                        weights=filter_1)
            # value_loss_1 = value_loss_1 / sum_filter_1

            filter_2 = is_not_done * action_1
            sum_filter_2 = tf.cast(tf.reduce_sum(filter_2), dtype=tf.float32)

            value_loss_2 = tf.losses.mean_squared_error(labels=advantage_2, predictions=value_agent_2,
                                                        weights=filter_2)
            # value_loss_2 = value_loss_2 / sum_filter_2

            value_loss = value_loss_1 + value_loss_2

            filter_1 = tf.cast(filter_1, dtype=tf.float32)
            filter_2 = tf.cast(filter_2, dtype=tf.float32)

            # policy losses
            # the agent loss is scaled with R-V*
            approx_advantage_1 = reward - value_agent_1
            approx_advantage_2 = reward - value_agent_2
            approx_advantage_1 = tf.stop_gradient(approx_advantage_1)
            approx_advantage_2 = tf.stop_gradient(approx_advantage_2)

            if self.off_policy:
                off_factor_1 = probs_agent_1 / (sampler_probs_agent_1 + EPSILON)
                off_factor_2 = probs_agent_2 / (sampler_probs_agent_2 + EPSILON)

                w_1 = filter_1 * approx_advantage_1 * off_factor_1
                w_2 = filter_2 * approx_advantage_2 * off_factor_2
            else:
                w_1 = filter_1 * approx_advantage_1
                w_2 = filter_2 * approx_advantage_2

            # if self.off_policy_stop_grad:
            #    w_1 = tf.stop_gradient(w_1)
            #    w_2 = tf.stop_gradient(w_2)


            logits_loss_1 = tf.losses.sparse_softmax_cross_entropy(labels=action_1, logits=logits_agent_1,
                                                                   weights=w_1)
            logits_loss_2 = tf.losses.sparse_softmax_cross_entropy(labels=action_2, logits=logits_agent_2,
                                                                   weights=w_2)

            logits_loss = logits_loss_1 + logits_loss_2

            # entropy_loss_1 = tf.reduce_sum(tf.reduce_sum(tf.nn.softmax(logits_agent_1+EPSILON) * tf.nn.log_softmax(logits_agent_1+EPSILON)) * filter_1) / sum_filter_1
            # entropy_loss_2 = tf.reduce_sum(tf.reduce_sum(tf.nn.softmax(logits_agent_2+EPSILON) * tf.nn.log_softmax(logits_agent_2+EPSILON)) * filter_2) / sum_filter_2


            # entropy loss with target distribution, chosen based on data set charistica
            initial_dist_agent_1_tf = tf.zeros((max_time, self.batch_size, 2)) + self.initial_dist_agent_1
            entropy_loss_1 = tf.reduce_sum(tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=initial_dist_agent_1_tf, logits=logits_agent_1),
                axis=-1) * filter_1) / sum_filter_1
            initial_dist_agent_2_tf = tf.zeros((max_time, self.batch_size, 4)) + self.initial_dist_agent_2
            entropy_loss_2 = tf.reduce_sum(tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=initial_dist_agent_2_tf, logits=logits_agent_2),
                axis=-1) * filter_2) / sum_filter_2

            entropy_loss = entropy_loss_1 + entropy_loss_2

            pred_loss = pred_loss * scale_pred
            value_loss = value_loss * scale_critic
            logits_loss = logits_loss * scale_actor
            entropy_loss = entropy_loss * scale_entropy

            summed_loss = pred_loss + value_loss + logits_loss + entropy_loss

            return summed_loss, pred_loss, logits_loss, value_loss, entropy_loss, pred, 0

    def _build_rnn_full_reader(self, embedded, time_length, q_a_words_embedded):
        with tf.variable_scope(self.scope.name, reuse=tf.AUTO_REUSE) as scope:

            cell = self._make_rnn_cell()
            range_to_stack = tf.reshape(tf.range(self.batch_size), shape=(-1,))

            def loop_fn(time, cell_output, cell_state, loop_state):
                if cell_output is None:  # time == 0
                    cell_state = cell.zero_state(self.batch_size, tf.float32)
                    (_, cell_output) = cell_state[-1]

                    next_index = tf.zeros(self.batch_size, dtype=tf.int32)
                    next_elements_finished = next_index >= time_length

                    action_agent_1 = tf.constant(self.agent_1_read, dtype=tf.int32, shape=(self.batch_size,))
                    action_agent_2 = tf.constant(self.agent_2_word, dtype=tf.int32, shape=(self.batch_size,))

                    output = [cell_output[0]] + [tf.zeros(1, dtype=t) for t in [tf.bool, tf.int32]]
                else:
                    previus_index = loop_state

                    # cell_state, cell_output = self.apply_dropout_lstm(cell_state, cell_output)

                    next_index = previus_index + tf.ones(self.batch_size, dtype=tf.int32)

                    # dropout for internal state
                    # NO DROPOUT RIGHT NOW, WILL BE MADe LATER GENERAL FOR STACKED RNN


                    next_index = tf.where(next_index > time_length, time_length, next_index)

                    next_elements_finished = next_index >= time_length

                    # construct output, is the word, and value/actions assoicaited with the given word
                    action_agent_1 = tf.constant(self.agent_1_read, dtype=tf.int32, shape=(self.batch_size,))
                    action_agent_2 = tf.constant(self.agent_2_word, dtype=tf.int32, shape=(self.batch_size,))

                    output = [cell_output, tf.expand_dims(previus_index < time_length, axis=1),
                              tf.expand_dims(action_agent_1, axis=1)]

                index_range = tf.stack([next_index, range_to_stack], axis=1)
                next_input_word = tf.gather_nd(embedded, index_range)

                next_input = tf.concat([next_input_word, tf.one_hot(action_agent_1, self.number_actions_1),
                                        tf.one_hot(action_agent_2, self.number_actions_2)], axis=1)

                # next_input = tf.concat([next_input_word,tf.one_hot(action_agent_1,self.number_actions_1), tf.one_hot(action_agent_2,self.number_actions_2)],axis=1)

                next_loop_state = (
                    next_index
                )
                return (next_elements_finished, next_input, cell_state, output, next_loop_state)

            outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
            cell_outputs = outputs_ta[0].stack()
            is_not_done = outputs_ta[1].stack()
            action_agent_1 = outputs_ta[2].stack()

            rnn_state = self._cell_ouputs_to_state(cell_outputs, final_state, is_not_done, action_agent_1)

            output_logits = self._make_output_logits(rnn_state, q_a_words_embedded)
            probs = tf.nn.softmax(output_logits)

            return probs

    def _make_rnn_cell(self):
        cells = [tf.contrib.rnn.LSTMCell(self.cell_size, name="LSTMCELL" + str(i)) for i in range(self.n_stacks)]
        stacked = tf.nn.rnn_cell.MultiRNNCell(cells)
        return stacked

    def _make_small_rnn_cell(self):
        cells = [tf.contrib.rnn.LSTMCell(self.skim_cells, name="LSTM_SMALL_" + str(i)) for i in range(self.n_stacks)]
        return cells

    def apply_action_1(self, action_agent_1, cell_state, previus_cell_state, previus_input):
        # in case of full jump and no skim cells
        # cell_state, _ = self.apply_dropout_lstm(cell_state,None)
        if self.skim_cells == 0:
            new_cell_state = []
            for i in range(self.n_stacks):
                (t_internal, t_output) = cell_state[i]
                (t_prev_internal, t_prev_output) = previus_cell_state[i]
                k_output = tf.where(tf.equal(action_agent_1, self.agent_1_read), t_output, t_prev_output)
                k_internal = tf.where(tf.equal(action_agent_1, self.agent_1_read), t_internal,
                                      t_prev_internal)
                new_cell_state.append(tf.contrib.rnn.LSTMStateTuple(k_internal, k_output))
            cell_state = tuple(new_cell_state)
            (_, output) = cell_state[-1]
        elif self.skim_cells > 0 and self.n_stacks == 1:
            new_cell_state = []
            for i in range(self.n_stacks):
                (t_internal, t_output) = cell_state[i]
                (t_prev_internal, t_prev_output) = previus_cell_state[i]

                small_internal_state = t_prev_internal[:, :self.skim_cells]
                small_output_state = t_prev_output[:, :self.skim_cells]
                small_state = tf.contrib.rnn.LSTMStateTuple(small_internal_state, small_output_state)
                input = tf.concat([t_prev_output[:, self.skim_cells:], previus_input], axis=1)
                # input = previus_input
                _, updated_small = self.small_rnns[i].__call__(input, small_state)
                (updated_small_internal, updated_small_output) = updated_small

                combined_new_internal = tf.concat([updated_small_internal, t_prev_internal[:, self.skim_cells:]],
                                                  axis=1)
                combined_new_out = tf.concat([updated_small_output, t_prev_output[:, self.skim_cells:]], axis=1)

                k_output = tf.where(tf.equal(action_agent_1, self.agent_1_read), t_output, combined_new_out)
                k_internal = tf.where(tf.equal(action_agent_1, self.agent_1_read), t_internal,
                                      combined_new_internal)
                new_cell_state.append(tf.contrib.rnn.LSTMStateTuple(k_internal, k_output))
                '''
                combined_internal = tf.concat((t_internal[:, :self.skim_cells], t_prev_internal[:, self.skim_cells:]),
                                              axis=1)
                combined_output = tf.concat((t_output[:, :self.skim_cells], t_prev_output[:, self.skim_cells:]),
                                              axis=1)
                k_output = tf.where(tf.equal(action_agent_1, self.agent_1_read), t_output, combined_output)
                k_internal = tf.where(tf.equal(action_agent_1, self.agent_1_read), t_internal,
                                      combined_internal)
                new_cell_state.append(tf.contrib.rnn.LSTMStateTuple(k_internal, k_output))
                '''
            cell_state = tuple(new_cell_state)
            (_, output) = cell_state[-1]
        # MUltiple layers require us to call the individual lstm layers again, after the first, as they are currently
        # updated assuming a full read. This is not hard to do, but is kinda "hacky". Not currently implemented, but
        # is straight forward
        else:
            None
        return cell_state, output

    def apply_dropout_lstm(self, cell_state, output):
        new_cell_state = []
        for state in cell_state:
            (internal, output) = state
            internal = tf.nn.dropout(internal, self.rnn_dropout_var)
            new_cell_state.append(tf.nn.rnn_cell.LSTMStateTuple(internal, output))
        cell_state = tuple(new_cell_state)
        if output is not None:
            output = tf.nn.dropout(output, self.rnn_dropout_var)
        return cell_state, output

    def _cell_ouputs_to_state(self, cell_outputs, last_state, is_filtered, action_agent_1):
        if self.attention:
            # compute logits for attention
            logits_attention = tf.squeeze(tf.layers.dense(cell_outputs, units=1, name="attention_logits"), axis=-1)
            # get max for softmax(x) = softmax(x-c)
            max_logits = tf.reduce_max(logits_attention, axis=0)
            max_logits = tf.stop_gradient(max_logits)
            logits_attention_shift = logits_attention - max_logits
            exp_attention = tf.expand_dims(tf.exp(logits_attention_shift), axis=-1)
            exp_attention_masked = exp_attention * tf.cast(is_filtered, dtype=tf.float32) * tf.cast(action_agent_1,
                                                                                                    dtype=tf.float32)
            summed_attention = tf.reduce_sum(exp_attention_masked, axis=0)
            exp_attention_masked_norm = exp_attention_masked / (summed_attention + 0.00001)
            exp_attention_masked_norm = tf.reshape(exp_attention_masked_norm, shape=(-1, self.batch_size, 1))
            weigthed_state = cell_outputs * exp_attention_masked_norm
            outputs = tf.reduce_sum(weigthed_state, axis=0)
        else:
            (_, outputs) = last_state[-1]  # the last output is kept through propagating
            # have not dropped out in the output during run, so do it here
            outputs = tf.nn.dropout(outputs, self.rnn_dropout_var)
        return outputs

    def _build_state_1(self, cell_state, input):
        (internal, out) = cell_state[0]
        use_as_input = out
        #use_as_input = tf.stop_gradient(use_as_input)
        concat_state = tf.concat([use_as_input, input], axis=1)
        state = tf.layers.dense(concat_state, self.size_state_1, activation=tf.nn.relu, name="state_1")
        return state

    def _build_policy_1(self, state):
        logits_agent_1 = tf.layers.dense(state, self.number_actions_1, name="logits_agent_1")
        return logits_agent_1

    def _build_value_1(self, state):
        value_agent_1 = tf.layers.dense(state, 1, name="value_agent_1")
        return value_agent_1

    '''
    def _reward_1(self, action_1,length):
        reward_update = tf.where(tf.equal(action_1,1),tf.const(0.5,self.batch_size),tf.const(1,self.batch_size))
        return reward_update
    '''

    def _build_state_2(self, state):
        (hidden, out) = state[0]
        use_as_input = out
        #use_as_input = tf.stop_gradient(use_as_input)
        state = tf.layers.dense(use_as_input, self.size_state_2, activation=tf.nn.relu, name="state_2")
        return state

    def _build_policy_2(self, state):
        logits_agent_2 = tf.layers.dense(state, self.number_actions_2, name="logits_agent_2")
        return logits_agent_2

    def _build_value_2(self, state):
        value_agent_2 = tf.layers.dense(state, 1, name="value_agent_2")
        return value_agent_2

    # add support for question answering later, with a slight change to _build_pred
    def _build_pred(self, state):
        pred_state_1 = tf.layers.dense(state, units=self.cell_size, activation=tf.nn.relu, name="prediction_state_1")
        output_logits = tf.layers.dense(pred_state_1, units=self.number_targets, name='prediction_logits')
        return output_logits

    def _build_pred_q_a(self, state, q_a_words_embedded):
        pred_state_1 = tf.layers.dense(state, units=self.embedding_size, activation=tf.nn.relu,
                                       name="prediction_state_1_q_a")
        outputs = tf.expand_dims(pred_state_1, -1)
        output_logits = tf.matmul(q_a_words_embedded, outputs)
        output_logits = tf.squeeze(output_logits, [-1])
        return output_logits

    def _make_output_logits(self, rnn_state, q_a_words_embedded):
        if q_a_words_embedded is None:
            output_logits = self._build_pred(rnn_state)
        else:
            output_logits = self._build_pred_q_a(rnn_state, q_a_words_embedded)

        return output_logits

    # states is batch x time
    def _embedding_lookup(self, states, embedding):
        with tf.variable_scope(self.scope.name) as scope:
            embedded_input = tf.nn.embedding_lookup(embedding, states)
            return embedded_input

    def _make_embedding(self, trainable=True, init=False):
        with tf.variable_scope(self.scope.name) as scope:
            W = tf.Variable(tf.random_uniform(shape=[self.vocab_size, self.embedding_size], minval=-0.05, maxval=0.05),
                            trainable=trainable, name="embedding")
            if init:
                embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_size])
                embedding_init = W.assign(embedding_placeholder)
                return (W, embedding_placeholder, embedding_init)
            else:
                return W

    def make_sync_op(self, master):
        ops = [tf.assign(my, master) for my, master in zip(self.train_vars, master.train_vars)]
        return tf.group(*ops)

    def _simple_action_sampler(self, logits):
        if self.sampling_method == "normal":
            samples = tf.multinomial(logits, 1, output_dtype=tf.int32)
            samples = samples[:, 0]
            return samples
        elif self.sampling_method == "greedy":
            samples = tf.argmax(logits, axis=1, output_type=tf.int32)
            return samples
        elif self.sampling_method == "eps-greedy":
            None

    def _probability_of_action(self, logits, samples, range_to_stack):
        to_index = tf.stack([range_to_stack, samples], axis=1)
        probs = tf.nn.softmax(logits)
        probs = tf.gather_nd(probs, to_index)
        return probs

    def _optimize(self, loss):
        gradients = tf.gradients(loss, self.train_vars)
        gradients, use_tran_vars = zip(*filter(lambda g: g[0] is not None, (zip(gradients, self.train_vars))))
        clipped_gs = [tf.clip_by_average_norm(g, self.grad_clip) for g in gradients]
        train_op = self.optimizer.apply_gradients(zip(clipped_gs, use_tran_vars))

        gradients = tf.gradients(loss, self.train_vars_no_embedding)
        gradients, use_tran_vars_no_embedding = zip(
            *filter(lambda g: g[0] is not None, (zip(gradients, self.train_vars))))
        clipped_gs = [tf.clip_by_average_norm(g, self.grad_clip) for g in gradients]
        train_op_no_embedding = self.optimizer.apply_gradients(zip(clipped_gs, use_tran_vars_no_embedding))

        return train_op, train_op_no_embedding

    @staticmethod
    def agent_1_skip():
        return 0

    @staticmethod
    def agent_1_read():
        return 1

    @staticmethod
    def agent_2_word():
        return 0

    @staticmethod
    def agent_2_comma():
        return 1

    @staticmethod
    def agent_2_punct():
        return 2

    @staticmethod
    def agent_2_end():
        return 3

    def __init__(self, batch_size, vocab_size, number_targets, device, scope_name, attention,
                 grad_clip, cell_size, size_state_1, size_state_2, embedding_dropout, rnn_dropout, n_stacks,
                 trainable_embedding,
                 skim_cells, sampling_method, off_policy, initial_dist_agent_1, initial_dist_agent_2, name_optimizer, include_entropy_in_full,
                 is_Q_A=False, consumer=None, learning_rate=None, scale_entropy=None, embedding_size=300):
        self.scope_name = scope_name
        self.batch_size = batch_size
        '''
        if is_Q_A:
            self.cell_size = embedding_size
        else:
            '''
        self.cell_size = cell_size

        self.attention = attention
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.number_targets = number_targets
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip

        self.sample_fn = self._simple_action_sampler

        self.size_state_1 = size_state_1
        self.number_actions_1 = 2

        self.size_state_2 = size_state_2
        self.number_actions_2 = 4

        self.n_stacks = n_stacks
        self.skim_cells = skim_cells
        self.sampling_method = sampling_method
        self.off_policy = off_policy
        self.include_entropy_in_full = include_entropy_in_full

        self.consumer = consumer

        self.embedding_dropout = embedding_dropout
        self.rnn_dropout = rnn_dropout

        self.initial_dist_agent_1 = initial_dist_agent_1  # tf.constant(([0.3, 0.7]),dtype=tf.float32)#tf.constant(([0.5, 0.5]),dtype=tf.float32)
        self.initial_dist_agent_2 = initial_dist_agent_2  # tf.constant([0.95, 0.03, 0.0199, 0.0001],dtype=tf.float32)#tf.constant([0.25, 0.25, 0.25, 0.25],dtype=tf.float32)

        # just some constants that are nice to have
        self.agent_1_skip = ActorCritic.agent_1_skip()
        self.agent_1_read = ActorCritic.agent_1_read()

        self.agent_2_word = ActorCritic.agent_2_word()
        self.agent_2_comma = ActorCritic.agent_2_comma()
        self.agent_2_punct = ActorCritic.agent_2_punct()
        self.agent_2_end = ActorCritic.agent_2_end()

        with tf.device(device):
            with tf.variable_scope(scope_name) as scope:
                self.scope = scope
            self.input = tf.placeholder(tf.int32, [None, self.batch_size])
            self.time_length = tf.placeholder(tf.int32, self.batch_size)
            self.comma = tf.placeholder(tf.int32, [None, self.batch_size])
            self.punctuation = tf.placeholder(tf.int32, [None, self.batch_size])

            # variables needed during training
            self.rnn_dropout_var = tf.Variable(self.rnn_dropout, dtype=tf.float32, trainable=False)
            self.rnn_dropout_placeholder = tf.placeholder(tf.float32)
            self.set_rnn_dropout = self.rnn_dropout_var.assign(self.rnn_dropout_placeholder)

            self.embedding_dropout_var = tf.Variable(self.embedding_dropout, dtype=tf.float32, trainable=False)
            self.embedding_dropout_placeholder = tf.placeholder(tf.float32)
            self.set_embedding_dropout = self.embedding_dropout_var.assign(self.embedding_dropout_placeholder)

            (embedding, self.embedding_placeholder, self.embedding_init) = self._make_embedding(
                trainable=trainable_embedding, init=True)
            embedded_input = self._embedding_lookup(self.input, embedding)
            embedded_input = tf.nn.dropout(embedded_input, self.embedding_dropout_var)

            if is_Q_A:
                self.q_a_words_input = tf.placeholder(tf.int32, [self.batch_size, None], name="q_a_potential_words")
                q_a_words_embedded = self._embedding_lookup(self.q_a_words_input, embedding)
            else:
                q_a_words_embedded = None

            if self.skim_cells > 0:
                self.small_rnns = self._make_small_rnn_cell()

            self.read_words, self.action_agent_1, self.value_agent_1, self.action_agent_2, self.value_agent_2, self.predictions, self.is_not_done, \
            self.probs_agent_1, self.probs_agent_2 \
                = self._build_rnn_sampler(embedded_input, self.time_length, self.comma, self.punctuation,
                                          q_a_words_embedded)

            if "sampler" in self.scope_name:
                self.train_vars = sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_name),
                                         key=lambda v: v.name)
                self.make_sync()
            elif self.scope_name == "consumer":
                if name_optimizer == "RMSE":
                    self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, use_locking=True)
                elif name_optimizer == "ADAM":
                    self.optimizer = tf.train.AdamOptimizer(self.learning_rate, use_locking=True)
                self.scale_entropy = scale_entropy

                # moving average of advantage
                self.moving_advantage = tf.Variable(0, dtype=tf.float32, trainable=False)
                self.decay_moving_advantage = 0.8

                self.sampled_input = tf.placeholder(tf.int32, [None, self.batch_size], name="sampled_input")
                self.sampled_time_length = tf.placeholder(tf.int32, [self.batch_size], name="sampled_time")
                self.sampled_advantage_agent_1 = tf.placeholder(tf.float32, [None, self.batch_size],
                                                                name="sampled_advantage_agent_1")
                self.sampled_action_agent_1 = tf.placeholder(tf.int32, [None, self.batch_size],
                                                             name="sampled_action_agent_1")
                self.sampled_advantage_agent_2 = tf.placeholder(tf.float32, [None, self.batch_size],
                                                                name="sampled_advantage_agent_2")
                self.sampled_action_agent_2 = tf.placeholder(tf.int32, [None, self.batch_size],
                                                             name="sampled_action_agent_2")
                self.sampled_is_not_done = tf.placeholder(tf.int32, [None, self.batch_size], name="is_not_done")
                self.target = tf.placeholder(tf.int32, self.batch_size, name="target")
                self.reward = tf.placeholder(tf.float32, self.batch_size, name="reward")

                # off policy learning, get sampled prob for action
                self.sampled_probs_agent_1 = tf.placeholder(tf.float32, [None, self.batch_size],
                                                            name="sampled_probs_agent_1")
                self.sampled_probs_agent_2 = tf.placeholder(tf.float32, [None, self.batch_size],
                                                            name="sampled_probs_agent_2")

                # factors
                self.scale_pred = tf.placeholder(tf.float32, shape=())
                self.scale_agent_critic = tf.placeholder(tf.float32, shape=())
                self.scale_agent_actor = tf.placeholder(tf.float32, shape=())

                embedded_sampled_input = self._embedding_lookup(self.sampled_input, embedding)
                self.total_loss, self.pred_loss, self.actor_loss, self.critic_loss, self.entropy_loss, self.consumer_pred, self.attn \
                    = self._build_rnn_consumer(embedded_sampled_input, self.sampled_time_length,
                                               self.sampled_advantage_agent_1,
                                               self.sampled_action_agent_1, self.sampled_advantage_agent_2,
                                               self.sampled_action_agent_2,
                                               self.sampled_is_not_done, self.target, self.reward, self.scale_pred,
                                               self.scale_agent_critic, self.scale_agent_actor, self.scale_entropy,
                                               q_a_words_embedded,
                                               self.sampled_probs_agent_1, self.sampled_probs_agent_2)

                self.train_vars = sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_name),
                                         key=lambda v: v.name)
                self.train_vars_no_embedding = sorted(list(filter(lambda x: "embedding" not in x.name,
                                                                  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                    self.scope_name))),
                                                      key=lambda v: v.name)
                self.train_op, self.train_op_no_embedding = self._optimize(self.total_loss)
                if self.include_entropy_in_full:
                    self.train_op_pred_only, self.train_op_no_embedding_pred_only = self._optimize(
                        self.pred_loss + self.entropy_loss)
                else:
                    self.train_op_pred_only, self.train_op_no_embedding_pred_only = self._optimize(
                        self.pred_loss)

                # make full reader pred only
                self.full_read_pred_only = self._build_rnn_full_reader(embedded_input, self.time_length,
                                                                       q_a_words_embedded)

                #
                entropy_critic_loss = self.critic_loss + self.entropy_loss
                self.train_entropy_critic_loss, self.train_no_embedding_entropy_critic_loss = self._optimize(
                    entropy_critic_loss)

    def set_dropout(self, sess):
        sess.run([self.set_embedding_dropout, self.set_rnn_dropout],
                 feed_dict={self.embedding_dropout_placeholder: self.embedding_dropout,
                            self.rnn_dropout_placeholder: self.rnn_dropout})

    def set_test(self, sess):
        sess.run([self.set_embedding_dropout, self.set_rnn_dropout],
                 feed_dict={self.embedding_dropout_placeholder: 1,
                            self.rnn_dropout_placeholder: 1})

    def initialize(self, sess, embedding=None):
        self.sess = sess
        if self.scope.name == "consumer" and embedding is not None:
            self.sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: embedding})
        else:
            self.sync()

    def get_sample(self, words, time_length, comma, punctuation, q_a_words):
        feed_dict = {
            self.input: words,
            self.time_length: time_length,
            self.comma: comma,
            self.punctuation: punctuation
        }
        if q_a_words is not None:
            feed_dict[self.q_a_words_input] = q_a_words

        read_words, action_agent_1, value_agent_1, action_agent_2, value_agent_2, predictions, is_not_done, probs_agent_1, probs_agent_2 = \
            self.sess.run(
                [self.read_words, self.action_agent_1, self.value_agent_1, self.action_agent_2, self.value_agent_2,
                 self.predictions, self.is_not_done, self.probs_agent_1, self.probs_agent_2],
                feed_dict)
        return read_words, action_agent_1, value_agent_1, action_agent_2, value_agent_2, predictions, is_not_done, probs_agent_1, probs_agent_2

    def consume_sample(self, sampled_input, sampled_time_length, sampled_advantage_agent_1, sampled_action_agent_1,
                       sampled_advantage_agent_2, sampled_action_agent_2, is_not_done, target, reward,
                       scale_pred, scale_critic, scale_actor, q_a_words, sampled_probs_agent_1, sampled_probs_agent_2,
                       embedding_train=True):
        if self.scope_name == "consumer":
            if embedding_train:
                train_op = self.train_op
            else:
                train_op = self.train_op_no_embedding
            feed_dict = {
                self.sampled_input: sampled_input,
                self.sampled_time_length: sampled_time_length,
                self.sampled_advantage_agent_1: sampled_advantage_agent_1,
                self.sampled_action_agent_1: sampled_action_agent_1,
                self.sampled_advantage_agent_2: sampled_advantage_agent_2,
                self.sampled_action_agent_2: sampled_action_agent_2,
                self.sampled_is_not_done: is_not_done,
                self.target: target,
                self.reward: reward,
                self.scale_pred: scale_pred,
                self.scale_agent_critic: scale_critic,
                self.scale_agent_actor: scale_actor,
                self.sampled_probs_agent_1: sampled_probs_agent_1,
                self.sampled_probs_agent_2: sampled_probs_agent_2
            }

            if q_a_words is not None:
                feed_dict[self.q_a_words_input] = q_a_words

            _, pred_loss, actor_loss, critic_loss, entropy_loss = self.sess.run(
                [train_op, self.pred_loss, self.actor_loss, self.critic_loss, self.entropy_loss], feed_dict)
            return pred_loss, actor_loss, critic_loss, entropy_loss
        else:
            return self.consumer.consume_sample(sampled_input, sampled_time_length, sampled_advantage_agent_1,
                                                sampled_action_agent_1,
                                                sampled_advantage_agent_2, sampled_action_agent_2, is_not_done, target,
                                                reward, scale_pred,
                                                scale_critic, scale_actor, q_a_words, sampled_probs_agent_1,
                                                sampled_probs_agent_2, embedding_train)

    def consume_sample_init(self, sampled_input, sampled_time_length, sampled_advantage_agent_1, sampled_action_agent_1,
                            sampled_advantage_agent_2, sampled_action_agent_2, is_not_done, scale_critic, q_a_words,
                            embedding_train=False):
        assert (self.scope_name == "consumer")
        if embedding_train:
            train_op = self.train_entropy_critic_loss
        else:
            train_op = self.train_no_embedding_entropy_critic_loss
        feed_dict = {
            self.sampled_input: sampled_input,
            self.sampled_time_length: sampled_time_length,
            self.sampled_action_agent_1: sampled_action_agent_1,
            self.sampled_advantage_agent_1: sampled_advantage_agent_1,
            self.sampled_action_agent_2: sampled_action_agent_2,
            self.sampled_advantage_agent_2: sampled_advantage_agent_2,
            self.sampled_is_not_done: is_not_done,
            self.scale_agent_critic: scale_critic,
        }
        if q_a_words is not None:
            feed_dict[self.q_a_words_input] = q_a_words
        self.sess.run(train_op, feed_dict)

    def consume_give_attention(self, words, time_length, q_a_words):
        assert (self.scope_name == "consumer")
        action_1 = np.zeros(shape=(np.max(time_length), self.batch_size), dtype=np.int32) + self.agent_1_read
        action_2 = np.zeros(shape=(np.max(time_length), self.batch_size), dtype=np.int32) + self.agent_2_word
        is_not_done = np.zeros(shape=(np.max(time_length), self.batch_size), dtype=np.int32)
        for i in range(self.batch_size):
            is_not_done[:time_length[i], i] = 1

        feed_dict = {
            self.sampled_input: words,
            self.sampled_time_length: time_length,
            self.sampled_action_agent_1: action_1,
            self.sampled_action_agent_2: action_2,
            self.sampled_is_not_done: is_not_done,
        }
        if q_a_words is not None:
            feed_dict[self.q_a_words_input] = q_a_words

        _, pred_loss, pred = self.sess.run([self.consumer_pred, self.attn], feed_dict)
        return pred_loss, pred

    def consume_sample_full_read(self, words, time_length, target, scale_pred, q_a_words, embedding_train=True):
        if self.scope_name == "consumer":
            if embedding_train:
                train_op = self.train_op_pred_only
            else:
                train_op = self.train_op_no_embedding_pred_only

            action_1 = np.zeros(shape=(np.max(time_length), self.batch_size), dtype=np.int32) + self.agent_1_read
            action_2 = np.zeros(shape=(np.max(time_length), self.batch_size), dtype=np.int32) + self.agent_2_word
            is_not_done = np.zeros(shape=(np.max(time_length), self.batch_size), dtype=np.int32)
            for i in range(self.batch_size):
                is_not_done[:time_length[i], i] = 1

            feed_dict = {
                self.sampled_input: words,
                self.sampled_time_length: time_length,
                self.sampled_action_agent_1: action_1,
                self.sampled_action_agent_2: action_2,
                self.scale_pred: scale_pred,
                self.sampled_is_not_done: is_not_done,
                self.target: target,
            }
            if q_a_words is not None:
                feed_dict[self.q_a_words_input] = q_a_words

            _, pred_loss, pred, ent = self.sess.run([train_op, self.pred_loss, self.consumer_pred, self.entropy_loss], feed_dict)
            return pred_loss, pred, ent
        else:
            return self.consumer.consume_sample_full_read(words, time_length, target, scale_pred, q_a_words,
                                                          embedding_train=embedding_train)

    def full_read(self, words, time_length, q_a_words):
        assert (self.scope_name == "consumer")
        feed_dict = {
            self.input: words,
            self.time_length: time_length,
        }
        if q_a_words is not None:
            feed_dict[self.q_a_words_input] = q_a_words
        pred = self.sess.run([self.full_read_pred_only], feed_dict)
        return pred

    def make_sync(self):
        self.sync_op = self.make_sync_op(self.consumer)

    def sync(self):
        if (self.scope.name == 'consumer'):
            return
        self.sess.run(self.sync_op)