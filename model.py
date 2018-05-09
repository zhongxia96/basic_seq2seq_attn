# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from util import linear
import os
import data

FLAGS = tf.app.flags.FLAGS

class SummarizationModel(object):
    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab

    def _add_placeholders(self):
        hps = self._hps
        self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
        self._enc_len = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
        self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')

        self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
        self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
        self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')

    def _make_feed_dict(self, batch, just_enc=False):
        feed_dict = {}
        feed_dict[self._enc_batch] = batch.enc_batch
        feed_dict[self._enc_len] = batch.enc_lens
        feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
        if not just_enc:
            feed_dict[self._dec_batch] = batch.dec_batch
            feed_dict[self._target_batch] = batch.target_batch
            feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
        return feed_dict

    def _add_encoder(self, encoder_inputs):
        """encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
           Return:
               encoder_outputs: List length sim_count of [batch_size, <=max_enc_steps, 2*hidden_dim]
               encoder_final_state: LSTMStateTuple([batch_size,hidden_dim],[batch_size,hidden_dim])
        """
        hps = self._hps
        with tf.variable_scope('encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(hps.hidden_dim, initializer=self.rand_unif_init)
            cell_bw = tf.contrib.rnn.LSTMCell(hps.hidden_dim, initializer=self.rand_unif_init)
            encoder_output, (fw_st, bw_st) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=self._enc_len, swap_memory=True)
            self.encoder_output = tf.concat(encoder_output, axis=2)
            state_c = linear([fw_st.c, bw_st.c], hps.hidden_dim, True, 'reduce_state_c')
            state_h = linear([fw_st.h, bw_st.h], hps.hidden_dim, True, 'reduce_state_h')
            self.encoder_final_state = tf.contrib.rnn.LSTMStateTuple(state_c, state_h)

    def _add_decoder(self, inputs):
        hps = self._hps
        with tf.variable_scope('attention_decoder'):
            attn_size = self.encoder_output.get_shape()[2].value
            attention_vec_size = attn_size

            def attention(decoder_state):
                encoder_state = tf.expand_dims(self.encoder_output, axis=2)
                encoder_padding_mask = self._enc_padding_mask
                with tf.variable_scope('Attention'):
                    W_h = tf.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
                    encoder_feature = tf.nn.conv2d(encoder_state, W_h, [1, 1, 1, 1], "SAME")
                    decoder_feature = linear(decoder_state, attention_vec_size, True)
                    decoder_feature = tf.expand_dims(tf.expand_dims(decoder_feature, axis=1), axis=1)
                    v = tf.get_variable("v", [attention_vec_size])
                    e = tf.reduce_sum(v * tf.tanh(encoder_feature + decoder_feature), [2, 3])
                    attn_dist = tf.nn.softmax(e)
                    attn_dist *= encoder_padding_mask
                    masked_sums = tf.reduce_sum(attn_dist, axis=1)
                    attn_dist = attn_dist / tf.reshape(masked_sums, [-1, 1])  # shape: (batch_size, attn_length)
                    context_vector = tf.reduce_sum(tf.reshape(attn_dist, [hps.batch_size, -1, 1, 1]) * encoder_state, [1, 2])
                return context_vector

            if hps.mode == "decode":  # Re-calculate the context vector from the previous step
                context_vector= attention(self.dec_in_state)
            else:
                context_vector = tf.zeros([hps.batch_size, attn_size])
            cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)
            state = self.dec_in_state
            self.vocab_dists = []
            for i, inp in enumerate(inputs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                input_size = inp.get_shape().with_rank(2)[1]
                x = linear([inp] + [context_vector], input_size, True)
                cell_output, state = cell(x, state)
                self.dec_out_state = state
                if hps.mode == "decode":
                    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                        context_vector = attention(state)
                else:
                    context_vector = attention(state)
                with tf.variable_scope("AttnOutputProjection"):
                    output = linear([cell_output] + [context_vector], cell.output_size, "True")
                vocab_dist = tf.nn.softmax(linear(output, self._vocab.size(), True, 'vocab_dist'))
                self.vocab_dists.append(vocab_dist)


    def _add_seq2seq(self):
        hps = self._hps
        vsize = self._vocab.size()
        with tf.variable_scope('seq2seq'):
            self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)
            with tf.variable_scope('embedding'):
                embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
                emb_enc_input = tf.nn.embedding_lookup(embedding, self._enc_batch)
                emb_dec_input = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch, axis=1)]
                #emb_dec_input = tf.unstack(tf.nn.embedding_lookup(embedding, self._dec_batch), axis=1)
            self._add_encoder(emb_enc_input)
            self.dec_in_state = self.encoder_final_state
            self._add_decoder(emb_dec_input)
            if hps.mode in ['train', 'eval']:
                with tf.variable_scope('loss'):
                    loss_per_step = []
                    targets = tf.unstack(self._target_batch, axis=1)
                    batch_nums = tf.range(0, limit=hps.batch_size)
                    for dist, target in zip(self.vocab_dists, targets):
                        indice = tf.stack([batch_nums, target], axis=1) # (batch_size, 2)
                        gold_probs = tf.gather_nd(dist, indice)
                        loss = -tf.log(tf.clip_by_value(gold_probs, 1e-8, 1000000))
                        loss_per_step.append(loss)
                    dec_lens = tf.reduce_sum(self._dec_padding_mask, axis=1)  # shape (batch_size)
                    values_per_step = [v * self._dec_padding_mask[:, i] for i, v in enumerate(loss_per_step)]  # (batch_size, dec_len)
                    values_per_ex = sum(values_per_step) / dec_lens  # (batch_size)
                    self._loss = tf.reduce_mean(values_per_ex)
                    tf.summary.scalar('loss', self._loss)
            if hps.mode == 'decode':
                assert len(self.vocab_dists) == 1 # final_dists is a singleton list containing shape (batch_size, extended_vsize)
                topk_probs, self._topk_ids = tf.nn.top_k(self.vocab_dists[0], hps.batch_size * 2)
                self._topk_log_probs = tf.log(topk_probs)

    def build_graph(self):
        tf.logging.info("build graph...")
        self._add_placeholders()
        self._add_seq2seq()
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        if self._hps.mode == "train":
            gradients = tf.gradients(self._loss, tf.trainable_variables(), aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
            grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)
            tf.summary.scalar('global_norm', global_norm)
            optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
            self._train_op = optimizer.apply_gradients(zip(grads, tf.trainable_variables()), self.global_step, name='train_step')
        self._summaries = tf.summary.merge_all()

    def run_train_step(self, sess, batch):
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'train_op': self._train_op,
            'loss': self._loss,
            'global_step': self.global_step,
            'summaries': self._summaries
        }
        return sess.run(to_return, feed_dict)

    def run_eval_step(self, sess, batch):
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'loss': self._loss,
            'global_step': self.global_step,
            'summaries': self._summaries
        }
        return sess.run(to_return, feed_dict)

    def run_beam_decode_step(self, sess, batch, vocab):
        feed_dict = self._make_feed_dict(batch, just_enc=True)
        to_return = [self.encoder_output, self.dec_in_state, self.global_step]
        # encoder_output ：[batch_size, <=max_enc_steps, 2*hidden_dim]
        encoder_output, dec_in_state, global_step = sess.run(to_return, feed_dict)
        dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
        hyps = [Hypothesis([vocab.word2id(data.START_DECODING)],
                     log_probs=[0.0],
                     state=dec_in_state) for _ in range(FLAGS.beam_size)]
        results = []
        steps = 0
        while steps < FLAGS.max_dec_steps and len(results) < FLAGS.beam_size:
            lastest_tokens = [h.latest_token for h in hyps]
            states = [h.state for h in hyps]
            cells = [np.expand_dims(s.c, axis=0) for s in states]
            hiddens = [np.expand_dims(s.h, axis=0) for s in states]
            new_c = np.concatenate(cells, axis=0)
            new_h = np.concatenate(hiddens, axis=0)
            new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            feed_dict = {
                self.encoder_output: encoder_output,
                self._enc_padding_mask: batch.enc_padding_mask,
                self.dec_in_state: new_dec_in_state,
                self._dec_batch: np.transpose(np.array([lastest_tokens]))
            }
            to_return = [self._topk_ids, self._topk_log_probs, self.dec_out_state]
            topk_ids, topk_log_probs, new_states = sess.run(to_return, feed_dict)
            new_states = [tf.contrib.rnn.LSTMStateTuple(new_states.c[i, :], new_states.h[i, :]) for i in range(FLAGS.beam_size)]
            all_hyps = []
            num_orig_hyps = 1 if steps == 0 else len(hyps)
            for i in range(num_orig_hyps):
                h, new_state = hyps[i], new_states[i]
                for j in range(FLAGS.beam_size * 2):
                    new_hyp = h.extend(
                        token=topk_ids[i, j],
                        log_prob=topk_log_probs[i, j],
                        state=new_state
                    )
                    all_hyps.append(new_hyp)
            hyps = []
            for h in sort_hyps(all_hyps):
                if h.latest_token == vocab.word2id(data.STOP_DECODING):
                    if steps > FLAGS.min_dec_steps:
                        results.append(h)
                else:
                    hyps.append(h)
                if len(hyps) == FLAGS.beam_size or len(results) == FLAGS.beam_size:
                    break
            steps += 1
        if len(results) == 0:
            results =hyps
        hyps_sorted = sort_hyps(results)
        return hyps_sorted[0].tokens[1:]


class Hypothesis(object):
  """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

  def __init__(self, tokens, log_probs, state):
      self.tokens = tokens
      self.log_probs = log_probs
      self.state = state
  def extend(self, token, log_prob, state):
    return Hypothesis(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def log_prob(self):
    # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
    return sum(self.log_probs)

  @property
  def avg_log_prob(self):
    # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
    return self.log_prob / len(self.tokens)



def sort_hyps(hyps):
  """Return a list of Hypothesis objects, sorted by descending average log probability"""
  return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)



