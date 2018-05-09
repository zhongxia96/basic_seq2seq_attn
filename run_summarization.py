# -*- coding:utf-8 -*-
import tensorflow as tf
from model import SummarizationModel
from data import Vocab
import os
from batcher import Batcher
import util
import numpy as np
from collections import namedtuple
import time
import data
import pyrouge
import logging

SECS_UNTIL_NEW_CKPT = 60  # max number of seconds before loading new checkpoint

FLAGS = tf.app.flags.FLAGS

# Where to find data

tf.app.flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')

# Where to save output
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')


def main(unused_argv):
    tf.logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
    if not os.path.exists(FLAGS.log_root):
        if FLAGS.mode == "train":
            os.makedirs(FLAGS.log_root)
        else:
            raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))
    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)  # create a vocabulary
    if FLAGS.mode == 'decode':
        FLAGS.batch_size = FLAGS.beam_size
    hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
                   'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt',
                   'pointer_gen']
    hps_dict = {}
    for key, val in FLAGS.__flags.iteritems():  # for each flag
        if key in hparam_list:  # if it's in the list
            hps_dict[key] = val  # add it to the dict

    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
    print "hps.hidden_dim:", hps.hidden_dim

    # Create a batcher object that will create minibatches of data
    batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)

    tf.set_random_seed(111)  # a seed value for randomness
    if hps.mode == 'train':
        print "creating model..."
        model = SummarizationModel(hps, vocab)
        print "finish create  model..."
        setup_training(model, batcher)
    elif hps.mode == 'eval':
        model = SummarizationModel(hps, vocab)
        run_eval(model, batcher)
    elif hps.mode == 'decode':
        decode_model_hps = hps._replace(max_dec_steps=1)
        model = SummarizationModel(decode_model_hps, vocab)
        run_decode(model, batcher, vocab)
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode")

def setup_training(model, batcher):
    print "build graph...."
    model.build_graph()
    train_dir = os.path.join(FLAGS.log_root, "train")
    if not os.path.exists(train_dir): os.makedirs(train_dir)
    saver = tf.train.Saver(max_to_keep=3)
    sv = tf.train.Supervisor(logdir=train_dir,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,  # save summaries for tensorboard every 60 secs
                             save_model_secs=60,  # checkpoint every 60 secs
                             global_step=model.global_step)
    summary_writer = sv.summary_writer
    sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
    tf.logging.info("Created session.")
    with sess_context_manager as sess:
        while True:
            tf.logging.info('enter')
            batch = batcher.next_batch()
            results = model.run_train_step(sess, batch)
            loss = results['loss']
            print 'loss:', loss
            tf.logging.info('loss: %f', loss)  # print the loss to screen
            summaries = results['summaries']
            train_step = results['global_step']
            summary_writer.add_summary(summaries, train_step)
            if train_step % 100 == 0:
                summary_writer.flush()
            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

def run_eval(model, batcher):
    print "build graph..."
    model.build_graph()
    saver = tf.train.Saver(max_to_keep=3)
    sess = tf.Session(config=util.get_config())
    eval_dir = os.path.join(FLAGS.log_root, "eval")
    summary_writer = tf.summary.FileWriter(eval_dir)
    running_avg_loss = 0
    while True:
        _ = util.load_ckpt(saver, sess)
        batch  = batcher.next_batch()
        results = model.run_eval_step(sess, batch)
        loss = results['loss']
        print 'loss:', loss
        tf.logging.info('loss: %f', loss)
        summaries = results['summaries']
        train_step = results['global_step']
        summary_writer.add_summary(summaries, train_step)
        running_avg_loss = calc_running_avg_loss(np.asscalar(loss), running_avg_loss, summary_writer, train_step)
        if train_step % 100 == 0:
            summary_writer.flush()

def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
  if running_avg_loss == 0:  # on the first iteration just take the loss
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)  # clip
  loss_sum = tf.Summary()
  tag_name = 'running_avg_loss/decay=%f' % (decay)
  loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)
  tf.logging.info('running_avg_loss: %f', running_avg_loss)
  return running_avg_loss


def run_decode(model, batcher, vocab):
    print "build graph..."
    model.build_graph()
    saver = tf.train.Saver(max_to_keep=3)
    sess = tf.Session(config=util.get_config())
    saver = tf.train.Saver()
    ckpt_path = util.load_ckpt(saver, sess)
    if FLAGS.single_pass:
        ckpt_name = "ckpt-" + ckpt_path.split('-')[-1]
        dirname = "decode_maxenc_%ibeam_%imindec_%imaxdec_%i" % (FLAGS.max_enc_steps, FLAGS.beam_size, FLAGS.min_dec_steps, FLAGS.max_dec_steps)
        decode_dir = os.path.join(FLAGS.log_root, dirname + ckpt_name)
        if os.path.exists(decode_dir):
            raise Exception('single_pass decode directory %s should not exist', decode_dir)
    else:
        decode_dir = os.path.join(FLAGS.log_root, 'decode')
    if not os.path.exists(decode_dir): os.mkdir(decode_dir)
    if FLAGS.single_pass:
      rouge_ref_dir = os.path.join(decode_dir, "reference")
      if not os.path.exists(rouge_ref_dir): os.mkdir(rouge_ref_dir)
      rouge_dec_dir = os.path.join(decode_dir, "decoded")
      if not os.path.exists(rouge_dec_dir): os.mkdir(rouge_dec_dir)
    counter = 0
    t0 = time.time()
    while True:
        batch = batcher.next_batch()  # 1 example repeated across batch
        if batch is None:  # finished decoding dataset in single_pass mode
            assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
            print("Decoder has finished reading dataset for single_pass.")
            print("Output has been saved in %s and %s. Now starting ROUGE eval...", rouge_ref_dir,
                            rouge_dec_dir)
            results_dict = rouge_eval(rouge_ref_dir, rouge_dec_dir)
            rouge_log(results_dict, decode_dir)
            return

        original_article = batch.original_articles[0]  # string
        original_abstract = batch.original_abstracts[0]  # string
        original_abstract_sents = batch.original_abstracts_sents[0]  # list of strings

        article_withunks = data.show_art_oovs(original_article, vocab)  # string
        abstract_withunks = data.show_abs_oovs(original_abstract, vocab, None)  # string

        # Run beam search to get best Hypothesis
        output = model.run_beam_decode_step(sess, batch, vocab)
        output_ids = [int(t) for t in output]
        decoded_words = data.outputids2words(output_ids, vocab, None)

        # Remove the [STOP] token from decoded_words, if necessary
        try:
            fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
            decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
            decoded_words = decoded_words
        decoded_output = ' '.join(decoded_words)  # single string

        if FLAGS.single_pass:
            write_for_rouge(original_abstract_sents, decoded_words, counter, rouge_ref_dir, rouge_dec_dir)  # write ref summary and decoded summary to file, to eval with pyrouge later
            counter += 1  # this is how many examples we've decoded
        else:
            print_results(article_withunks, abstract_withunks, decoded_output)  # log output to screen
            # Check if SECS_UNTIL_NEW_CKPT has elapsed; if so return so we can load a new checkpoint
            t1 = time.time()
            if t1 - t0 > SECS_UNTIL_NEW_CKPT:
                tf.logging.info('We\'ve been decoding with same checkpoint for %i seconds. Time to load new checkpoint',
                                t1 - t0)
                _ = util.load_ckpt(saver, sess)
                t0 = time.time()

def write_for_rouge(reference_sents, decoded_words, ex_index, rouge_ref_dir, rouge_dec_dir):
        """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

        Args:
          reference_sents: list of strings
          decoded_words: list of strings
          ex_index: int, the index with which to label the files
        """
        # First, divide decoded output into sentences
        decoded_sents = []
        while len(decoded_words) > 0:
            try:
                fst_period_idx = decoded_words.index(".")
            except ValueError:  # there is text remaining that doesn't end in "."
                fst_period_idx = len(decoded_words)
            sent = decoded_words[:fst_period_idx + 1]  # sentence up to and including the period
            decoded_words = decoded_words[fst_period_idx + 1:]  # everything else
            decoded_sents.append(' '.join(sent))

        # pyrouge calls a perl script that puts the data into HTML files.
        # Therefore we need to make our output HTML safe.
        decoded_sents = [make_html_safe(w) for w in decoded_sents]
        reference_sents = [make_html_safe(w) for w in reference_sents]

        # Write to file
        ref_file = os.path.join(rouge_ref_dir, "%06d_reference.txt" % ex_index)
        decoded_file = os.path.join(rouge_dec_dir, "%06d_decoded.txt" % ex_index)

        with open(ref_file, "w") as f:
            for idx, sent in enumerate(reference_sents):
                f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")
        with open(decoded_file, "w") as f:
            for idx, sent in enumerate(decoded_sents):
                f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")

        print"Wrote example %i to file" ,ex_index

def print_results(article, abstract, decoded_output):
        """Prints the article, the reference summmary and the decoded summary to screen"""
        print ""
        print 'ARTICLE: ', article
        print 'REFERENCE SUMMARY: ', abstract
        print 'GENERATED SUMMARY: ', decoded_output
        print ""

def make_html_safe(s):
        """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
        s.replace("<", "&lt;")
        s.replace(">", "&gt;")
        return s

def rouge_eval(ref_dir, dec_dir):
        """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
        r = pyrouge.Rouge155()
        r.model_filename_pattern = '#ID#_reference.txt'
        r.system_filename_pattern = '(\d+)_decoded.txt'
        r.model_dir = ref_dir
        r.system_dir = dec_dir
        logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
        rouge_results = r.convert_and_evaluate()
        return r.output_to_dict(rouge_results)

def rouge_log(results_dict, dir_to_write):
        """Log ROUGE results to screen and write to file.

        Args:
          results_dict: the dictionary returned by pyrouge
          dir_to_write: the directory where we will write the results to"""
        log_str = ""
        for x in ["1", "2", "l"]:
            log_str += "\nROUGE-%s:\n" % x
            for y in ["f_score", "recall", "precision"]:
                key = "rouge_%s_%s" % (x, y)
                key_cb = key + "_cb"
                key_ce = key + "_ce"
                val = results_dict[key]
                val_cb = results_dict[key_cb]
                val_ce = results_dict[key_ce]
                log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
        tf.logging.info(log_str)  # log to screen
        results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
        tf.logging.info("Writing final ROUGE results to %s...", results_file)
        with open(results_file, "w") as f:
            f.write(log_str)

if __name__ == '__main__':
  tf.app.run()