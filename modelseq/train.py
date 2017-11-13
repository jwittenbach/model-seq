import logging
from time import strftime

import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)


def cv_batch_fit(model, data, cv_mask=None, batch_masks=None, n_steps=None, dir=None, seed=0):

    if dir is None:
        subdir = strftime("%y.%m.%d-%H.%M.%S")
    else:
        subdir = dir

    np.random.seed(seed)

    n_batches = len(batch_masks)

    cv_targets = data[cv_mask]
    cv_feed_dict = {
        model.batch_mask: cv_mask,
        model.batch_targets: cv_targets,
    }
    batch_targets = [data[mask] for mask in batch_masks]

    # logging / checkpointing
    train_summary = tf.summary.scalar("training_loss", model.loss)
    test_summary = tf.summary.scalar("testing_loss", model.loss)

    neg_log_probs = -model.estimate.log_prob(model.batch_targets)
    max_log_prob = tf.summary.scalar("max_log_proba", tf.reduce_max(neg_log_probs))
    min_log_prob = tf.summary.scalar("min_log_proba", tf.reduce_min(neg_log_probs))

    graph = tf.get_default_graph()
    variable_summaries = []
    params = ['X', 'C', 'G']
    # params = ['X', 'C', 'G', 's_c', 's_g']
    for p in params:
        val = graph.get_tensor_by_name(p + ':0')
        variable_summaries.append(tf.summary.scalar('max_' + p, tf.reduce_max(val)))
        variable_summaries.append(tf.summary.scalar('min_' + p, tf.reduce_min(val)))
    params = []
    # params = ['a', 'b', 'r']
    for p in params:
        val = graph.get_tensor_by_name(p + ':0')
        variable_summaries.append(tf.summary.scalar(p, val))

    summary_writer = tf.summary.FileWriter("logs/" + subdir, graph=tf.get_default_graph())

    saver = tf.train.Saver()
    save_path = "checkpoints/" + subdir + "/ckpt"

    log_interval = 10

    opt = tf.train.AdamOptimizer().minimize(model.loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        i = -1
        try:

            while True:
                i += 1
                if n_steps and i > n_steps:
                    break

                logger.debug("sgd step {}...".format(i))

                # batch = i % n_batches
                batch = np.random.randint(n_batches)

                feed_dict = {
                    model.batch_mask: batch_masks[batch],
                    model.batch_targets: batch_targets[batch],
                }
                sess.run(opt, feed_dict)

                if i % log_interval == 0:
                    logger.info("step {}, logging...".format(i))

                    # training loss
                    loss, summary = sess.run([model.loss, train_summary], feed_dict)
                    summary_writer.add_summary(summary, i)

                    # testing loss
                    loss, summary = sess.run([model.loss, test_summary], cv_feed_dict)
                    summary_writer.add_summary(summary, i)

                    _, min_prob_summ, max_prob_summ\
                        = sess.run([neg_log_probs, min_log_prob, max_log_prob], feed_dict)
                    summary_writer.add_summary(min_prob_summ, i)
                    summary_writer.add_summary(max_prob_summ, i)

                    summaries = sess.run(variable_summaries)
                    for summary in summaries:
                        summary_writer.add_summary(summary, i)

                    # checkpoint
                    # saver.save(sess, save_path)

        except KeyboardInterrupt:
            logger.info("interrupted...")

        cv_loss = sess.run(model.loss, cv_feed_dict)
        saver.save(sess, save_path)

    return cv_loss


def make_masks(shape, cv_frac, batch_frac, seed=0):

    logger.info("generating batches...")
    np.random.seed(seed)

    n_elems = np.prod(shape)
    n_cv = np.floor(cv_frac * n_elems).astype(int)
    n_batch = np.floor(batch_frac * n_elems).astype(int)
    n_batches = np.round((n_elems - n_cv) / n_batch).astype(int)

    inds = np.random.permutation(np.arange(n_elems))

    cv_inds = inds[:n_cv]
    cv_mask = np.zeros(shape, dtype=bool)
    cv_mask[np.unravel_index(cv_inds, shape)] = True

    train_inds = inds[n_cv:]
    batches = np.array_split(train_inds, n_batches)
    batch_masks = []
    for i in range(n_batches):
        batch_masks.append(np.zeros(shape, dtype=bool))
        batch_masks[i][np.unravel_index(batches[i], shape)] = True

    return cv_mask, batch_masks