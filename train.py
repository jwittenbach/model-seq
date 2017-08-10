import tensorflow as tf
import numpy as np
from time import strftime

def cv_batch_fit(model, data, cv_frac, batch_frac, seed=0):
    # get sizes
    n_elems = np.prod(model.shape)
    n_cv = np.floor(cv_frac * n_elems).astype(int)
    n_batch = np.floor(batch_frac * n_elems).astype(int)
    n_batches = np.round((n_elems - n_cv)/n_batch).astype(int)

    # generate batches
    print("generating batches...")
    inds = np.random.permutation(np.arange(n_elems))

    cv_inds = inds[:n_cv]
    cv_mask = np.zeros(model.shape, dtype=bool)
    cv_mask[np.unravel_index(cv_inds, model.shape)] = True
    cv_targets = data[cv_mask]

    train_inds = inds[n_cv:]
    batches = np.array_split(train_inds, n_batches)
    batch_masks = []
    for i in range(n_batches):
        batch_masks.append(np.zeros(model.shape, dtype=bool))
        batch_masks[i][np.unravel_index(batches[i], model.shape)] = True
    batch_targets = [data[mask] for mask in batch_masks]

    # logging / checkpointing
    train_summary = tf.summary.scalar("training_loss", model.loss)
    test_summary = tf.summary.scalar("testing_loss", model.loss)

    subdir = strftime("%y.%m.%d-%H.%M.%S")
    summary_writer = tf.summary.FileWriter("./logs/"+subdir, graph=tf.get_default_graph())

    saver = tf.train.Saver()
    save_path = "./checkpoints/" + subdir + "/model.ckpt"

    log_interval = 100

    opt = tf.train.AdamOptimizer().minimize(model.loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #for i in range(4200):
        i = -1
        while True:
            i += 1
            batch = i % n_batches
            print(i)
            feed_dict = {
                model.batch_mask: batch_masks[batch],
                model.batch_targets: batch_targets[batch],
            }
            print("SGD step...")
            sess.run(opt, feed_dict)
            # write logs and checkppoint model
            if i % log_interval == 0:
                print("logging...")
                # training loss
                loss, summary = sess.run([model.loss, train_summary], feed_dict)
                summary_writer.add_summary(summary, i)
                # testing loss
                feed_dict = {
                    model.batch_mask: cv_mask,
                    model.batch_targets: cv_targets,
                }
                loss, summary = sess.run([model.loss, test_summary], feed_dict)
                summary_writer.add_summary(summary, i)
                # checkpoint
                saver.save(sess, save_path)
