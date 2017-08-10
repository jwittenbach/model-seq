import tensorflow as tf
import numpy as np
from time import strftime

def get_batch(n_elems, n_batch, cv_inds):
    """
    get indices for minibatch, avoid indices held out for CV
    """
    inds = np.random.randint(n_elems, size=n_batch, dtype=np.int64)
    return np.setdiff1d(inds, cv_inds, assume_unique=True)

def reshape_inds(inds, shape):
    return list(zip(*np.unravel_index(inds, shape)))

def cv_batch_fit(model, data, cv_frac, batch_frac, seed=0):
    # get sizes
    n_elems = np.prod(model.shape)
    n_cv = np.floor(cv_frac * n_elems).astype(int)
    n_batch = np.floor(batch_frac * n_elems).astype(int)

    # get test set to hold out for CV
    cv_inds = np.random.randint(n_elems, size=n_cv, dtype=np.int64)
    cv_targets = data.flatten()[cv_inds]
    cv_inds_2d = reshape_inds(cv_inds, model.shape)

    # get batches
    print("generating batches...")
    train_inds = np.setdiff1d(np.arange(n_elems), cv_inds)
    n_batches = np.round((n_elems - n_cv)/n_batch).astype(int)
    batches = np.array_split(np.random.permutation(train_inds), n_batches)

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
            print(i)
            # update model
            print("generating batch...")
            #batch_inds = get_batch(n_elems, n_batch, cv_inds)
            batch_inds = batches[i % n_batches]
            print("reshaping batch...")
            batch_inds_np = np.unravel_index(batch_inds, model.shape)
            batch_inds_2d = list(zip(*batch_inds_np))
            #batch_inds_2d = reshape_inds(batch_inds, model.shape)
            print("getting targets...")
            batch_targets = data[batch_inds_np]
            #batch_targets = data.flatten()[batch_inds]
            feed_dict = {
                model.batch_inds: batch_inds_2d,
                model.batch_targets: batch_targets,
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
                    model.batch_inds: cv_inds_2d,
                    model.batch_targets: cv_targets,
                }
                loss, summary = sess.run([model.loss, test_summary], feed_dict)
                summary_writer.add_summary(summary, i)
                # checkpoint
                saver.save(sess, save_path)
        #vals = sess.run(list(model.params.values()))
    #return dict(zip(model.params.keys(), vals))

