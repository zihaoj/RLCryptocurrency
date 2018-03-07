import tensorflow as tf


# cluster specification
ps_server = "localhost:2220"
workers = [
    "localhost:2222",
    "localhost:2223",
]

cluster = tf.train.ClusterSpec({
    "ps": [ps_server],
    "worker": workers,
})


# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# start the server
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)


if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":

    # define the graph
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:{:d}".format(FLAGS.task_index),
        cluster=cluster,
    )):
        with tf.variable_scope("SimpleGraph"):
            global_step = tf.train.get_or_create_global_step()
            x_placeholder = tf.placeholder(dtype=tf.float32, shape=(), name="x")

            output_ops = tf.square(x_placeholder, name="output")
            global_step_increment_ops = tf.assign(global_step, global_step+1)

    # run the session
    hooks = [tf.train.StopAtStepHook(num_steps=10000)]
    with tf.train.MonitoredTrainingSession(
        master=server.target,
        is_chief=(FLAGS.task_index == 0),
        checkpoint_dir="checkpoints/",
        hooks=hooks,
    ) as sess:
        while not sess.should_stop():
            result, step = sess.run([output_ops, global_step_increment_ops], feed_dict={x_placeholder: 5})
            print "{:s} - {:d}: {:d}, {:.4f}".format(FLAGS.job_name, FLAGS.task_index, step-1, result)

else:
    raise NotImplementedError("Unknown job name {:s}".format(FLAGS.job_name))
