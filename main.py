import tensorflow as tf

# checkpoint_dir = "/home/cwq/data/model/slim/mobilenetv2"
checkpoint_dir = "/home/cwq/data/checkpoint/tf_crnn/simple_no_lstm_more_bg"
print("Restoring checkpoint from: " + checkpoint_dir)

ckpt = tf.train.latest_checkpoint(checkpoint_dir)
if ckpt is None:
    print("Checkpoint not found")
    exit(-1)

meta_file = ckpt + '.meta'

print('Restore variables from {}'.format(ckpt))
print('Restore meta_file from {}'.format(meta_file))

config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph(meta_file)
    saver.restore(sess, ckpt)

    input_graph_def = tf.get_default_graph().as_graph_def()

    # Print all node name in graph
    for node in input_graph_def.node:
        print(node.name)
