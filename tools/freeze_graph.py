import tensorflow as tf
import argparse
import os

from tensorflow.python.framework.graph_util_impl import remove_training_nodes

import _init_paths
from nets.mobilenet_v1 import mobilenetv1
from utils import common

# noinspection PyProtectedMember
def main(args):
  with tf.Session() as sess:
    # Load the model metagraph and checkpoint
    print('Model directory: %s' % args.ckpt_dir)
    meta_file, ckpt_file = get_model_filenames(args.ckpt_dir)

    print('Metagraph file: %s' % meta_file)
    print('Checkpoint file: %s' % ckpt_file)

    # saver = tf.train.import_meta_graph(meta_file, clear_devices=True)

    net = mobilenetv1()
    net.create_architecture("TEST",
                            num_classes=2,
                            tag='default',
                            anchor_width=16,
                            anchor_h_ratio_step=0.7,
                            num_anchors=10)

    saver = tf.train.Saver()
    tf.get_default_session().run(tf.global_variables_initializer())
    tf.get_default_session().run(tf.local_variables_initializer())
    saver.restore(sess, ckpt_file)

    input_graph_def = tf.get_default_graph().as_graph_def()
    print(type(input_graph_def))

    be = [n.name for n in input_graph_def.node]
    print("Before: %d" % len(be))

    # input_graph_def = remove_training_nodes(input_graph_def)
    af = [n.name for n in input_graph_def.node]
    print("After: %d" % len(af))

    variable_names_blacklist = []
    for v in sess.graph.get_operations():
      # print(v.name)
      if 'generate_anchors' in v.name:
        variable_names_blacklist.append(v.name)

    # We only output rpn ops for freeze graph, cas tensorflow can't freeze py_func() ops
    # https://github.com/tensorflow/tensorflow/issues/12073
    # output of RPN and R-CNN
    output_node_names = [net._predictions['rpn_cls_prob'].name[:-2],
                         net._predictions['rpn_bbox_pred'].name[:-2],
                        net._predictions['cls_prob'].name[:-2],
                        net._predictions['bbox_pred'].name[:-2]]

    print("output node names: %s" % output_node_names)

    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(
      sess,  # The session is used to retrieve the weights
      input_graph_def,  # The graph_def is used to retrieve the nodes
      output_node_names,
      variable_names_blacklist=variable_names_blacklist# The output node names are used to select the usefull nodes
    )

    # Serialize and dump the output graph to the filesystem
    output_file = os.path.join(args.output_dir, args.output_name)
    with tf.gfile.GFile(output_file, 'wb') as f:
      f.write(output_graph_def.SerializeToString())
      pb_file_size = f.size() / 1024. / 1024.
    print("%d ops in the final graph: %s, size: %d mb" %
          (len(output_graph_def.node), output_file, pb_file_size))


def get_model_filenames(model_dir):
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    ckpt_file_basename = os.path.basename(ckpt.model_checkpoint_path)
    meta_file = os.path.join(model_dir, ckpt_file_basename + '.meta')
    return meta_file, ckpt.model_checkpoint_path


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--ckpt_dir', type=str, default='../output/mobile/voc_2007_trainval/default',
                      help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')

  parser.add_argument('--output_dir', type=str, default='../output/model',
                      help='Output dir for the exported graphdef protobuf (.pb)')

  parser.add_argument('--output_name', type=str, default='ctpn.pb',
                      help='Filename for the exported graphdef protobuf (.pb)')

  args, _ = parser.parse_known_args()
  common.check_dir(args.output_dir)
  main(args)
