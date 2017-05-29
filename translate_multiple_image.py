from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import json
import base64
import cv2
import os


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True, help="directory containing exported model")
parser.add_argument("--device_opts", default="/gpu:0", help="device options")
parser.add_argument("--input_dir", required=True, help="directory containing input PNG image files")
parser.add_argument("--output_dir", required=True, help="directory containing output PNG image files")
a = parser.parse_args()

def main():
    #with open(a.input_file) as f:
    #    input_data = f.read()
    image_list = os.listdir(a.input_dir)
    input_ext = os.path.splitext(image_list[0])[1]
    if input_ext != ".png":
        os._exit()

    if not os.path.isdir(a.output_dir):
        os.makedirs(a.output_dir)


    with tf.Session() as sess:
        with tf.device(a.device_opts):
            saver = tf.train.import_meta_graph(a.model_dir + "/export.meta")
            saver.restore(sess, a.model_dir + "/export")
            input_vars = json.loads(tf.get_collection("inputs")[0])
            output_vars = json.loads(tf.get_collection("outputs")[0])
            input = tf.get_default_graph().get_tensor_by_name(input_vars["input"])
            output = tf.get_default_graph().get_tensor_by_name(output_vars["output"])

            for imagename in image_list:
                image_file = os.path.join(a.input_dir,imagename)
                with open(image_file) as f:
                    input_data = f.read()    
                input_instance = dict(input=base64.urlsafe_b64encode(input_data), key="0")
                input_instance = json.loads(json.dumps(input_instance))
                input_value = np.array(input_instance["input"])
                output_value = sess.run(output, feed_dict={input: np.expand_dims(input_value, axis=0)})[0]

                output_instance = dict(output=output_value, key="0")

                b64data = output_instance["output"].encode("ascii")
                b64data += "=" * (-len(b64data) % 4)
                output_data = base64.urlsafe_b64decode(b64data)

                name = os.path.splitext(imagename)[0]
                output_file = os.path.join(a.output_dir,name+"_trans.png")
                with open(output_file, "w") as f:
                    f.write(output_data)

            print("Done!")
main()