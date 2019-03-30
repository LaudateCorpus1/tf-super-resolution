#from PIL import Image
import argparse
import os

import numpy as np
import tensorflow as tf
#This is for testing stuff, like the stuff used for deepai stuff. haha yes.

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# params
parser = argparse.ArgumentParser()
#parser.add_argument('--model_name', default='model.pb', help='path of the model file (.pb)')
parser.add_argument('--input_path', default='LR', help='base path of low resolution (input) images')
parser.add_argument('--output_path', default='SR', help='base path of super resolution (output) images')
parser.add_argument('--use_gpu', action='store_true', help='enable GPU utilization (default: disabled)')
args = parser.parse_args()

def save_image_in_memory(image):
    #if data_format == 'channels_first':
        #image = np.transpose(image, [1, 2, 0])  # CHW --> HWC
    print(image)
    image *= 255
    image = np.clip(image, 0, 255)
    imgByteArr = io.BytesIO()
    imsave(imgByteArr, image.astype(np.uint8), 'JPEG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

def main():
  if (not args.use_gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  
  # load and build graph
  with tf.Graph().as_default():
    model_input_path = tf.placeholder(tf.string, [])
    model_output_path = tf.placeholder(tf.string, [])
    data = tf.placeholder(tf.string,shape=[])
    
    image = tf.read_file(model_input_path)
    image = [tf.image.decode_png(image, channels=3, dtype=tf.uint8)]
    image = tf.cast(image, tf.float32)
    
    with tf.gfile.GFile("4pp_eusr_pirm.pb", 'rb') as f:
      model_graph_def = tf.GraphDef()
      model_graph_def.ParseFromString(f.read())
    
    model_output = tf.import_graph_def(model_graph_def, name='model', input_map={'sr_input:0': image}, return_elements=['sr_output:0'])[0]
    print(model_output)
    model_output = model_output[0, :, :, :]
    model_output = tf.round(model_output)
    model_output = tf.clip_by_value(model_output, 0, 255)
    model_output = tf.cast(model_output, tf.uint8)
    print(model_output)
    image = tf.image.encode_png(model_output)#RIGHT. HERE.
    #image = tf.image.random_brightness(image)
    
    #image = tf.image.encode_png(image)
    write_op = tf.write_file(model_output_path, image)#it's literally right here smartass
    image = tf.image.adjust_saturation(tf.io.decode_png(image),float(100))
    #experiment time
    print(image)
    #image = tf.io.decode_png(image)
    print(image)
    #ttt = image.eval()
    init = tf.global_variables_initializer()
    tf.print(image)
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=False,
        allow_soft_placement=True
    ))
    print("post sess declare")

    print(data)
    sess.run(init)
    #print(sess.run([toot],feed_dict={data:image}))
    #with sess.as_default():   # or `with sess:` to close on exit
        #assert sess is tf.get_default_session()
        #assert image.eval() == sess.run(image)
  # get image path list
  image_path_list = []
  image_byte_list = []
  for root, subdirs, files in os.walk(args.input_path):
    for filename in files:
      if (filename.lower().endswith('.png')):
        input_path = os.path.join(args.input_path, filename)
        output_path = os.path.join(args.output_path, filename)

        image_path_list.append([input_path, output_path])
  print('Found %d images' % (len(image_path_list)))
  #global data_format
  # iterate
  for input_path, output_path in image_path_list:
    print('- %s -> %s' % (input_path, output_path))
    sess.run([write_op], feed_dict={model_input_path:input_path, model_output_path:output_path})
    #file = Image.open(output_path,'r')
    #imgbytes = save_image_in_memory(test)
 #   image_byte_list.append(imgbytes)
    #print(imgbytes)
    #test=sess.run()
    #print(test)
    #sess.run([toot],feed_dict={data:image})
    #with sess.as_default():
        #print(image.eval())
  print('Done')


if __name__ == '__main__':
  main()