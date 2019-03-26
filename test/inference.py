import io
import time
import traceback

import numpy as np
import tensorflow as tf
from adain.coral import coral
from adain.image import load_image, prepare_image
from adain.nn import build_vgg, build_decoder
from adain.norm import adain
from adain.weights import open_weights
from scipy.misc import imsave


def save_image_in_memory(image, data_format='channels_first'):
    if data_format == 'channels_first':
        image = np.transpose(image, [1, 2, 0])  # CHW --> HWC
    image *= 255
    image = np.clip(image, 0, 255)
    imgByteArr = io.BytesIO()
    imsave(imgByteArr, image.astype(np.uint8), 'JPEG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


vgg = None
encoder = None
decoder = None
target = None
weighted_target = None
image = None
content = None
style = None
persistent_session = None
data_format = 'channels_first'


def _build_graph(vgg_weights, decoder_weights, alpha, data_format):
    if data_format == 'channels_first':
        image = tf.placeholder(shape=(None, 3, None, None), dtype=tf.float32)
        content = tf.placeholder(shape=(1, 512, None, None), dtype=tf.float32)
        style = tf.placeholder(shape=(1, 512, None, None), dtype=tf.float32)
    else:
        image = tf.placeholder(shape=(None, None, None, 3), dtype=tf.float32)
        content = tf.placeholder(shape=(1, None, None, 512), dtype=tf.float32)
        style = tf.placeholder(shape=(1, None, None, 512), dtype=tf.float32)

    target = adain(content, style, data_format=data_format)
    weighted_target = target * alpha + (1 - alpha) * content

    with open_weights(vgg_weights) as w:
        vgg = build_vgg(image, w, data_format=data_format)
        encoder = vgg['conv4_1']

    if decoder_weights:
        with open_weights(decoder_weights) as w:
            decoder = build_decoder(weighted_target, w, trainable=False,
                                    data_format=data_format)
    else:
        decoder = build_decoder(weighted_target, None, trainable=False,
                                data_format=data_format)

    return image, content, style, target, encoder, decoder


def initialize_model():
    global vgg
    global encoder
    global decoder
    global target
    global weighted_target
    global image
    global content
    global style
    global persistent_session
    global data_format
    alpha = 1.0

    graph = tf.Graph()
    #original code start
    
    if (not args.use_gpu):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  
  # load and build graph
  with tf.Graph().as_default():
    model_input_path = tf.placeholder(tf.string, [])
    model_output_path = tf.placeholder(tf.string, [])
    
    image = tf.read_file(image)
    image = [tf.image.decode_png(image, channels=3, dtype=tf.uint8)]
    image = tf.cast(image, tf.float32)
    #directly take in model instead of using --modelname
    with tf.gfile.GFile("4pp_eusr_pirm.pb", 'rb') as f:
      model_graph_def = tf.GraphDef()
      model_graph_def.ParseFromString(f.read())
    
    model_output = tf.import_graph_def(model_graph_def, name='model', input_map={'sr_input:0': image}, return_elements=['sr_output:0'])[0]
    
    model_output = model_output[0, :, :, :]
    model_output = tf.round(model_output)
    model_output = tf.clip_by_value(model_output, 0, 255)
    model_output = tf.cast(model_output, tf.uint8)
    
    image = tf.image.encode_png(model_output)
    write_op = tf.write_file(model_output_path, image)
    
    init = tf.global_variables_initializer()
    
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=False,
        allow_soft_placement=True
    ))
    sess.run(init)
  
    #line 46, test.py^ , original code over

    # build the detection model graph from the saved model protobuf
    

        target = adain(content, style, data_format=data_format)
        weighted_target = target * alpha + (1 - alpha) * content

        with open_weights('models/vgg19_weights_normalized.h5') as w:
            vgg = build_vgg(image, w, data_format=data_format)
            encoder = vgg['conv4_1']

        with open_weights('models/decoder_weights.h5') as w:
            decoder = build_decoder(weighted_target, w, trainable=False, data_format=data_format)

        # the default session behavior is to consume the entire GPU RAM during inference!
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.12

        # the persistent session across function calls exposed to external code interfaces
        persistent_session = tf.Session(graph=graph, config=config)

        persistent_session.run(tf.global_variables_initializer())

    print('Initialized model')


def infer(inputs_dict):
    global data_format
  # get image path list - original code
  image_path_list = []
  for root, subdirs, files in os.walk(args.input_path):
    for filename in files:
      if (filename.lower().endswith('.png')):
        input_path = os.path.join(args.input_path, filename)
        output_path = os.path.join(args.output_path, filename)

        image_path_list.append([input_path, output_path])
  print('Found %d images' % (len(image_path_list)))
  
  # iterate
  for input_path, output_path in image_path_list:
    print('- %s -> %s' % (input_path, output_path))
    sess.run([write_op], feed_dict={model_input_path:input_path, model_output_path:output_path})
  #original code over
  print('Done')
    # only update the negative fields if we reach the end of the function - then update successfully
    result_data = {"content-type": 'text/plain',
                   "data": None,
                   "success": False,
                   "error": None}

    try:
        print('Starting inference')
        start = time.time()

        content_size = 512
        style_size = 512
        crop = False
        preserve_color = False

        content_image = load_image(io.BytesIO(inputs_dict['content']), content_size, crop)


        if preserve_color:
            style_image = coral(style_image, content_image)
        style_image = prepare_image(style_image)
        content_image = prepare_image(content_image)
        style_feature = persistent_session.run(encoder, feed_dict={
            image: style_image[np.newaxis, :]
        })
        content_feature = persistent_session.run(encoder, feed_dict={
            image: content_image[np.newaxis, :]
        })
        target_feature = persistent_session.run(target, feed_dict={
            content: content_feature,
            style: style_feature
        })

        output = persistent_session.run(decoder, feed_dict={
            content: content_feature,
            target: target_feature
        })

        output_img_bytes = save_image_in_memory(output[0], data_format=data_format)

        result_data["content-type"] = 'image/png'
        result_data["data"] = output_img_bytes
        

        print('Finished inference and it took ' + str(time.time() - start))
        return result_data


    except Exception as err:
        traceback.print_exc()
        result_data["error"] = traceback.format_exc()
        return result_data