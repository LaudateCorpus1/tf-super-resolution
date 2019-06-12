import io
import os
import tensorflow as tf
from scipy.misc import imsave
from PIL import Image
import ai_integration


# TODO ensure model loads only once
# TODO no temp files


def save_image_in_memory(image):
    image = image.convert('RGB')
    imgByteArr = io.BytesIO()
    imsave(imgByteArr, image, 'JPEG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def initialize_model():
    with tf.Graph().as_default():
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        with tf.gfile.GFile("test/4pp_eusr_pirm.pb", 'rb') as f:
            model_graph_def = tf.GraphDef()
            model_graph_def.ParseFromString(f.read())

        config.gpu_options.per_process_gpu_memory_fraction = 0.12
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True
        ))
        sess.run(init)

        print('Initialized model')
        while True:
            with ai_integration.get_next_input(inputs_schema={
                "image": {
                    "type": "image"
                }
            }) as inputs_dict:
                print("post sess declare")
                image = inputs_dict["image"]
                image = [tf.image.decode_image(image, dtype=tf.uint8, channels=3)]
                image = tf.cast(image, tf.float32)

                model_output = tf.import_graph_def(model_graph_def, name='model', input_map={'sr_input:0': image},
                                                   return_elements=['sr_output:0'])[0]
                model_output = model_output[0, :, :, :]
                model_output = tf.round(model_output)
                model_output = tf.clip_by_value(model_output, 0, 255)
                model_output = tf.cast(model_output, tf.uint8)
                image = tf.image.encode_png(model_output)
                result_data = {"content-type": 'text/plain',
                               "data": None,
                               "success": False,
                               "error": None}

                (png_bytes) = sess.run([image], feed_dict={})
                output_img_bytes = png_bytes
                print('Done')
                result_data["data"] = output_img_bytes
                result_data["content-type"] = 'image/png'
                result_data["success"] = True
                result_data["error"] = None
                print('Finished inference')
                ai_integration.send_result(result_data)
