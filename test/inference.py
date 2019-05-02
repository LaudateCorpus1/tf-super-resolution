import io
import os
import tensorflow as tf
from scipy.misc import imsave
from PIL import Image
import ai_integration


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
        model_input_path = tf.placeholder(tf.string, [])
        model_output_path = tf.placeholder(tf.string, [])
        data = tf.placeholder(tf.string,shape=[])
        config.gpu_options.allow_growth = True
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
                image = [tf.io.decode_image(image,dtype=tf.uint8,channels = 3)]
                image = tf.cast(image, tf.float32)
                with tf.gfile.GFile("test/4pp_eusr_pirm.pb", 'rb') as f:
                    model_graph_def = tf.GraphDef()#example
                    model_graph_def.ParseFromString(f.read())
     
                model_output = tf.import_graph_def(model_graph_def, name='model', input_map={'sr_input:0': image}, return_elements=['sr_output:0'])[0]
                model_output = model_output[0, :, :, :]
                model_output = tf.round(model_output)
                model_output = tf.clip_by_value(model_output, 0, 255)
                model_output = tf.cast(model_output, tf.uint8)
                image = tf.image.encode_png(model_output)#RIGHT. HERE.
                write_op = tf.write_file(model_output_path, image)
                result_data = {"content-type": 'text/plain',
                               "data": None,
                               "success": False,
                               "error": None}
                image_path_list = []
                image_byte_list = []
                out = 'dummy.png'
                output_path = os.path.join('SR', out)
                input_path = os.path.join('LR', 'bleh.png')
                print('- %s -> %s' % ('', output_path))
                sess.run([write_op], feed_dict={model_input_path:input_path, model_output_path:output_path})
                file = Image.open(output_path,'r')
                imgbytes = save_image_in_memory(file)
                output_img_bytes = imgbytes
                print('Done')
                result_data["data"] = output_img_bytes
                result_data["content-type"] = 'image/jpeg'
                result_data["success"] = True
                result_data["error"] = None
                os.remove(output_path)
                print('Finished inference')
                ai_integration.send_result(result_data)
                
