import io
import time
import traceback
import os
import numpy as np
import tensorflow as tf
from scipy.misc import imsave, imread
#from scipy import imageio
from PIL import Image
import ai_integration
def save_image_in_memory(image, data_format='channels_first'):
    #image = image.convert('RGB')
    imgByteArr = io.BytesIO()
    imsave(imgByteArr, image, 'JPEG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr
def convert_to_png(image):
    #image = image.convert('RGB')
    imgByteArr = io.BytesIO()
    imsave(imgByteArr, image, 'PNG')
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


def initialize_model():
    with tf.Graph().as_default():
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.12
        #tf.print(image)
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
        #test2 code start
                print(tf.io.decode_raw(inputs_dict["image"],out_type = tf.uint8))
                print("post sess declare")
                #print(data)
                
                model_input_path = tf.placeholder(tf.string, [])
                model_output_path = tf.placeholder(tf.string, [])
                data = tf.placeholder(tf.string,shape=[])
                #image = tf.read_file(inputs_dict["image"])
                image = inputs_dict["image"]
                print("initial image ",image)
                image = image*4
                print("post *4 ",image)
                image = tf.io.decode_raw(image,out_type = tf.float32)
                #image/=4
                print("post decode ",image)
                image = tf.reshape(image,[-1,1,1,1])
                print("post reshape",image)
                #image = tf.cast(image, tf.float32)
                #image = tf.read_file(model_input_path)
                #image = inputs_dict["image"]
                #image = [tf.image.decode_png(image, channels=3, dtype=tf.uint8)]
                
                
                #image = imread(image, mode='RGB') 
                #image = convert_to_png(image)
                #print(image)
                #image = tf.cast(image, tf.float32)
                #print(image)
               # print(io.BytesIO(inputs_dict['content'])
                #image = image.astype(np.float32)
                #image = tf.reshape(image,[600,400,3]) 
                #image /= 255
                with tf.gfile.GFile("test/4pp_eusr_pirm.pb", 'rb') as f:
                    model_graph_def = tf.GraphDef()#example
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
                write_op = tf.write_file(model_output_path, image)
                #image = tf.image.adjust_saturation(tf.io.decode_png(image),float(100))
    #experiment time
                #print(image)
    #image = tf.io.decode_png(image)
    #image = tf.image.encode_jpeg(image)
                #print(image)
    #ttt = image.eval()
                
        #end test2 code
               

    

            # only update the negative fields if we reach the end of the function - then update successfully
                result_data = {"content-type": 'text/plain',
                               "data": None,
                               "success": False,
                               "error": None}

                image_path_list = []
                image_byte_list = []
                
                #for root, subdirs, files in os.walk('LR'):
                   # for filename in files:
                   #     if (filename.lower().endswith('.png')):
                   #         input_path = os.path.join('LR', filename)
                    #        output_path = os.path.join('SR', filename)
                    #        image_path_list.append([input_path, output_path])
                #print('Found %d images' % (len(image_path_list)))
  #global data_format
  # iterate
                #if image_path_list != []:
                   # for input_path, output_path in image_path_list:
                output_path = os.path.join('SR', 'test.png')
                input_path = os.path.join('LR', 'bleh.png')
                print('- %s -> %s' % ('', 'SR/test.png'))
                sess.run([write_op], feed_dict={model_input_path:input_path, model_output_path:output_path})
                file = Image.open(output_path,'r')
                imgbytes = save_image_in_memory(file)
     
                print(imgbytes)
                output_img_bytes = imgbytes
                print('Done')
                result_data["data"] = output_img_bytes
 #   image_byte_list.append(imgbytes)
    #print(imgbytes)
    #test=sess.run()
    #print(test)
    #sess.run([toot],feed_dict={data:image})
    #with sess.as_default():
        #print(image.eval())
                
                result_data["content-type"] = 'image/jpeg'
                result_data["success"] = True
                result_data["error"] = None
                os.remove(output_path)
                print('Finished inference')
                ai_integration.send_result(result_data)
                #except Exception as err:
                  #  print('image path is literally empty wth')
                   # result_data["error"] = 'failed'
                   # ai_integration.send_result(result_data)
                

                
                
            