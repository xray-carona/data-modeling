import cv2
import numpy as np
import glob
import os

# If tensorflow 2.0 is installed
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

#If tensorflow 1.0 is installed
import tensorflow as tf

class Evaluator(object):
    def __init__(self, model_dir):
        self.labels = ["Normal", "Non-COVID19 Viral", "COVID-19 Viral"]
        self.INPUT_SIZE = (224,224)
        self.MODEL_GRAPH = model_dir + "model.meta_eval"
        self.MODEL = model_dir + "model-2069"
        
    def load_img(self, img):
        return cv2.imread(img)
        
    def pre_process(self, img):
        img_arr = cv2.resize(img, self.INPUT_SIZE) # resize
        img_arr = img_arr.astype('float32') / 255.0
        img_arr = img_arr.reshape(1, self.INPUT_SIZE[0], self.INPUT_SIZE[1], 3)
        return img_arr
    
    def evaluate(self, img):
        tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(self.MODEL_GRAPH)
            saver.restore(sess, self.MODEL)
            graph = tf.get_default_graph()

            # Get tensor names
            x = graph.get_tensor_by_name("input_1:0")
            op_to_restore = graph.get_tensor_by_name("dense_3/Softmax:0")

            # Preprocess image input
            img = self.load_img(img)
            processed_img = self.pre_process(img)
            feed_dict ={x: processed_img}
            result_index = sess.run(op_to_restore,feed_dict)
            return self.labels[np.argmax(result_index)]
        sess.close()
        
if __name__ == "__main__":
    model = Evaluator("model/COVID-Net Large/")
    sample_test = model.evaluate("data/test/covid-19-pneumonia-15-PA.jpg")
    print(sample_test)
