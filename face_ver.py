from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.graphics.texture import Texture

import cv2
import os
import numpy as np
import tensorflow as tf
from cs_objects import l1_distance

class CamApp(App):
    def build(self):
        #modules
        self.web_cam = Image(size_hint=(1,0.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,0.1))
        self.verification_label = Label(text="Initiate Verification", size_hint=(1,0.1))

        #add modules as widgets to the box-layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        #instantiate model
        self.model = tf.keras.models.load_model('Siamese-Recognizer.h5', custom_objects={'l1_distance':l1_distance})
        
        #setup webcam
        self.cap = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout

    def preprocess(self,pth):
        byte_img=tf.io.read_file(pth)
        jpg_img=tf.io.decode_jpeg(byte_img)
        img=tf.image.resize(jpg_img,(100,100))
        img=img/255.0
        return img

    def update(self, *args):
        ret, frame = self.cap.read()
        frame = frame[120:120+250, 200:200+250,:]
        buff=cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buff, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture
    
    def verify(self, *args):
        detection_threshold=0.99
        verification_threshold=0.8

        input_save_path = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame=self.cap.read()
        frame = frame[120:120+250, 200:200+250,:]
        cv2.imwrite(input_save_path, frame)

        results=[]
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            inp = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            val = self.preprocess(os.path.join('application_data', 'verification_images', image))
            result = self.model.predict(list(np.expand_dims([inp,val], axis=1)))
            results.append(result)
        
        detected = np.sum(np.array(results)>detection_threshold)
        verification = detected/len(os.path.join('application_data', 'verification_images'))
        verified = (verification > verification_threshold)

        self.verification_label.text = "Verified" if (verified==True) else "Unverified"

        Logger.info(detected)
        Logger.info(verified)
        
        return results, verified

if __name__=='__main__':
    CamApp().run()