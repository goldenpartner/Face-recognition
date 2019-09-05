# Face-recognition
张江人工智能岛照片识别

Download and extract the folowwing files to the source directory

http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

Usage

1-Resize.ipynb will resize all images in a directory and subdirectories. It will save all resized images in a new place. DO NOT put the result folder in the source folder as it will become an infinite recursion

2-Get_face_descriptor.ipynb will give a pkl file contains face descriptors for photos, if an image contain more than one faces, multiple records will be inserted into the file

3-distance.ipynb contains functions to find images with similar faces providing an original image

4-Video chopping.ipynb chops videos into images twice every second, this will take very long time to run

faces_all.pkl is a prepared dataset ready to use. 
