# Working of Drowsy_Driver_Detection_System(DDDS)
Our DDDS analyses the captured frames for the sleepiness detection after receiving an input from the colour video camera mounted in front of the driver. The detection system consists of a video camera and software that periodically checks the driver's eye to measure the length of the eye blink. 

Using the Viola Jones face detector from the OpenCV library, we begin by identifying faces. Then, we utilised the STASM library's neural network-based eye detector to find the locations of the pupils. The STASM is a variant of Coote's implementation of the Active Shape Model. From the STASM library, which consists of a collection of neural networks that supply eye locations, we merely deduced the Rowley's eye detection code for real-time speed restrictions.
The vertical locations of both eyes are used to infer the orientation of the face after eye detection. We calculate the angle between these two pupil sites if they are not in the same location. The entire frame is then rotated in the opposite way to rectify the face's alignment. The face centre serves as the rotation's starting point. By doing this, we may reduce the face's tendency to roll to the left and right by up to 25 degrees. 
Finally, from the pupil area, we extract a rectangle region. The region of interest is scaled to 2015 for normalisation, with the width and height of the zone of interest set, respectively, to 0.5 and 0.16 of Inter Pupillary Distance (IPD).
