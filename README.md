# CNN-Keras-TF-Surveillance-System
Detect motion for input video source using background subtraction algorithm. Then dynamically extract the images from the video source as test images. This images are fed to trained model "64x3-CNN.model" to predict if the motion is caused by Humans or not. Then display alert message on Security feed.

# Output

## Night Vision vs Thermal:
Object under observation is behind bushes
<img width="615" src="Output2.png">
Objects viewed from sky
<img width="615" src="Output3.png">
                                 
## Background Subtraction:
<img width="628" alt="Screenshot 2019-05-21 at 10 03 21 PM" src="https://user-images.githubusercontent.com/39873118/58114596-b25b5c00-7c15-11e9-83dc-6aa3ff209e26.png">

## Frame Dialation:
<img width="615" alt="Screenshot 2019-05-21 at 10 03 38 PM" src="https://user-images.githubusercontent.com/39873118/58114697-f2bada00-7c15-11e9-8f4f-87c1be790f22.png">

## Final Security Feed:
<img width="625" alt="Screenshot 2019-05-21 at 10 04 03 PM" src="https://user-images.githubusercontent.com/39873118/58114746-0e25e500-7c16-11e9-8fdf-0b5b851c2c55.png">
