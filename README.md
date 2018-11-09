# Autopoll
## Electrical Engineering 49 Project

Repository to hold all of the code and files for Boris Fedorov and Dylan Callaway's Fall 2018 EE49 project, AutoPoll.

### Project Proposal

#### Abstract and Features
Choose  a  short  description  of  your  project,  including  the  motivation  and  goal,  major  elements,  and capabilities.  You may type your responses on separate sheets and staple them to this document.  But do not lose the document, you will submit it at semester’s end for final scoring!

The motivation for this project is to explore the possibility of using TensorFlow’s Object Detection API to detect the number of raised hands in a crowd.

As a case study, we are aiming to build a device that would be useful in classrooms as a replacement for iClicker-type devices. At a minimum, we hope to build an accurate raised-hand detection model that reports the results in a useful manner.

The camera will be accessed by a Raspberry Pi and the results will be displayed on a website or phone-app. There may need to be an intermediate computer to run the hand-detection model, depending on the size of the model required to achieve the desired accuracy.


#### Sensing
Acquire  information  from  the  “real  world”  such  as  environmental  parameters,  motion,  location, etc).  Describe how your project leverages sensing.

We will use a webcam connected to a Raspberry Pi.

The data gathered via the webcam (RGB images) will be used by an algorithm known as Faster R-CNN (Faster Region Proposal Convolutional Neural Network (these machine learning people need to chill...)).

The network, which is trained on images of raised hands, will then classify and locate all of the raised hands in the image, and return how many of them there are.


#### Actuation
Acting on the environment with motors, switches, light, sound, . . . .

We will use either a website, phone app, or iClicker type buttons to select what option is being polled.

The results will be displayed on the computer, phone, or device.
Computing
The program orchestrates the show, using data gathered from sensors to control the actuators and send and get data from the cloud.  Explain the main programming tasks, proposed implementation and challenges you foresee.

As described previously in the “Sensing” section, the results will be calculated by a neural network.

There are numerous programming tasks that go into building a neural network, especially ones with the complexity of Faster R-CNN. Luckily, we don’t have to do any of it, because TensorFlow exists, and Google, along with other contributors, made it open source.

The main programming and related tasks that we have to accomplish are:
Capture and annotate ~500 training and ~50 test images.
Create a pipeline to convert .jpg images and .xml annotations into the native TensorFlow file format, TFRecord (.record).
Select a model architecture.
We will use Faster R-CNN, but there are many implementations like Resnet 50/100, Inception V1/2, NAS, etc.
Write a configuration file for the selected model architecture (.json).
The model is made by TensorFlow, but there are many user-definable parameters that must be set before it can be used.
Create a pipeline for training the model on the TFRecords images.
Create a pipeline for evaluating the performance of the model on the test images.
Train the model until the accuracy meets our requirements.
Collect more images if results are not good enough.
Create an inference pipeline on the Raspberry Pi.
Option 1: Capture image, run inference, send results (probably using MQTT).
Option 2: Capture image, send image to another computer with a GPU, other computer sends results (probably using MQTT).
Option 3: Capture image, send image to faster CPU, run inference on faster CPU, send results using SSH or sockets.
Create a phone app, website, or write the firmware for a button device to run the poll.


#### Communication
Communicate (wirelessly).

We will probably use MQTT to send/receive images and results data. We were not able to use MQTT to accomplish what we needed.

We will use SSH or sockets (TCP/IP) to send the image and results data.


#### Parts List

Raspberry Pi Zero W
Raspberry Pi Mini Spy Camera
Camera/board enclosure (3D printed)


#### Milestone 1
Describe the objectives for the first milestone.  E.g. parts ordered/received, hardware assembled, software written and tested.

Our objectives for Milestone 1 are to capture and annotate images of raised hands, and use them to train the detection model. This means we will need to have written most of the conversion and training pipelines discussed above. For annotations, we are going to use the open source application LabelImg.

We also hope to have tested the hardware and software setup on the Raspberry Pi (ensuring webcam functionality (USB webcam did not work, ordered a Pi cam), installing TensorFlow, running inference with a pretrained model), just to make sure we can move all of our code onto it later and have it still work.

We also hope that our initial setup and parameters will be sufficient, and that after the model is trained, we will have detected our first raised hand.


#### Milestone 2
Describe the objectives for the second milestone.  List major functionality and your testing plan.

Since inference is working on our personal laptops from Milestone 1, we will move it fully to the Raspberry Pi. This will allow us to test the detection speed and make and necessary changes to the model to allow inference to be run locally on the Raspberry Pi.

If this is possible (doubtful), we will write all of the companion software that allows the Raspberry Pi to capture and send images at the correct time.

If not possible, we will write the image transfer pipeline and run inference on our computers with GPUs like we had been doing before.

At this time we will also have made the decision between a phone app, website, or button device, and will create that, as well.

We will also 3D print the camera enclosure for the Raspberry Pi and USB webcam, and any other hardware components.


#### Final Project Presentation and Demo
Describe what you planning to present and demonstrate.

As mentioned in the abstract, we are using the classroom poll as a case study to determine if this type of device can be made accurate enough to perform its intended function.

On demo day, after a quick presentation on the device and how we designed it, we will ask 5 - 25 students to move their chairs into a classroom-esque orientation and run a quick poll using the device. We will evaluate the accuracy, add students to the group, and move them closer together until the device accuracy drops significantly.

This simulates a classroom environment and will provide an answer to the questions which we posed at the beginning of the project.
