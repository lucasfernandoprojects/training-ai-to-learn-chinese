# Training AI to Learn Chinese

*I've recently posted a tutorial about this project on YouTube. You can watch it [here](https://www.youtube.com/watch?v=XQRtSKdzxjc).*

![Thumbnail of the video training AI to learn Chinese](https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/thumbnails/thumbnail-final-version.png)

This is the Chinese character for half (半). What you’re in the images below is the result of a fascinating experiment: I trained an AI to recognize handwritten Mandarin characters using a dataset I created myself. Then, I connected everything to an Arduino to make it interactive.

<div style="display: flex; flex-wrap: wrap;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/2.jpg" width="400" height="250" style="margin: 10px;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/3.jpg" width="400" height="250" style="margin: 10px;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/4.jpg" width="400" height="250" style="margin: 10px;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/5.jpg" width="400" height="250" style="margin: 10px;">
</div>
</br>

Running inference with the Arduino controller.

<div style="display: flex; flex-wrap: wrap;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/6.jpg" width="400" height="250" style="margin: 10px;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/7.jpg" width="400" height="250" style="margin: 10px;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/8.jpg" width="400" height="250" style="margin: 10px;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/9.jpg" width="400" height="250" style="margin: 10px;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/10.jpg" width="400" height="250" style="margin: 10px;">
</div>
</br>

This entire system runs on a budget PC I built for under $300. The project combines data science, computer vision, coding, and electronics, and it started with a simple idea:

*Could I teach a machine something I barely understood myself?*

Let me walk you through how I made it happen.

## Why I Built This

Some time ago, I started learning Simplified Mandarin. It’s been one of the hardest challenges I’ve taken on: the tones, the stroke order, memorizing characters - it’s a whole new way of thinking compared to my native language Portuguese.

And somewhere along the way, while struggling to memorize a character for the fifth time, I thought:

*What if I tried to teach Mandarin to an AI?*

I wasn’t just curious about whether it would work. I wanted to understand how an AI learns and whether its learning process could teach me something about my own. That’s when the idea took off.

## The Plan: AI Image Classification

Firstly, I had to decide what kind of artificial intelligence I would build. From the dozens of options available, I focused on just one: image classification.

In a few words, image classification is the categorization of images based on specific rules. The central idea of this technique is that, instead of telling the computer what the rules are, it uses special algorithms to learn those rules by itself.

Let me give you an example. Imagine I have a task that requires me to build a computer program that classifies photos as either “dog” or “cat”. In this scenario, instead of explicitly telling the computer what a dog or a cat looks like, I show it a bunch of photos of these animals and let the program figure out the concepts of “dog” and “cat” by itself.

![How image classification works](https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/schematics/how-image-classification-works.png)

This special class of algorithms that allows machines to learn by interacting with data is called Machine Learning. My goal was to train an AI model that could recognize 10 different Chinese characters based on real-world photos.

But first, I needed data.

## Collecting the Dataset

The first step towards building an AI model is data collection. The goals of your project will determine the kind of data you need. Sometimes, you can find open-source datasets on the internet that suit your necessities. This makes development faster since someone else has already collected the information.

But in many cases, that's not possible. In my case, I had to build the dataset myself. To accomplish this, I wrote down each of the 10 Mandarin characters multiple times using pencils and pens. Then, I took 100 photos of each character using a standard USB webcam.

The result: a dataset of 1,000 images, with natural variations in lighting, angle, and handwriting.

I split the dataset into:

+ **80% training data** (to teach the model)
+ **10% validation data** (to evaluate during training)
+ **10% test data** (to evaluate after training)

Once the dataset was ready and organized, it was time to train the model.

<div style="display: flex; flex-wrap: wrap;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/12.png" width="400" height="250" style="margin: 10px;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/13.png" width="400" height="250" style="margin: 10px;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/14.png" width="400" height="250" style="margin: 10px;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/15.png" width="400" height="250" style="margin: 10px;">
</div>
</br>

## Training AI on a $300 PC

Training is the stage where the AI learns from data. There are a lot of technical details about this topic I'll skip here because they deserve their own tutorial. But since this is a README, I’ll include some behind-the-scenes context for those interested in how the system works.

### Hardware Specs

AI training usually happens in the cloud because it requires a lot of memory and processing power, but cloud services are expensive. To keep things simple (and cheap), I trained the model locally on a PC I built myself for under $300.

Here's the specs:

+ **CPU**: Intel Xeon E5-2670 v3 @ 2.30GHz
+ **RAM**: 16GB DDR4 @ 2133 MHz
+ **GPU**: Nvidia GT 1030 (2GB)
+ **Storage**: 512GB 2.5” SSD
+ **Power Supply**: Corsair CV650 80 Plus Bronze, 650W
+ **Operating System**: Ubuntu 24.04.2 LTS

It’s far from a top-tier workstation, but with the right optimizations, it got the job done.

*If you are interesting in this PC, I posted [a video on YouTube](https://www.youtube.com/watch?v=LfR-cdyeaEk&t=5s) explaining how I built it.*

![Thumbnail of the custom $300 dollars PC video](https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/thumbnails/custom-pc-for-300-dollars-video-thumbnail.jpg)

### The Model Architecture

To make this work, I needed a model that was fast and lightweight. From the countless algorithms that exist for training, I chose MobileNetV2.

MobileNet is a lightweight Convolutional Neural Network (CNN) designed to run AI on small devices with limited computational resources. But not only deployment, its training doesn’t require huge resources either, which makes it perfect for this project.

The model is built using **TensorFlow** and consists of:

+ A pre-trained **MobileNetV2 base** (with include_top=False and pooling='avg')
+ A custom **Dense(128)** layer with ReLU activation and L2 regularization
+ A **Dropout(0.4)** layer to prevent overfitting
+ A final **Dense(num_classes)** output layer with Softmax activation

The base model was frozen during initial training, and only the top layers were updated.

You can see the architecture visualized in plots/model_architecture.png.

### Data Pipeline

Data was augmented and loaded using ImageDataGenerator with the following settings:

+ Rescaled inputs (1./255)
+ Random rotation, width/height shift, shear, zoom
+ Light brightness variation to simulate natural lighting
+ Image size: 96x96
+ Batch size: 32
+ Class mode: categorical
+ Color mode: rgb

### Training Configuration

Initial training parameters I applied:

+ Optimizer: Adam with learning rate 0.001
+ Loss: Categorical Crossentropy
+ Epochs: 50
+ Callbacks:
  + EarlyStopping on val_loss (patience: 5)
  + ModelCheckpoint to save best model
  + ReduceLROnPlateau to lower LR when plateauing

```
EarlyStopping(patience=5, min_delta=0.001, restore_best_weights=True)
ModelCheckpoint(monitor='val_accuracy', save_best_only=True)
ReduceLROnPlateau(factor=0.2, patience=3)
```

### Fine-Tuning the Base Model

After the initial training phase, I unfroze the last 8 layers of the MobileNetV2 base and ran a short fine-tuning session with:

+ Lower learning rate (1e-5)
+ 10 additional epochs
+ More aggressive early stopping

```
for layer in model.layers[0].layers[-8:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True
```

### Results

I installed CUDA and cuDNN to leverage my Nvidia GPU on Ubuntu. Setting up the environment was time-consuming, but eventually everything worked.

Then came the moment of truth.

I ran the training script, and to my surprise, the model reached **85% accuracy on the test set on the first try**.

I expected way worse. I assumed the dataset was too small. But the model kept improving:

+ **Second run**: 87%
+ **Third run**: 89%
+ **With fine-tuning**: nearly 100% test accuracy

You can see the performance of the last training (with fine-tuning) on the training x validation accuracy and loss curves below. I also added the confusion matrix for reference as well.

![Accuracy curve](https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/plots/accuracy_curve.png)

![Loss curve](https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/plots/loss_curve.png)

![Confusion matrix](https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/plots/confusion_matrix.png)

That shocked me. I was expecting much worse performance because of the limitations I discussed earlier. I thought I’d need to take hundreds - or thousands - more photos to even approach 80% accuracy. But somehow, it was already doing great.

This was fascinating and also a little scary.

Mandarin has been one of the hardest things I’ve ever studied. It’s completely different from my mother tongue. It felt like I was learning to speak again from scratch - like a baby.

Naturally, I assumed the AI would struggle just as much. But instead, it adapted quickly.

## Real-World Testing

To evaluate the model in action, I wrote a Python script that:

1. Loads the trained model
2. Accesses the webcam
3. Waits for user input
4. Captures a photo
5. Sends it to the model
6. Displays the prediction on screen

And it worked beautifully.

<div style="display: flex; flex-wrap: wrap;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/16.png" width="400" height="250" style="margin: 10px;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/17.png" width="400" height="250" style="margin: 10px;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/18.png" width="400" height="250" style="margin: 10px;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/19.png" width="400" height="250" style="margin: 10px;">
</div>
</br>

Pretty cool, isn’t it? Of course, the model doesn’t get everything right, because no model is perfect. Even though it reached nearly 100% during training, it still makes mistakes.

But the goal isn’t to be perfect. It’s to be useful.

Like George E. P. Box said:

> All models are wrong, but some are useful.

Even though AI models are simplifications of reality - and they always carry some imperfections - they can still be powerful tools in the real world.

## Bringing in Arduino

The project could’ve ended there, but I wanted to take it further. I decided to build an AI controller using Arduino.

The Arduino side of the project is responsible for handling user input, navigating menus, and displaying both AI recognition results and learning content on a small TFT screen. It runs on an Arduino Nano connected to a ST7789 IPS screen, 3 push buttons, and 2 indicator LEDs (green and red) - you can use a RGB LED as well.

*If you need help at setting up the ST7789 display, I recorded [a tutorial](https://www.youtube.com/watch?v=cxj0jDbT5vc&t=9s) about this.*

![Thumbnail of the ST7789 display video](https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/thumbnails/st7789-video-thumbnail.jpg)

### Navigation and Button Mapping

The system is controlled using three push buttons, each with a clearly defined function:

+ **Right Button (Pin 2) → Scroll Down**
  + Used to move down in menus or lists.

+ **Middle Button (Pin 3) → Scroll Up**
  + Used to move up in menus or lists.

+ **Left Button (Pin 6) → Select / Enter**
  + Confirms the selected menu item, enters character view, or captures a photo during inference.

These buttons let you easily navigate through menus, view Chinese characters, or initiate photo-based recognition.

### Navigation Flow

When the Arduino boots up, the following interface options are shown on the display:

**Main Menu Options**

+ **Inference**: Activates photo mode. The AI system waits for a photo input via serial (triggered by the button). It then processes the image and sends back the recognized character name, which is shown on the screen.

+ **Learn**: Opens a scrollable list of Mandarin characters stored locally in the code. Selecting one will show its Hanzi, Pinyin, and meaning (the full bitmap). Perfect for self-study or offline revision.

+ **About**: Displays author credits and project version info.

To make this work, I wrote a second Python script to handle serial communication and inference logic. And as you can see below, it worked pretty well.

**Main menu**

<div style="display: flex; flex-wrap: wrap;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/20.jpg" width="400" height="250" style="margin: 10px;">
</div>
</br>

**Inference Mode**

When "Inference" is selected:

+ The screen shows **“Waiting for photo...”**
+ Pressing the **enter button** (left) triggers a CAPTURE message over serial to the PC
+ The PC-side Python script captures the image and returns a class prediction in the format: RESULT:ban_4

Once received, the screen displays:

+ The recognized Chinese character (using a bitmap)
+ A green LED lights up (success)
+ After 5 seconds, the system auto-returns to photo mode

The **down button** (right) can be pressed at any time to exit back to the main menu.

<div style="display: flex; flex-wrap: wrap;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/21.jpg" width="400" height="250" style="margin: 10px;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/22.jpg" width="400" height="250" style="margin: 10px;">
</div>
</br>

**Learn Mode**

When “Learn” is selected:

+ You’ll see a list of Pinyin entries (e.g. ban, ren, yi, etc.)
+ Use **up (middle)** and **down (right)** buttons to scroll
+ Press **enter (left)** to view the selected character

Each character view displays:

+ A large bitmap version of the Mandarin symbol
+ Its **Pinyin** pronunciation and **English meaning**
+ An instruction to press **enter** to go back to the list

At the bottom of the list is the **Go back** option to return to the main menu.

<div style="display: flex; flex-wrap: wrap;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/23.jpg" width="400" height="250" style="margin: 10px;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/24.jpg" width="400" height="250" style="margin: 10px;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/25.jpg" width="400" height="250" style="margin: 10px;">
</div>
</br>

**About Screen**

Displays simple credits and version information:

+ Creator name
+ GitHub open-source notice
+ YouTube call to action

Press **enter (left)** to return to the main menu.

<div style="display: flex; flex-wrap: wrap;">
    <img src="https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/26.jpg" width="400" height="250" style="margin: 10px;">
</div>
</br>

**Visual Feedback**

Two onboard LEDs provide real-time status:

+ **Red LED ON**: The system is in inference mode and ready to capture a photo.
+ **Green LED ON**: A result was received and displayed successfully.
+ **Both OFF**: Idle mode or main menu/learning screen.

### Hardware Connections

| Component | Pin |
| --------- | --- |
| TFT CS | D10 |
| TFT DC | D8 |
| TFT RST | D9 |
| TFT BL | D7 (always on) |
| Green LED | D4 |
| Red LED | D5 |
| Enter Button | D6 |
| Up Button | D3 |
| Down Button | D2 |

*All buttons use INPUT_PULLUP logic and are **active LOW**.*

**Schematics**

If you want to replicate the hardware setup, here is the schematics:

![Arduino controller schematics](https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/schematics/arduino-controller.png)

And here is the materials list:

+ 1 x Breadboard
+ 1 x Arduino Nano
+ 1 x ST7789 IPS display (8 pins)
+ 1 x Red LED (5mm)
+ 1 x Green LED (5mm)
+ 2 x 220Ω resistors
+ 3 x push buttons
+ Male-to-male jumpers wires

Just assemble everything on a breadboard, connect the components according to the diagram, upload the sketch to Arduino, and you're good to go.

This user interface gives the AI project a full offline interactive experience. You can teach yourself Chinese characters or let the AI recognize them live, all with just a few clicks.

![Arduino controller over a table](https://github.com/lucasfernandoprojects/training-ai-to-learn-chinese/blob/main/photos/project/28.jpg)

## What I Learned

This was the first time I built an end-to-end AI project, from collecting data to building a real-world interface. And it was one of the most satisfying accomplishments I’ve ever done.

I learned:

+ How to create a dataset from scratch
+ How to train and evaluate a deep learning model locally
+ How to connect AI with physical computing through Arduino

But beyond the technical skills, this project helped me reflect on what Artificial Intelligence is really for.

In a time when AI is rapidly changing the world by automating jobs and reshaping industries, we need to ask ourselves:

*Who is this technology being built for? And most importantly, how do we make sure it benefits everyone?*

Learning about Machine Learning and creating AI tools are only meaningful if we use that knowledge to make the world better. Because if we build a future where AI concentrates power and wealth in the hands of a few - while everyone else struggles - then what was all this progress for?

If we’re not careful, AI might create a world where power is concentrated and inequality gets worse. However, if we do it right, it can empower people and solve humanity's greatest challenges.

That’s why projects like this matter. They show that AI doesn’t have to be some mysterious force on the hands of big corporations. It can be a great tool in the hands of makers, students, and dreamers of a better world.
