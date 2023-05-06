
## Deep Learning Crossy Road

Artificial Neural Networks and Deep Learning, DIS Copenhagen Semester, Final Project.

![[https://crossyroad.fandom.com/wiki/Chicken](https://crossyroad.fandom.com/wiki/Chicken)](https://cdn-images-1.medium.com/max/2880/1*RJnL4uG4UnYLLh8ttWy3DQ.png)

You may or may not have heard of the reasonably popular game [*Crossy Road](https://en.wikipedia.org/wiki/Crossy_Road)*. Based on the common riddle joke “[Why did the chicken cross the road?](https://en.wikipedia.org/wiki/Why_did_the_chicken_cross_the_road%3F)”, *Crossy Road *is an arcade style video game involving controlling a chicken to cross as many roads and ponds and railroads as you can — or an “endless runner version of [*Frogger](https://en.wikipedia.org/wiki/Frogger)*,” if you are familiar with the more iconic arcade game *Frogger*.

Enough comparisons and textual descriptions. Here’s what it looks like to play *Crossy Road:*

 <iframe src="https://medium.com/media/c260b9c23a33f472d0262148af3ba205" frameborder=0></iframe>

Our goal is simply to build a program that plays this game using deep learning methods.

I’ve never heard nor played this game before doing this project (to keep it 100), but I’ve always been a fan of the “endless runner” genre growing up. *Crossy Road* has a different graphical perspective compared to endless runner classics such as *Temple Run* or *Jetpack Joyride*, and is much more demanding in terms of mechanics and quickness of reaction and spatial awareness (especially compared games like to *Subway Surfers* and *Temple Run*), but it does share the property of having simple and straightforward controls with these other endless runner games. This property inspired our approach.

There are only four possible moves in this game: forward (“up” key), backward (“down” key), left, and right — or five, if you count idling or “no-operation.” It is therefore accurate to say: when playing *Crossy Road*, for every moment, there exists a set of moves that will not lead to losing the game. Furthermore, one may observe that in this set of valid moves, some moves can be more advantageous than others: when there is no immediately approaching traffic in front of the chicken, for example, it is more advantageous to carry out the “forward” move, even though idling would also not lead to losing the game (immediately, at least, since you’re not allowed to idle for too long: an eagle would come and snatch you). When facing multiple valid moves (i.e., moves that won’t lead to losing the game), a (skilled) human player would more likely choose the most advantageous move; and in cases where multiple moves are equally advantageous, a human player, theoretically, should be equally likely to choose any of the equally advantageous moves.

So, given the reasonably small domain of possible moves, I thought that the game *Crossy Road* might be reducible to an image classification problem.

## Model

(The final version of our model can be found [here](https://github.com/yilongsong/deep_learning_crossy_road).)

So we want a model suitable for image classification. For this purpose, we make the basic choice of a convolutional neural network. The architecture is equally basic. For those interested, here’s the snippet of [code](https://github.com/yilongsong/deep_learning_crossy_road/blob/main/model.py) defining the architecture.

    # From model.py
    
    import torch.nn as nn
    
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )
            
            self.fc_layers = nn.Sequential(
                nn.Linear(128*5*14, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(512, 2),
            )
        
        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            x = self.fc_layers(x)
            return x

As visible in the code above, the hidden convolutional and max pooling layers, and the fully connected layer that follows are all quite typical. I have not yet experimented with fancier architectures, because, as it becomes immediately obvious in the very next steps and throughout the process of development, there are other much trickier, more pressing parts to this project.

A requirement that warrants consideration, on the other hand, is speed. Within every second multiple forward passes will have to be carried out through the neural network for the program to be able to play the game. Furthermore, given how data collection and game-playing are specifically carried out (which will be described in the next section), the model is expected to performed forward passes in almost negligible time when running on my laptop. So the model needs to be quick; it has to be light-weight — and from experimentation we conclude that it is with this architecture.

## Gameplay

Given the premise of reducing *Crossy Road *to an image classification problem, you may already have a rough idea of how the model is going to be used to play the game. Here’s a pseudo code version of [play.py](https://github.com/yilongsong/deep_learning_crossy_road/blob/main/play.py), the file responsible for this task.

    # Pseudo code version of play.py
    
    model = load('pretrained_convnet_model.pth')
    n = approx_number_screenshot_per_second
    
    while True:
      screenshot = take_screenshot()
      prediction = model.forward(screenshot)
      if prediction == 'up':
         move_up()
      elif prediction == 'down':
         move_down()
      elif prediction == 'left':
         move_left()
      elif prediction == 'right':
         move_right()
      elif prediction == 'noop':
         noop()
      
      sleep(1/n)

Viewing [play.py](https://github.com/yilongsong/deep_learning_crossy_road/blob/main/play.py) now, you will find that in the final version, the options “down,” “left,” and “right” are removed. The only options are “up” and “noop” (no-operation). This modification will be discussed in the section **Optimization: Movement Restriction**.

To summarize the pseudo code above in words, we take a screenshot, make a prediction with the pretrained ConvNet model, make a move accordingly, then pauses for a set amount of time before repeating this process again — intuitive enough.

## **Data Collection**

We’ve been ignoring the question crucial to any deep learning project: where does the data come from?

The answer, perhaps embarrassingly, is we collect the data ourselves manually. Here’s a pseudo code version of [collect_data.py](https://github.com/yilongsong/deep_learning_crossy_road/blob/main/collect_data.py) to illustrate the process.

    # Pseudo code version of collect_data.py
    
    n = approx_number_screenshot_per_second
    
    def on_press:
      if key == 'up':
         save_screenshot(screenshot(), 'data/up')
      elif key == 'down':
         save_screenshot(screenshot(), 'data/down')
      elif key == 'left':
         save_screenshot(screenshot(), 'data/left')
      elif key == 'right':
         save_screenshot(screenshot(), 'data/right')
    
    key_listener_thread = Thread() # Create new thread
    key_listener_thread.start(on_press) # Start thread
    
    while True: # On the main thread thread
      save_screenshot(screenshot(), 'data/noop')
      sleep(1/n)

Again, the final version of [collect_data.py](https://github.com/yilongsong/deep_learning_crossy_road/blob/main/collect_data.py) looks different. This will be discussed in the section **Optimization: Data Collection Strategy**.

To use this script, the human data collector simply runs the script and starts playing. Every time the human makes a move, a screenshot is taken and saved to the corresponding folder. This happens on one thread. On the main thread, screenshots with the no-operation label are taken and saved to the “noop” folder at a constant rate.

This data collection scheme works quite well if this model is applied to a game in which key presses are less frequent: while no-op screenshots may still be taken at moments very near to when key-press screenshots are taken (these no-op screenshots are not helpful in training the model; they are too similar and therefore should be classified as whatever type of key-press screenshot taken a moment before or after it), such occurrences are substantially rarer due to the low frequency at which keys are pressed. But, as we will soon find after our first round of data collection and training, this data collection scheme requires improvement.

To tie an untied knot from before, observe that during data collection, the classification of each screenshot is essentially “the correct decision based on moments before the screenshot,” while during the program’s gameplay, the decision of the model, at the point in time which it is carried out, is, ideally, “the correct (this is the idealistic part) decision based on the screenshot taken moments before.” Even assuming that both decisions are correct, there’s a gap in time that is the sum of the length of the “moments” in the descriptions above. This is why the process of taking a screenshot, processing it to the right datatype, and running it through the model needs to be quick enough to be considered negligible, as mentioned before.

After all of the above is implemented, data (around 15000 images in total) is collected, we train the model using Adam stochastic optimization, with a learning rate of 0.001 for 40 epochs (more epochs are experimented with but result in overfitting), we obtained an underwhelming test accuracy of 40%. Below is a video demonstrating the performance of the model at this point in game play.

 <iframe src="https://medium.com/media/4ba898f62c0d43380d0eab1211b507d3" frameborder=0></iframe>

It is noticeable that at this point, even without implementing movement restriction, the model is much more likely to output no-op and up just by the game’s nature. It is the case, first and foremost, that all players, our data collectors included, press the “up” key much more frequently than the other three, resulting in a significant imbalance in the amount of data in each category (~40% no-op, ~40% up, ~18% left and right, ~2% down). It is also true that depending on whether or not you encounter an immovable object (which depends on luck/probability and will force you to move left or right), you can get pretty far in the game just by pausing and going forward.

We see that the model does show some level of “intelligence” or “good judgement” at this point, but not much. So we thought about our approach conceptually and came up with optimizations.

## Optimization: Avatar Tracking

The first optimization I implemented was avatar tracking. All data used before, for training and during runtime alike, are screenshots of the entire game screen (downsampled, of course, for efficiency of data transfer, training, and real-time prediction). This is not ideal. The set of valid movement depends on only a small portion of the game screen, specifically, the portion immediately in front of the avatar. All the rest is a distraction in training, particularly when the position of the avatar is unclear.

One approach to ameliorate this situation is to simply limit the portion of the game screen being looked at. This approach will be effective in filtering out some distraction and noise, because the avatar is constrained in a sub-region of the game screen, but the effect will be limited, as the position of the avatar changes quite a lot both vertically and horizontally, and in the end there’s not that much we can crop out.

The other approach is to track the avatar. With the position of the avatar, more precise cropping can be done to each screenshot, showing both a precisely determined distance in front of the avatar and the current position of the avatar, both of which should be helpful for categorization. We implemented this approach of avatar tracking using template matching from the cv2 library. The specifics can be found in [smartcrop_data.py](https://github.com/yilongsong/deep_learning_crossy_road/blob/main/smartcrop_data.py).

Here is a few examples of avatar tracking and cropping from real data that we used in training our model.

![](https://cdn-images-1.medium.com/max/2000/1*BMFNUJXqlPeTeQgIhTDBSw.png)

![](https://cdn-images-1.medium.com/max/2000/1*oRjpZy4sEX51_j48Dc7woQ.png)

![Before cropping according to avatar position](https://cdn-images-1.medium.com/max/2000/1*uFljfXqL0DQfLEy9REGhBQ.png)

![](https://cdn-images-1.medium.com/max/2000/1*VCeP-3xV9kUQI26u_kd4GQ.png)

![](https://cdn-images-1.medium.com/max/2000/1*mIMGqKkDTqkvpg3HP9tKCQ.png)

![After cropping using avatar position](https://cdn-images-1.medium.com/max/2000/1*BI4S5Fbu5lEb-44fbGVdmw.png)

A directly observable issue with this approach is that avatar tracking doesn’t have an 100% accuracy, as shown in the examples below.

![](https://cdn-images-1.medium.com/max/2000/1*8vXuR6HUXdhA1tl-XJH2Mw.png)

![Example or error possibly due to atypical avatar model/surrounding pixel color](https://cdn-images-1.medium.com/max/2000/1*gTXyXeZ8mzcja-xQw8zvOA.png)

![](https://cdn-images-1.medium.com/max/2000/1*CdlU0pLqDp2TnoHg1J_yGw.png)

![Example of error due to a large portion of the avatar blocked by in-game objects](https://cdn-images-1.medium.com/max/2000/1*ipoAFyxr67ynqomt8sywqg.png)

Here’s how the model performs after implementing this optimization (all else being equal).

 <iframe src="https://medium.com/media/ed30d8d3ccd6834fbc001174e7140654" frameborder=0></iframe>

The performance is underwhelming, which may be hinted by the 35% testing accuracy of the model after training. While this video shows a worse performance, testing with more instances of gameplay shows that the level of “judgment” demonstrated with tracking implemented is in fact on par with the program without tracking.

## Optimization: Movement Restriction

The implement of tracking, as shown in the video above, somehow makes the model more prone to make “left,” “right,” and “down” predictions. This inspires the hypothesis that restricting the possible movements to only “up” and “no-op” should simplify the classification problem and make the model perform better. This optimization is relatively simple, so here’s an example of the result directly.

 <iframe src="https://medium.com/media/fe83ab6a36dcbc39c5e060fb73c18664" frameborder=0></iframe>

This optimization, despite its simplicity, advances the performance of the model considerably. The testing accuracy also increases to around 70%.

## Optimization: Data Collection Strategy

Our final attempt at optimizing involve changing the way we collect data. Instead of capturing “no-op” screenshots at a certain frequency, the problem of which has been discussed earlier, the data collector manually collects “no-op” screenshot by pressing the left shift button when deeming the current situation as belonging to “no-op” while playing the game. This manual mechanism conceptually ensures that the new dataset is of higher quality, and it is: using the same ConvNet architecture and same training hyper-parameters, repeated over 5 iterations (I changed the way I press shift, and/or the speed at which I move forward each time, because I realized that there’s a buffer effect to the forward movements (if I press forward quickly consecutively, each movement takes time to perform so the actual movement comes later, but the screenshots are taken earlier when the keys were pressed) which obstructs reliable prediction of the model) we obtain an average testing accuracy of 90%. Here’s an example of the model’s performance during gameplay.

 <iframe src="https://medium.com/media/d7757515377a6764ac3b2043038cb128" frameborder=0></iframe>

After more instances of testing, we found that the performance of our model has become more reliable, compared to the previous iteration.

## Conclusion

Through hours upon hours of building this program, implementing new features, and playing the game for data collection purposes (which unsurprisingly makes something that’s supposed to be fun feel like hard, repetitive labor), I slowly realized that my initial idea of reducing *Crossy Road* into an image recognition problem is greatly flawed. The game *Crossy Road *is substantially more complex compared to other games in the endless runner genre, particularly in the abundance of different moving objects and the game’s exceptional requirement for spatial awareness, the latter of which is something that cannot be fully captured by individual screenshots. Pictures, especially screenshots of a video game, cannot convey information on relative speed whatsoever, but an idea of the relative speed of all the moving parts is extremely important for decision making in *Crossy Road*. We can therefore conclude that to build a model that plays *Crossy Road* on the same level as a decent human player or at a superhuman level, a much more complex approach needs to be taken.

Ruminating on the basis of this perspective, however, we can in fact say that our model performs well — perhaps even better than expected. After all, how well can a human play *Crossy Road* if made to decide whether or not to step forward based on nothing but individual screenshots? Without hesitation our model made such decisions, and courageously stepped forward, into high-speed, murderously traffic time and time again. It fails sometimes — most of the times, in fact, but we still admire it for its determination, and gleefully cheer for every one of its rare instances of prudence.

(Written by Yilong Song; Project by Abdul Rauf Abdul Karim, Juliet Mankin, Yilong Song)
