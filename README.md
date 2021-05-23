# object_and_lane_detection
I use darknet algorithm for object detection and LaneAF for lane detection.

Here is a picture
![result](./result.png)


After cloning this repo use 

```
python main.py --weights_laneaf LaneAF\tusimple-weights\net_0012.pth --input_img LaneAF\data\tusimple_test_image\3.jpg
```

Testing in real time

![test gif](ezgif-2-41845940fc26.gif)

Things to do

- [ ] Get better FPS on GTX 1060
- [ ] Automatically update dataset to train the current model for lane detection
- [ ] Use MLOps concepts to make code more sophisticated and applicable
- [ ] Develop flow chart to understand how data flows through the model