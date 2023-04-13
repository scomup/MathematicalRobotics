# MathematicalRobotics

## What is this?
The development of robotics is always closely related to mathematics. But sometime pure mathematical expressions are boring and difficult to understand, so I hope to show the magic of mathematics through some interesting robotics demonstrations.


## The goals of our project.
We want to select some widely used and practical algorithms. for each algorithm, we aim to 
* Provide a readable a python implementation.
* Show a detailed mathematical proof.
* To show the math behind it, minimal use of third-party libraries. 

## Requirements 

```bash
pip3 install -r requirements.txt
```

## Demo Lists

### guass_newton_method
#### gauss newton for 2d points matching.
![demo2d](./imgs/demo2d.gif)
#### gauss newton for 3d points matching.
![demo3d](./imgs/demo3d.gif)
#### gauss newton for linear regression.
![demo_line](./imgs/demo_line.png)

### graph_optimization
#### 2d loop closing
![demo_pose2d_graph](./imgs/demo_pose2d_graph.gif)
#### 3d loop closing
![demo_pose3d_graph](./imgs/demo_pose3d_graph.gif)
#### polygon
Generate a trajectory that avoids polygons(obstacles) as much as possible.  
![demo_polygon](./imgs/demo_polygon.gif)

