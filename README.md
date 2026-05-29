# GUIDE_ROS_Planner

GUIDE is a diffusion-based autonomous exploration framework using global graph inference for efficient long-horizon exploration in large-scale environments.

The planner constructs a sparse global connectivity graph from partial observations and predicts informative unexplored structures through region-aware graph inference. A diffusion policy then generates long-horizon exploration actions with reduced redundant backtracking and improved exploration efficiency.



## Installation

Tested on:

- Ubuntu 18.04 + ROS Melodic
- Ubuntu 20.04 + ROS Noetic
Install dependencies:
```
sudo apt-get install ros-noetic-octomap
```
We recommend to use [conda](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html#) for package management. 
Our planner is coded in Python and based on [Pytorch](https://pytorch.org/get-started/locally/).
Other than Pytorch, please install following packages by:
```
pip install scikit-image matplotlib
```
We tested our planner in various version of these packages so you can just install the latest one.
Then you can download this repo and compile it.
```
git clone https://github.com/chezj7/GUIDE_ROS_Planner.git
cd GUIDE_ROS_Planner
catkin_make
```
**Note:** We only use CPU to do the network inference, so you do not need a GPU.

### 2. Development environments
In practice, our planner needs to cooperate with a Lidar SLAM module which outputs sensor odometry and Lidar scan, and a waypoint follower module which navigates the robot to the planned waypoint.
Fortunately, you can test our planner easily in the development environments provided by [CMU Robotics Institute](https://www.cmu-exploration.com/development-environment).

Please follow instructions for [CMU Development Environment](https://www.cmu-exploration.com/development-environment) to set up the Gazebo simulation, SLAM module, and waypoint follower module.

### Checkpoints
Please place the trained checkpoint under:

```bash
src/scripts/model/

### 3. Run the code
To run the development environment, go to the development environment folder in a terminal and run:
```
source devel/setup.bash 
roslaunch vehicle_simulator system_indoor.launch
```
Our planner can work in two indoor environments which is provided by [FAEL](https://github.com/SYSU-RoboticsLab/FAEL)
The CMU Development Environment also provides three of their environments: indoor, forest, and tunnel.

To run GUIDE planner, go to the planner folder in another terminal (launch your conda environment if any) and run:
```
source devel/setup.bash 
roslaunch rl_planner rl_planner.launch
```
if running the planner in the forest environment, run:
```
roslaunch rl_planner rl_planner_forest.launch
```
if running the planner in the tunnel environment, run:
```
roslaunch rl_planner rl_planner_tunnel.launch
```




## Credit
[Development environment](https://www.cmu-exploration.com/development-environment) is from CMU.
[Test Environment](https://github.com/SYSU-RoboticsLab/FAEL) is from FAEL
[RL-Based Planner](https://github.com/marmotlab/ARiADNE-ROS-Planner) serves as the foundational framework of this project, upon which the diffusion-based exploration planner is further developed and integrated.
[Octomap](https://octomap.github.io/) is from University of Freiburg.
[Quad tree](https://github.com/toastdriven/quads) is from [Daniel Lindsley](https://github.com/toastdriven).


