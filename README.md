# DS4420-Final-Project
Developing a deep reinforcement learning algorithm which utilizes equivariant graph neural networks to learn the optimal way to navigate a robot through a dynamic environment.

## Setup
To set up this project, make sure you have all requirements in requirements.txt. I don't recommend installing requirements.txt directly as pytorch geometric requires specific installs based on the system.

## Running
To train the project, run `main.py`. It has command line arguments, so use `--help` to assist you. Example `python main.py TrainingWorlds/BasicWorld.py configs/gcnn.json results/gcnn-basic`

To evaluate performance, run `evaluate.py`. It also has command line arguments.Example `python evaluate.py results/gcnn-basic TestingWorlds`

The `visualize_results.py` and `visualize_world.py` were used to generate visualizations found in the paper, but `visualize_robot.py` showcases the actual model running on a real world. Examples `python visualize_robot.py results/gcnn-basic TestingWorlds/SizeWorld50.py --save gcnn-basic.gif` and `python visualize_world.py TrainingWorlds/CircleWorld.py`