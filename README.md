# Hunter and Criminal Agents

Autonomous hunter agents that pursue and capture evasive criminal agents in a simulated grid based environment. The system simulates a battlefield filled with intelligent hunter bots, intelligent criminals, and obstacles, requiring the hunters to navigate, predict, and communicate the criminal targets intelligently under uncertainty.

The question explored: how does neural network sound classification (CNN), predictive tracking (Kalman filter, A* and vision), and cooperative communication contribute individually and collectively to the efficiency of autonomous hunter agents in pursuing evasive criminal targets?

## The agents

Hunters have vision that replicates human vision, where it is stronger and more up close. They share sightings and request backup through a communication system, classify sounds like walking, running and obstacle breaking with a CNN, predict a criminal's future position with a Kalman filter, navigate with A*, and remember what they have seen.

Criminals switch between walking, running and hiding, flee using evasive techniques when detected, generate sound as they move, and communicate to warn each other of hunter sightings.

## How the intelligence works

Raw audio for each sound category was pre-processed into mel spectrograms using librosa and saved as 64x64 RGB images, augmented with time stretching, pitch shifting and volume scaling. `TinyCNN` is built in PyTorch with two convolutional layers using ReLU activations and max pooling, two fully connected layers, and a final output layer with four neurons. Misclassified sounds are stored in a CSV and retrained on the next run, so the CNN improves every run.

Each bot creates a Kalman filter instance when it hears and classifies a sound. The filter has a 4-dimensional state vector `[x, y, dx, dy]` representing the criminal's estimated position and velocity, predicts using a constant velocity motion model, then corrects when new observations arrive. The higher the uncertainty in the prediction, the more weight is given to the new observation. The filter deactivates if no further sound updates are received, to prevent hunters from chasing endlessly.

Bots then use a tailored A* to move towards the criminal, using Manhattan distance for the heuristic as it's the most suitable heuristic for grid movement.

## Results

Each experiment was run ten times to reduce variability.

| Setup | CNN | Kalman | Comms | Avg time (s) | Std dev |
| --- | --- | --- | --- | --- | --- |
| 1 | No | No | No | 97.43 | 15.92 |
| 2 | Yes | No | No | 85.68 | 13.08 |
| 3 | Yes | No | Yes | 60.65 | 9.21 |
| 4 | Yes | Yes | No | 56.66 | 7.64 |
| 5 | Yes | Yes | Yes | 42.14 | 7.38 |

Each successive setup not only lowers the median capture time but also reduces the spread, so the more intelligent features the hunters can utilise, the faster and more consistent they are.

A second set of experiments varied the number of hunters and criminals from one to three each. Capture time increases with more criminals and fewer hunters, and the jump is sharpest with a single hunter: two criminals averaged just under 40 seconds, three criminals averaged over 90.

## Built with

Python, Tkinter for the battlefield GUI, PyTorch for the CNN, librosa for spectrograms, NumPy for the Kalman filter matrix operations, pandas and matplotlib for the results.
