# Kalman Filter
### COSC 69, Winter 2020
### Rylee Stone

## Design of kalman_fil.py:
This file is split into 3 classes: one (`Kalman`) to implement the Kalman Filter, one (`Run`) to subscribe to _cmd_vel_ and _scan_ messages and publish a _PoseWithCovarianceStamped_ message for each estimate, and one (`SubAndPlot`) that subscribes to _cmd_vel_, _pose_ and _scan_ to make more Kalman estimates and to plot the information. 

## How to Run the Program:
This program is currently set up to simply be run (i.e. `python kalman_fil.py`) during the execution of a bag file (or during the publishing of messages). No additional commands or input needed, but messages must be being published when the program is run. 
