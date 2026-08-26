import numpy as np

class KalmanTracker:
    def __init__(self, init_x, init_y): #to create an tracker object a decently accurate x,y of the noise is passed in 
        self.state = np.array([[init_x], [init_y], [0], [0]])  # 4x1 matrix - init the tracker set x,y and vel to 0 for now so its not moving

        # Uncertainty matrix (4x4 because of velocity now)
        self.uncertainty = np.eye(4) * 500 #4x4 matrix represnting uncertainty as of now uncertain but when more observations come this will shrink

        self.state_transition = np.array([ # this matrix tells the kalman filter how to predict the next state of the obj based of curr state
            [1, 0, 1, 0],  # x' = x + dx
            [0, 1, 0, 1],  # y' = y + dy
            [0, 0, 1, 0],  # dx stays same
            [0, 0, 0, 1]   # dy stays same
        ])

        # Process noise (movement randomness)
        self.process_noise = np.eye(4) * 0.5 # process noise and add uncertainty to each state var

        # Measurement matrix: we only observe [x, y]
        self.measurement_matrix = np.array([ #retrieves only x,y no vel
            [1, 0, 0, 0], #extract x
            [0, 1, 0, 0] #extract y
        ])

        #print(self.process_noise)

        self.sound_noise = np.eye(2) * 25 # we use this matrix to help how much the sound should be trusted

        self.last_source = None

    def predict(self):

        self.state = self.state_transition @ self.state # predict the new state
        self.uncertainty = ( #how confident/uncertain with the new state guess 
            self.state_transition @ self.uncertainty @ self.state_transition.T #matrix manipulation 
            + self.process_noise
        )


    def update(self, observation, source="sound"):

  
        measurement_noise = self.sound_noise


        error = observation - self.measurement_matrix @ self.state # diff between pred and measurement 


        comb_uncert = ( # how uncertain we are about pred and actual
            self.measurement_matrix @ self.uncertainty @ self.measurement_matrix.T
            + measurement_noise
        )

        invert = np.linalg.inv(comb_uncert)
        Kalman_gain= self.uncertainty @ self.measurement_matrix.T @ invert  #kalman gain how much much we trust the measurement vs. our prediction


        self.state = self.state + Kalman_gain @ error

        Identity_matr = np.eye(4) #identity matrx 
        self.uncertainty = (Identity_matr - Kalman_gain @ self.measurement_matrix) @ self.uncertainty

        self.last_source = source


    def get_position(self):
        return int(self.state[0][0]), int(self.state[1][0]) # to retrieve x,y out of the4x1 matrix

    def reset(self): # wipe the mem tracker
        self.state = np.array([[0.0], [0.0], [0.0], [0.0]]) #rese x,y and vel
        self.uncertainty = np.eye(4) * 1000 #increase uncert by twice 
        self.last_source = None #reset source to None
