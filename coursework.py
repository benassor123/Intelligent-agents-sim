import tkinter as tk
import random
import numpy as np
import heapq
from astar_bayesian_search import aStarSearch

from sound_processing import modify_sound_with_distance
from predict_sound import predict_sound
import time
from kalman_filter import KalmanTracker
import os
import shutil
from train_cnn import retrain_on_misclassified
MISCLASSIFIED_DIR = "misclassified_data/"

CONFIG = {
    "enable_bot_comms": True,
    "enable_criminal_comms": True,
    "enable_kalman": True ,
    "enable_sound_detection" : True,
    "random_movement_only": False
    
}


class Bot_Communication_System: #like imagine the bots have headphones on or are on a call:)
    def __init__(self):
        self.sightings_ = {}        # {criminal_id: {"x": x, "y": y, "by": bot_id, "time": timestamp}}
        self.sound_heards = {}      # {bot_id: [{"type": "running", "x": x, "y": y, "time": t}, ...]}
        self.help_requests = {}     # {bot_id: {"criminal_id": cid, "x": x, "y": y, "time": t}}
        self.targets_found = {}     # {criminal_id: {"claimed_by": bot_id, "time": t}}
        self.investigations = {}  
        


    def report_criminal_sighting(self, bot_id, criminal, x, y):
            self.sightings_[id(criminal)] = {
                "x": x,
                "y": y,
                "by": bot_id,
                "time": time.time()
            }


    def get_recent_sightings(self, max_age=10):
        now = time.time()
        fresh_sightings = {}

        for criminal_id, info in self.sightings_.items():
            time_seen = info["time"]
            if now - time_seen <= max_age:
                fresh_sightings[criminal_id] = info

        return fresh_sightings



    def assign_bots_to_sound(self, bots, sound_type, source_x, source_y, sound_id):
        eligible_bots = []
        for bot in bots:
            if not bot.busy:
                dist = abs(bot.grid_x - source_x) + abs(bot.grid_y - source_y)
                eligible_bots.append((bot, dist))

        for i in range(len(eligible_bots)):
            for j in range(i + 1, len(eligible_bots)):
                if eligible_bots[j][1] < eligible_bots[i][1]:
                    eligible_bots[i], eligible_bots[j] = eligible_bots[j], eligible_bots[i]

        if not eligible_bots:
            return #bots are busy

        first_bot = eligible_bots[0][0]
        first_bot.hear_sound(sound_type, eligible_bots[0][1], source_x, source_y, skip_cnn=False, sound_id=sound_id)

        if len(eligible_bots) > 1:
           
            second_bot = eligible_bots[1][0]
            offset_x, offset_y = source_x + 2, source_y
            offset_x = min(max(0, offset_x), second_bot.battlefield.num_cols - 1)
            offset_y = min(max(0, offset_y), second_bot.battlefield.num_rows - 1)
            dist = abs(second_bot.grid_x - offset_x) + abs(second_bot.grid_y - offset_y)

            second_bot.hear_sound(sound_type, dist, offset_x, offset_y, skip_cnn=False, sound_id=sound_id)




    def request_backup_and_assign_backup(self, bots, requester_id, criminal, x, y):
        idle_bots = []

  
        for bot in bots:
            if bot.id != requester_id and not bot.busy:
                dist = abs(bot.grid_x - x) + abs(bot.grid_y - y)
                idle_bots.append((bot, dist))

        #sort using the sorting alg bubble sort
        n = len(idle_bots)
        for i in range(n):
            for j in range(0, n - i - 1):
                if idle_bots[j][1] > idle_bots[j + 1][1]:
          
                    idle_bots[j], idle_bots[j + 1] = idle_bots[j + 1], idle_bots[j]

        if idle_bots:
            backup_bot = idle_bots[0][0]
            print(" Bot " + str(requester_id) + " is calling for backup.  Assigning Bot " + str(backup_bot.id) + " to help at location -  (" + str(x) + ", " + str(y) + ")")

            backup_bot.kalman_tracker = KalmanTracker(x, y)
            backup_bot.kalman_origin = (x, y)
            backup_bot.last_kalman_update_time = time.time()

            observation = np.array([[x], [y]])
            backup_bot.kalman_tracker.update(observation, source="vision")

            backup_bot.investigation_target = (x, y)
            backup_bot.set_path_to_sound_source()
            backup_bot.busy = True





class Criminal_Communication_System:
    def __init__(self):

        self.alerts = []

    def report_bot_sighting(self, criminal_id, x, y):
        self.alerts.append({
            "created_by": criminal_id,
            "x": x,
            "y": y,
            "time": time.time()
        })
        print("Criminal " + str(criminal_id) + " has spotted a bot at (" + str(x) + ", " + str(y) + ")")


    def get_recent_alerts(self, max_age=5):
        now = time.time()
        return [a for a in self.alerts if now - a["time"] <= max_age]



class Battlefield:
    def __init__(self, width, height, cell_size, num_bots, num_criminals, num_obstacles, show_grid=True):
        self.communication_system = Bot_Communication_System() #like imagine the bots have headphones on - only one instance 
        self.criminal_comms = Criminal_Communication_System()

        self.width = width
        self.height = height
        self.cell_size = cell_size  
        self.num_cols = width // cell_size
        self.num_rows = height // cell_size
        self.prob_grid = np.zeros((self.num_rows, self.num_cols), dtype=float) 
        self.obstacle_grid = np.zeros((self.num_rows, self.num_cols), dtype=int) 
        self.criminal_grid = np.zeros((self.num_rows, self.num_cols), dtype=int)
        self.safe_zone_size = 4  
        self.num_bots = num_bots
        self.num_criminals = num_criminals
        self.num_obstacles = num_obstacles
        self.show_grid = show_grid 
        self.canvas = None
        self.bots = []
        self.criminals = []
        self.obstacles = []
        self.init_game_objects()
        self.start_time = None
        

      
  


    def check_bot_criminal_collisions(self):
        for bot in self.bots:
            for criminal in self.criminals[:]:
                if bot.grid_x == criminal.grid_x and bot.grid_y == criminal.grid_y:
                    print("Criminal has been caught by Bot " + str(bot.id) + " at location (" + str(criminal.grid_x) + ", " + str(criminal.grid_y) + ")")

                    self.criminals.remove(criminal)
                    criminal.alive = False
                    self.criminal_grid[criminal.grid_y, criminal.grid_x] = 0

                    for bot in self.bots:
                        if bot.tracking_criminal == criminal:
                            bot.tracking_criminal = None
                            bot.kalman_tracker = None
                            bot.path = []
                            bot.busy = False
                            bot.running = False

            

    def init_game_objects(self):
        id = 0
        for bot in range(self.num_bots):
            id = id+1
            bot_x = random.randint(1, self.safe_zone_size - 1)
            bot_y = random.randint(1, self.safe_zone_size - 1)
            self.bots.append(Bot(self, x=bot_x, y=bot_y,id=id, comms=self.communication_system))

        id = 0
        for crim in range(self.num_criminals):
            id= id+1
            criminal = Criminal(self, self.safe_zone_size,id,self.criminal_comms) 
            self.criminals.append(criminal)
            self.criminal_grid[criminal.grid_y, criminal.grid_x] = 2  



        for obstacle in range(self.num_obstacles):
            self.obstacles.append(Obstacle(self))
        

    def draw(self, canvas):
        self.canvas = canvas
        if self.canvas:
            self.canvas.delete("all")
            for obj in self.bots + self.criminals + self.obstacles:
                obj.draw(self.canvas)




    # def draw(self, canvas):
    #     self.canvas = canvas
    #     if self.canvas:
    #         self.canvas.delete("all")

    #         if self.show_grid:
    #             for row in range(self.num_rows):
    #                 for col in range(self.num_cols):
    #                     x1 = col * self.cell_size
    #                     y1 = row * self.cell_size
    #                     x2 = x1 + self.cell_size
    #                     y2 = y1 + self.cell_size

    #                     # Draw grid cell
    #                     canvas.create_rectangle(x1, y1, x2, y2, outline="dim gray")

    #                     # Draw (x, y) label in each cell
    #                     label = f"({col},{row})"
    #                     canvas.create_text(
    #                         x1 + self.cell_size // 2,
    #                         y1 + self.cell_size // 2,
    #                         text=label,
    #                         fill="dark gray",
    #                         font=("Arial", 7)
    #                     )

    #         for obj in self.bots + self.criminals + self.obstacles:
    #             obj.draw(self.canvas)

    def update_prob_density_map(self, searched_x, searched_y, found):
        p = self.prob_grid[searched_y, searched_x]  
        q = 0.8 

        if found:
            self.prob_grid[:, :] = 0  
            self.prob_grid[searched_y, searched_x] = 1.0  
        else:
            self.prob_grid[searched_y, searched_x] = (p * (1 - q)) / ((1 - p) + (p * (1 - q))) #bayesian probability equation 
            normalization_factor = 1 / (1 - p * q)
            self.prob_grid *= normalization_factor 
       
      


    def distribute_probability(self):  
        num_cells = self.num_rows * self.num_cols 
        if num_cells > 0: 
            self.prob_grid = np.ones((self.num_rows, self.num_cols), dtype=float) / num_cells 




    def update(self):
        if self.start_time is None:
            self.start_time = time.time()
            print("starting timer")
        self.canvas.delete("all")  

        for bot in self.bots:
            bot.move()

        for criminal in self.criminals:
            criminal.update_behavior()
            criminal.move()
        self.check_bot_criminal_collisions()


# 🔹 Collision Detection: Remove criminals caught by bots
        # for bot in self.bots:
        #     for criminal in self.criminals[:]:  # Copy the list to avoid issues while removing
        #         if bot.grid_x == criminal.grid_x and bot.grid_y == criminal.grid_y:
        #             print(f"🚔 Criminal caught at ({criminal.grid_x}, {criminal.grid_y})!")
        #             self.criminals.remove(criminal)
        #             self.criminal_grid[criminal.grid_y, criminal.grid_x] = 0

        #             # Reset tracking for all bots who might have been chasing this one
        #             for b in self.bots:
        #                 if b.tracking_criminal == criminal:
        #                     b.tracking_criminal = None
        #                     b.kalman_tracker = None
        #                     b.path = []

        # 🧹 OPTIONAL: If no criminals left, clean everything up
        if not self.criminals:
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            print("All criminals have been caught time to clear the  Battlefield ")
            print("Total time taken: " + str(elapsed_time) + "seconds")
            for b in self.bots:
                b.tracking_criminal = None
                b.kalman_tracker = None
                b.path = []


        self.draw(self.canvas)  # Redraw everything
        self.canvas.after(500, self.update)  # Update every 500ms





class Bot:
    def __init__(self, battlefield, x, y,id,comms):
        self.battlefield = battlefield  
        self.grid_x = x  
        self.grid_y = y  
        self.pixel_x = x * battlefield.cell_size  
        self.pixel_y = y * battlefield.cell_size  
        self.size = battlefield.cell_size // 2  
        self.moving = False  
        self.path = []  
        self.direction = "UP" 
        self.comms = comms
        self.id = id


        self.ahead_dx, self.ahead_dy = (0, -1)  
        self.left_dx, self.left_dy = (-1, 0)   
        self.right_dx, self.right_dy = (1, 0)   


        self.eye_x = self.pixel_x
        self.eye_y = self.pixel_y
        self.last_seen_direction = None

        self.kalman_tracker = None
        self.tracking_criminal = None  
        self.speed = 1
        self.running = False
        self.busy = False
        self.help_requested = False
        self.max_stamina = 100
        self.stamina = self.max_stamina
        self.stamina_cost = 2
        self.recovery_rate = 1
        self.recent_positions = []
        self.max_recent_positions = 5
        self.last_seen_criminal_pos = None
        self.last_seen_time = None



 
    



    def draw(self, canvas):


        fill_color = "red" if self.running else "blue"
        canvas.create_rectangle(
            self.pixel_x, self.pixel_y,
            self.pixel_x + self.size, self.pixel_y + self.size,
            fill=fill_color, outline="white"
        )




        canvas.create_text(
            self.pixel_x + self.size // 2, self.pixel_y + self.size // 2,  
            text="H", fill="white", font=("Arial", 10, "bold")
        )  



        canvas.create_oval(
            self.eye_x - 3, self.eye_y - 3,
            self.eye_x + 3, self.eye_y + 3,
            fill="black"
        )


        cell_size = self.battlefield.cell_size
        ahead_px = self.pixel_x + (self.ahead_dx * cell_size)
        ahead_py = self.pixel_y + (self.ahead_dy * cell_size)
        
        left_px = self.pixel_x + (self.left_dx * cell_size)
        left_py = self.pixel_y + (self.left_dy * cell_size)

        right_px = self.pixel_x + (self.right_dx * cell_size)
        right_py = self.pixel_y + (self.right_dy * cell_size)


        # canvas.create_oval(ahead_px - 5, ahead_py - 5, ahead_px + 5, ahead_py + 5, fill="blue")   # BLUE  AHEAD
        # canvas.create_oval(left_px - 5, left_py - 5, left_px + 5, left_py + 5, fill="green")     # GREEN LEFT
        # canvas.create_oval(right_px - 5, right_py - 5, right_px + 5, right_py + 5, fill="orange") # ORANGE RIGHT





    def standard_movement(self):
        for _ in range(30):  # Try up to 30 times
            x = random.randint(0, self.battlefield.num_cols - 1)
            y = random.randint(0, self.battlefield.num_rows - 1)

            # Avoid  the recently visited cells
          #  if (x, y) in self.recent_positions:
             #   continue
            if not self.tracking_criminal and (x, y) in self.recent_positions:
                continue

             

            if self.is_valid_target(x, y):
                path = aStarSearch(self.battlefield, (self.grid_x, self.grid_y), (x, y))
                if path:
                    self.path = path
                    self.recent_positions.append((x, y))
                    
                    # Trim history to max limit
                    if len(self.recent_positions) > self.max_recent_positions:
                        self.recent_positions.pop(0)

        
                    print("Bot " + str(self.id) + " is going to (" + str(x) + ", " + str(y) + ") ")

                    return

        print(" Bot " + str(self.id) + " couldn't find a good random path.")


    def move(self):

        if CONFIG.get("random_movement_only"):
            self.standard_movement()
            
        if CONFIG.get("enable_kalman") and self.kalman_tracker:
            # Predict next estimated criminal location
            self.kalman_tracker.predict()
            predicted_x, predicted_y = self.kalman_tracker.get_position()


            if not self.is_valid_target(predicted_x, predicted_y):
            
                print("Invalid predicted target (" + str(predicted_x) + ", " + str(predicted_y) + ").")


                self.kalman_tracker = None
                self.tracking_criminal = None
                self.path = []
                self.running = False
                self.busy = False
                return
            

         
            if (self.grid_x, self.grid_y) == (predicted_x, predicted_y):
                print("Already at predicted target (" + str(predicted_x) + ", " + str(predicted_y) + "). No movement  is needed as of now.")

                self.kalman_tracker = None
                self.tracking_criminal = None
                self.path = []
                self.running = False
                self.busy = False

                return


            # Use A* to get there
            self.path = aStarSearch(
                self.battlefield,
                (self.grid_x, self.grid_y),
                (predicted_x, predicted_y)
            )

            if self.path:
                print("Tracking the criminal to predicted position: (" + str(predicted_x) + ", " + str(predicted_y) + ")")

            else:
                print("Couldn't find a path to the predicted target. Reverting to random movement.")
                #self.lose_target()  # drop tracker if target becomes unreachable. Optional tbh

        elif not self.path:
            self.compute_new_path()

        if self.moving:
            return

        self.moving = True
        self.step()




    def compute_new_path(self):

        target_x, target_y = self.find_survivor()
        new_path = aStarSearch(self.battlefield, (self.grid_x, self.grid_y), (target_x, target_y))

        if new_path:
            print(" New A* Path found: " + str(new_path))

            self.path = new_path
        else:
            print("No valid path  is found Bot is recalculating.")
            self.path = [(self.grid_x + random.choice([-1, 1]), self.grid_y + random.choice([-1, 1]))]  




    




    def find_survivor(self): 
     

        highest_prob_grid_cell = np.max(self.battlefield.prob_grid)
        target_cells_arr = np.argwhere(self.battlefield.prob_grid == highest_prob_grid_cell)

        valid_cells = [
            (x, y) for y, x in target_cells_arr
            if 0 <= x < self.battlefield.num_cols and 0 <= y < self.battlefield.num_rows and self.battlefield.obstacle_grid[y, x] != 3
        ]

        if valid_cells:
            return random.choice(valid_cells)  


        return (
            random.randint(0, self.battlefield.num_cols - 1),
            random.randint(0, self.battlefield.num_rows - 1),
        )


    def step(self):
        if not self.moving:
            return

        

        if not self.kalman_tracker and not self.tracking_criminal and self.running:
            self.running = False
            self.speed = 1
        

        if self.kalman_tracker and self.last_kalman_update_time:
            time_since_last_update = time.time() - self.last_kalman_update_time

            if time_since_last_update > 10:
                predicted_x, predicted_y = self.kalman_tracker.get_position()
                dist = abs(predicted_x - self.grid_x) + abs(predicted_y - self.grid_y)
         
                print("Kalman tracker  has now expired due to lack of updates near sound.")
                self.busy= False
                self.kalman_tracker = None
                self.kalman_origin = None
                self.path = []
                self.running = False
                self.speed =1 



        if not self.path:
            self.compute_new_path()
            self.running = False
            self.speed = 1
            self.battlefield.canvas.after(500, self.step)
            return  

        self.scan_environment() 

 
        next_x, next_y = self.path[0]


        if next_x > self.grid_x:
            self.direction = "RIGHT"
            self.ahead_dx, self.ahead_dy = (1, 0)  # Ahead is right
            self.left_dx, self.left_dy = (0, -1)   # Left is up
            self.right_dx, self.right_dy = (0, 1)  # Right is down
        elif next_x < self.grid_x:
            self.direction = "LEFT"
            self.ahead_dx, self.ahead_dy = (-1, 0)  # Ahead is left
            self.left_dx, self.left_dy = (0, 1)     # Left is down
            self.right_dx, self.right_dy = (0, -1)  # Right is up
        elif next_y > self.grid_y:
            self.direction = "DOWN"
            self.ahead_dx, self.ahead_dy = (0, 1)   # Ahead is down
            self.left_dx, self.left_dy = (1, 0)     # Left is right
            self.right_dx, self.right_dy = (-1, 0)  # Right is left
        elif next_y < self.grid_y:
            self.direction = "UP"
            self.ahead_dx, self.ahead_dy = (0, -1)  # Ahead is up
            self.left_dx, self.left_dy = (-1, 0)    # Left is left
            self.right_dx, self.right_dy = (1, 0)   # Right is right




        if self.running:
            self.stamina -= self.stamina_cost
            if self.stamina <= 0:
                self.stamina = 0
                self.running = False
                self.speed = 1
                print("Bot " + str(self.id) + " is now  exhausted! Stamina gone.")

        else:
            if self.stamina < self.max_stamina:
                self.stamina += self.recovery_rate


        for index in range(self.speed):
            if not self.path:
                break

            next_x, next_y = self.path.pop(0)

            if abs(next_x - self.grid_x) + abs(next_y - self.grid_y) > 1:

                print("Invalid step detected while running.  Now Adjusting path.")
                self.path = []
                self.running = False
                self.speed = 1
                break
            self.grid_x, self.grid_y = next_x, next_y
            self.pixel_x = self.grid_x * self.battlefield.cell_size
            self.pixel_y = self.grid_y * self.battlefield.cell_size
        



        self.pixel_x = self.grid_x * self.battlefield.cell_size
        self.pixel_y = self.grid_y * self.battlefield.cell_size


        for criminal in self.battlefield.criminals[:]:  # iterate on a copy
            if self.grid_x == criminal.grid_x and self.grid_y == criminal.grid_y:

                print("Criminal caught at (" + str(criminal.grid_x) + ", " + str(criminal.grid_y) + ")!")

                self.busy = False
                self.battlefield.criminals.remove(criminal)
                criminal.alive = False
                self.battlefield.criminal_grid[criminal.grid_y, criminal.grid_x] = 0
                self.speed = 1
                self.running = False
                self.tracking_criminal = None
                self.kalman_tracker = None
                self.running = False
                self.speed = 1
                self.path = []



        cell_size = self.battlefield.cell_size
        self.eye_x = self.pixel_x + (self.ahead_dx * cell_size // 2)
        self.eye_y = self.pixel_y + (self.ahead_dy * cell_size // 2)

        if self.battlefield.canvas:
            self.battlefield.draw(self.battlefield.canvas)

        if self.battlefield.canvas:
            self.battlefield.canvas.after(500, self.step)


    def scan_environment(self):
        """
        Extends the bot's vision beyond the short-term range.
        Vision **decreases** over distance, and obstacles **block sight**.
        """
        vision_ranges = {
            1: 1.0,  # 100% accuracy for 1 tile
            4: 0.75, # 75% accuracy for 4 tiles
            8: 0.50, # 50% accuracy for 8 tiles
            12: 0.25 # 25% accuracy for 12+ tiles
        }

        directions = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0)
        }

        if self.direction not in directions:
            return

        dx, dy = directions[self.direction]

        def scan_direction(dx, dy, max_distance):

            for distance in range(1, max_distance + 1):
                check_x = self.grid_x + (dx * distance)
                check_y = self.grid_y + (dy * distance)


                if not (0 <= check_x < self.battlefield.num_cols and 0 <= check_y < self.battlefield.num_rows):
                    break

        
                accuracy = 1.0
                for dist_threshold, acc in vision_ranges.items():
                    if distance >= dist_threshold:
                        accuracy = acc
                        

                if self.battlefield.obstacle_grid[check_y, check_x] == 3:
                    print("Obstacle detected at (" + str(check_x) + ", " + str(check_y) + ") - Vision  now stopped")

                    break  

                elif self.battlefield.criminal_grid[check_y, check_x] == 2:


                        print("Criminal SEEN at (" + str(check_x) + ", " + str(check_y) + ") with " + str(int(accuracy * 100)) + "% accuracy!")

                        print("criminal sighting  has been reported to group!")

                        dir_map = {
                            "UP": (0, -1),
                            "DOWN": (0, 1),
                            "LEFT": (-1, 0),
                            "RIGHT": (1, 0)
                        }

                        for criminal in self.battlefield.criminals:
                            if not criminal.alive:
                                continue
                            if criminal.grid_x == check_x and criminal.grid_y == check_y:
                                seen_direction = criminal.direction
                                dx, dy = dir_map.get(seen_direction, (0, 0))

                                print("Direct pursuit. Bot is now heading to exact location of criminal at (" + str(check_x) + ", " + str(check_y) + ")")



                                if accuracy >= 0.65:
                                    self.last_seen_criminal_pos = (check_x, check_y)
                                    self.last_seen_time = time.time()
                                 #   self.comms.report_criminal_sighting(self.id, criminal, check_x, check_y)
                                    if CONFIG["enable_bot_comms"]:
                                        self.comms.request_backup_and_assign_backup(self.battlefield.bots, self.id, criminal, check_x, check_y)
                                 
                                    if self.tracking_criminal == criminal:
                                        self.running = True
                                        self.speed = 2
                                        self.battlefield.draw(self.battlefield.canvas)

                                        # Update path only if theposition changed
                                        if (check_x, check_y) != self.path[-1]:
                                            self.path = aStarSearch(
                                                self.battlefield,
                                                (self.grid_x, self.grid_y),
                                                (check_x, check_y)
                                            )
                                    else:
                                        # New criminal target
                                        self.tracking_criminal = criminal
                                        self.path = aStarSearch(
                                            self.battlefield,
                                            (self.grid_x, self.grid_y),
                                            (check_x, check_y)
                                        )
                                        if self.stamina > 10:
                                            self.running = True
                                            self.speed = 2
                                            ...
                                        else:
                                            self.running = False
                                            self.speed = 1

                                        self.battlefield.draw(self.battlefield.canvas)


                                    self.kalman_tracker = None  # Drop Kalman if vision is good


     
        scan_direction(dx, dy, 20)  
        
        # Scan left and right
        left_dx, left_dy = -dy, dx  # Rotate 90 degrees left
        right_dx, right_dy = dy, -dx  # Rotate 90 degrees right
        scan_direction(left_dx, left_dy, 20)
        scan_direction(right_dx, right_dy, 20)

        print("Long-Term Vision Scan Completed!")

        if self.last_seen_criminal_pos and self.last_seen_time:
            time_since_seen = time.time() - self.last_seen_time
            if time_since_seen > 2: 

                if CONFIG["enable_kalman"]:
                    x, y = self.last_seen_criminal_pos
                    self.kalman_tracker = KalmanTracker(x, y)
                    self.kalman_origin = (x, y)
                    self.last_kalman_update_time = time.time()

                    observation = np.array([[x], [y]])
                    self.kalman_tracker.update(observation, source="vision")

                    self.tracking_criminal = None
                    self.running = False
                    self.speed = 1

                self.last_seen_criminal_pos = None
                self.last_seen_time = None
            


    def hear_sound(self, sound_type, distance, guessed_x, guessed_y, skip_cnn=False,sound_id = None):
        if not CONFIG.get("enable_sound"):
            return 
        if not skip_cnn:
            distorted_path = modify_sound_with_distance(sound_type, distance)
            predicted_label = predict_sound(distorted_path)

            if predicted_label != sound_type:
    
                print("Mismatch data. Expected " + str(sound_type) + ", but heard " + str(predicted_label))

                wrong_dir = os.path.join(MISCLASSIFIED_DIR, sound_type)
                os.makedirs(wrong_dir, exist_ok=True)
                new_filename = f"{sound_type}_mis_{int(time.time())}.png"
                shutil.copy(distorted_path, os.path.join(wrong_dir, new_filename))
                return

        print("sound id is " + str(sound_id) )
        print("Bot has been heard " + str(sound_type).upper() + " from approx (" + str(guessed_x) + ", " + str(guessed_y) + ") (true distance " + str(distance) + ")")


        self.busy = True
        self.investigation_target = (guessed_x, guessed_y)
        if CONFIG.get("enable_kalman"):

            if self.kalman_tracker is None:
                self.kalman_tracker = KalmanTracker(guessed_x, guessed_y)
                self.kalman_origin = (guessed_x, guessed_y)
            else:
                ox, oy = self.kalman_origin
                if abs(guessed_x - ox) + abs(guessed_y - oy) > 6:

                    print("Ignored distant sound not near current Kalman origin: (" + str(guessed_x) + "," + str(guessed_y) + ")")

                    return

            self.last_kalman_update_time = time.time()

            observation = np.array([[guessed_x], [guessed_y]])
            self.kalman_tracker.update(observation, source="sound")
        self.set_path_to_sound_source()




    def is_valid_target(self, x, y):
        return (
            0 <= x < self.battlefield.num_cols and
            0 <= y < self.battlefield.num_rows and
            self.battlefield.obstacle_grid[y, x] != 3
        )

    def set_path_to_sound_source(self):
        if not hasattr(self, "investigation_target") or self.investigation_target is None:
            print("No investigation target set.")
            return

        target_x, target_y = self.investigation_target
        if not self.is_valid_target(target_x, target_y):
            print("Invalid sound target: (" + str(target_x) + ", " + str(target_y) + ") — IGNORING.")

            return

        # Run A* pathfinding to the guessed sound location
        new_path = aStarSearch(
            self.battlefield,
            (self.grid_x, self.grid_y),
            (target_x, target_y)
        )

        if new_path:
            print("Investigating sound source at " + str(self.investigation_target))
            self.path = new_path
        else:

            print("Couldn't find a path to sound source at " + str(self.investigation_target))






        
import random

class Criminal:
    def __init__(self, battlefield, safe_zone_size,id,comms):
        self.battlefield = battlefield
        self.path = []
        self.state = "hiding"  
        self.speed = 1  
        self.hiding = True
        self.state_timer = random.randint(3, 5) 
        self.last_sound_time = 0  
        self.last_emitted = 0  
        self.direction = "UP"
        self.ahead_dx, self.ahead_dy = (0, -1)
        self.left_dx, self.left_dy = (-1, 0)
        self.right_dx, self.right_dy = (1, 0)
        self.last_emitted_sound = None  
        self.alive = True
        self.id = id
        self.moving = False
        self.comms = comms

        while True:
            grid_x = random.randint(safe_zone_size, battlefield.num_cols - 1)
            grid_y = random.randint(safe_zone_size, battlefield.num_rows - 1)
            
            if battlefield.obstacle_grid[grid_y, grid_x] == 0 and battlefield.criminal_grid[grid_y, grid_x] == 0:
                self.grid_x = grid_x
                self.grid_y = grid_y
                self.pixel_x = self.grid_x * battlefield.cell_size  
                self.pixel_y = self.grid_y * battlefield.cell_size  
                self.size = battlefield.cell_size // 2  
                battlefield.criminal_grid[self.grid_y, self.grid_x] = 2 
                break  


    def update_behavior(self):
   
        self.state_timer -= 1  # Reduce the timer


        if CONFIG.get("enable_criminal_comms") and self.comms:
            alerts = self.comms.get_recent_alerts(max_age=6)

            for alert in alerts:
                ax, ay = alert["x"], alert["y"]
                dist = abs(ax - self.grid_x) + abs(ay - self.grid_y)

   
                if dist <= 6:
              
                    print("Criminal " + str(self.id) + " reacting to recent alert at (" + str(ax) + ", " + str(ay) + ")")


                    if self.state != "running":
                        self.state = "running"
                        self.speed = 2
                        self.hiding = False
                        self.state_timer = random.randint(4, 8)
                        self.path = self.plan_move_away_from(ax, ay)
                    return

        if self.state_timer > 0:
            return  

        # Pick a  random state
        new_state = random.choice(["hiding", "walking", "running"])
        self.state = new_state


        if new_state == "hiding":
            self.speed = 0
            self.hiding = True
            self.find_hiding_spot()
        elif new_state == "walking":
            self.speed = 1
            self.hiding = False
        else:  # Running
            self.speed = 2
            self.hiding = False

       
        self.state_timer = random.randint(3, 10)  # Stay in this state for 3-10 updates




    def plan_move_away_from(self, danger_x, danger_y):

        dx = self.grid_x - danger_x
        dy = self.grid_y - danger_y

        dx = int(np.sign(dx))
        dy = int(np.sign(dy))

        candidates = [
            (self.grid_x + dx * 5, self.grid_y + dy * 5),
            (self.grid_x + dx * 4, self.grid_y + dy * 4),
            (self.grid_x + dy * 4, self.grid_y - dx * 4),
            self.find_safe_spot()
        ]

        for tx, ty in candidates:
            if 0 <= tx < self.battlefield.num_cols and 0 <= ty < self.battlefield.num_rows:
                if self.battlefield.obstacle_grid[ty, tx] != 3:
                    path = aStarSearch(self.battlefield, (self.grid_x, self.grid_y), (tx, ty))
                    if path:

                        print("Criminal " + str(self.id) + " avoiding  the alert area,now is  moving to (" + str(tx) + ", " + str(ty) + ")")

                        return path

        return []




    def move(self):
        if not self.alive or self.state == "hiding":
            return

        if not self.path:
            self.compute_new_path()

        if self.moving:
            return

        self.moving = True
        self.step()




    def step(self):
        if not self.alive or self.state == "hiding":
            self.moving = False
            return

        steps = self.speed

        for index in range(steps):
            if not self.path:
                break

            next_x, next_y = self.path.pop(0)

            if (
                0 <= next_x < self.battlefield.num_cols and 
                0 <= next_y < self.battlefield.num_rows and
                self.battlefield.obstacle_grid[next_y, next_x] == 0 and 
                self.battlefield.criminal_grid[next_y, next_x] == 0
            ):
       
                self.battlefield.criminal_grid[self.grid_y, self.grid_x] = 0
                self.grid_x, self.grid_y = next_x, next_y
                self.pixel_x = self.grid_x * self.battlefield.cell_size
                self.pixel_y = self.grid_y * self.battlefield.cell_size
                self.battlefield.criminal_grid[next_y, next_x] = 2

         
                dx = next_x - self.grid_x
                dy = next_y - self.grid_y
                if dx == 1:
                    self.direction = "RIGHT"
                    self.ahead_dx, self.ahead_dy = (1, 0)
                    self.left_dx, self.left_dy = (0, -1)
                    self.right_dx, self.right_dy = (0, 1)
                elif dx == -1:
                    self.direction = "LEFT"
                    self.ahead_dx, self.ahead_dy = (-1, 0)
                    self.left_dx, self.left_dy = (0, 1)
                    self.right_dx, self.right_dy = (0, -1)
                elif dy == 1:
                    self.direction = "DOWN"
                    self.ahead_dx, self.ahead_dy = (0, 1)
                    self.left_dx, self.left_dy = (1, 0)
                    self.right_dx, self.right_dy = (-1, 0)
                elif dy == -1:
                    self.direction = "UP"
                    self.ahead_dx, self.ahead_dy = (0, -1)
                    self.left_dx, self.left_dy = (-1, 0)
                    self.right_dx, self.right_dy = (1, 0)

       
                self.scan_for_bots()


        if self.state in ["walking", "running"]:
            self.emit_sound(self.state)

        self.moving = False
        if self.battlefield.canvas:
            self.battlefield.draw(self.battlefield.canvas)
     #   self.battlefield.canvas.after(500, self.step)




    def find_hiding_spot(self):

        possible_hiding_spots = []
        for y in range(self.battlefield.num_rows):
            for x in range(self.battlefield.num_cols):
                if self.battlefield.obstacle_grid[y, x] == 3:  
           
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        hx, hy = x + dx, y + dy
                        if 0 <= hx < self.battlefield.num_cols and 0 <= hy < self.battlefield.num_rows:
                            if self.battlefield.obstacle_grid[hy, hx] == 0:
                                possible_hiding_spots.append((hx, hy))

        if possible_hiding_spots:

            closest_hiding_spot = None
            shortest_distance = float("inf")

            for spot in possible_hiding_spots:
                spot_x, spot_y = spot
                distance = abs(spot_x - self.grid_x) + abs(spot_y - self.grid_y)  # Manhattan distance eq

                if distance < shortest_distance:
                    shortest_distance = distance
                    closest_hiding_spot = spot  

    
            hx, hy = closest_hiding_spot
            self.grid_x, self.grid_y = hx, hy
            self.pixel_x = self.grid_x * self.battlefield.cell_size
            self.pixel_y = self.grid_y * self.battlefield.cell_size
            print("Criminal is now Hiding at (" + str(self.grid_x) + ", " + str(self.grid_y) + ")")


    def compute_new_path(self):

        if self.state == "hiding":
            return  # If hiding, don't compute a path

        target_x, target_y = self.get_destination()
        self.path = aStarSearch(self.battlefield, (self.grid_x, self.grid_y), (target_x, target_y))

    def get_destination(self):

        while True:
            x = random.randint(0, self.battlefield.num_cols - 1)
            y = random.randint(0, self.battlefield.num_rows - 1)

            if self.battlefield.obstacle_grid[y, x] == 0:
                return x, y

    def draw(self, canvas):

        #most of this is for debugging helped me test inteligence visually and improve it 

        color = "gray" if self.state == "hiding" else "yellow"
        canvas.create_oval(
            self.pixel_x, self.pixel_y,
            self.pixel_x + self.size, self.pixel_y + self.size,
            fill=color, outline="black"
        )
        canvas.create_text(
            self.pixel_x + self.size // 2, self.pixel_y + self.size // 2,
            text="C", fill="black", font=("Arial", 10, "bold")
        )

        # Show the facing direction
        cell_size = self.battlefield.cell_size

        # AHEAD
        ahead_px = self.pixel_x + (self.ahead_dx * cell_size)
        ahead_py = self.pixel_y + (self.ahead_dy * cell_size)
        # canvas.create_oval(ahead_px - 5, ahead_py - 5, ahead_px + 5, ahead_py + 5, fill="blue")   # BLUE  AHEAD for debugging

        # LEFT
        left_px = self.pixel_x + (self.left_dx * cell_size)
        left_py = self.pixel_y + (self.left_dy * cell_size)
        # canvas.create_oval(left_px - 5, left_py - 5, left_px + 5, left_py + 5, fill="green")     # GREEN  LEFT  for debugging

        # RIGHT
        right_px = self.pixel_x + (self.right_dx * cell_size)
        right_py = self.pixel_y + (self.right_dy * cell_size)
        # canvas.create_oval(right_px - 5, right_py - 5, right_px + 5, right_py + 5, fill="orange") # ORANGE RIGHT  for debugging
    


    def emit_sound(self, sound_type):

        if not self.alive or not CONFIG.get("enable_sound"):
            return  
        if not self.alive:
            return
        is_new_sound = sound_type != self.last_emitted_sound
        if is_new_sound:
            self.last_emitted_sound = sound_type
        

        sound_id = sound_type + "_" + str(int(time.time() * 1000))


        self.battlefield.communication_system.assign_bots_to_sound(
        self.battlefield.bots,
        sound_type,
        self.grid_x,
        self.grid_y,
        sound_id
    )
        



    def find_safe_spot(self):

        bots = self.battlefield.bots
        max_dist = -1
        best_spot = (self.grid_x, self.grid_y)

        for _ in range(30):  # Try up to 30 potential spots
            x = random.randint(0, self.battlefield.num_cols - 1)
            y = random.randint(0, self.battlefield.num_rows - 1)

            if self.battlefield.obstacle_grid[y, x] == 0:
                min_bot_dist = min(
                    abs(bot.grid_x - x) + abs(bot.grid_y - y) for bot in bots
                )
                if min_bot_dist > max_dist:
                    max_dist = min_bot_dist
                    best_spot = (x, y)

        return best_spot



    def compute_escape_path(self, bot_x, bot_y):
        """Plan a smart escape route away from a detected bot."""
        dx = self.grid_x - bot_x
        dy = self.grid_y - bot_y

        # Normalize the escape vector direction
        dx = int(np.sign(dx))
        dy = int(np.sign(dy))

    
        escape_targets = [
            (self.grid_x + dx * 5, self.grid_y + dy * 5),          
            (self.grid_x + dx * 4, self.grid_y + dy * 4),          
            (self.grid_x + dy * 4, self.grid_y - dx * 4),         
            self.find_safe_spot()                               
        ]

        for tx, ty in escape_targets:
            if 0 <= tx < self.battlefield.num_cols and 0 <= ty < self.battlefield.num_rows:
                if self.battlefield.obstacle_grid[ty, tx] != 3:
                    path = aStarSearch(
                        self.battlefield,
                        (self.grid_x, self.grid_y),
                        (tx, ty)
                    )
                    if path:
                        self.path = path

                        print("Criminal " + str(self.id) + " plotted escape route from bot at (" + str(bot_x) + ", " + str(bot_y) + ")")

                        return


        print("Criminal " + str(self.id) + " has now failed to find an escape route! RECALCULATING")


        


    def scan_for_bots(self):
        if not self.alive:
            return

        vision_ranges = {
            1: 1.0,
            4: 0.75,
            8: 0.5,
            12: 0.25
        }

        directions = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0)
        }

        if self.direction not in directions:
            return

        dx, dy = directions[self.direction]

        def scan_direction(dx, dy, max_distance):
            for distance in range(1, max_distance + 1):
                check_x = self.grid_x + dx * distance
                check_y = self.grid_y + dy * distance

                if not (0 <= check_x < self.battlefield.num_cols and 0 <= check_y < self.battlefield.num_rows):
                    break

                accuracy = 1.0
                for threshold, acc in vision_ranges.items():
                    if distance >= threshold:
                        accuracy = acc

                if self.battlefield.obstacle_grid[check_y, check_x] == 3:
                    break  # obstacle blocks vision

                # Check for a bot
                for bot in self.battlefield.bots:
                    if bot.grid_x == check_x and bot.grid_y == check_y:
                       
                        print("Criminal " + str(self.id) + " sees Bot " + str(bot.id) + " at (" + str(check_x) + "," + str(check_y) + ") with " + str(int(accuracy * 100)) + "% accuracy!")


                    
                        if accuracy >= 0.6:
                            if CONFIG.get("enable_criminal_comms") and self.comms:
                                self.comms.report_bot_sighting(self.id, bot.grid_x, bot.grid_y)
                        
                            print("Criminal " + str(self.id) + " is running from Bot " + str(bot.id) + "!")


                            self.path = []  # Drop  the old path regardless of the current state
                            self.compute_escape_path(bot.grid_x, bot.grid_y)

                            if self.state != "running":
                                self.state = "running"
                                self.speed = 2
                                self.hiding = False
                                self.state_timer = random.randint(3, 6)
                            

                            self.compute_escape_path(bot.grid_x, bot.grid_y)


                            pass
                          #  self.run_from_bot(bot)
                          #  self.alert_other_criminals(bot.grid_x, bot.grid_y)
                        return  # stop after seeing 1 bot

        scan_direction(dx, dy, 20)

        # Optional: scan left/right too like bots
        left_dx, left_dy = -dy, dx
        right_dx, right_dy = dy, -dx
        scan_direction(left_dx, left_dy, 20)
        scan_direction(right_dx, right_dy, 20)









class Obstacle:
    def __init__(self, battlefield):
        self.battlefield = battlefield
        self.size = random.choice([battlefield.cell_size, battlefield.cell_size * 2])  

        while True:
            grid_x = random.randint(0, battlefield.num_cols - 1)
            grid_y = random.randint(0, battlefield.num_rows - 1)

  
            num_cells = self.size // battlefield.cell_size  

            if all(
                0 <= grid_x + dx < battlefield.num_cols and
                0 <= grid_y + dy < battlefield.num_rows and
                battlefield.obstacle_grid[grid_y + dy, grid_x + dx] == 0  
                for dx in range(num_cells) for dy in range(num_cells)
            ):
              
                for dx in range(num_cells):
                    for dy in range(num_cells):
                        battlefield.obstacle_grid[grid_y + dy, grid_x + dx] = 3  

              
                self.grid_x = grid_x
                self.grid_y = grid_y
                self.pixel_x = self.grid_x * battlefield.cell_size
                self.pixel_y = self.grid_y * battlefield.cell_size
                break  

    def draw(self, canvas):
        canvas.create_rectangle(
            self.pixel_x, self.pixel_y,
            self.pixel_x + self.size, self.pixel_y + self.size,
            fill="black", outline="white"
        )



def main():
    retrain_on_misclassified() # retrain the neural network on misclassified specotgrams to improve the bots intelligence
    window = tk.Tk()
    window.title("Battlefield Simulation")




    
    battlefield_width = 1400
    battlefield_height = 850 
    canvas = tk.Canvas(window, width=battlefield_width, height=battlefield_height, bg="gray22")
    canvas.pack()

    #options for testing the game below

    CONFIG["enable_bot_comms"] = True 
    CONFIG["enable_criminal_comms"] = True 
    CONFIG["enable_kalman"] = True
    CONFIG["enable_sound"] = True
    CONFIG["random_movement_only"] = True



    
    battlefield = Battlefield(
        width=battlefield_width, 
        height=battlefield_height, 
        cell_size=30, 
        num_bots=3, 
        num_criminals=3,  
        num_obstacles=150,  
        show_grid=1 
    )
    battlefield.distribute_probability()



    battlefield.draw(canvas)
    battlefield.update()
    
    window.mainloop()





import csv
import time

# The different setups
SETUPS = [
    {"cnn": False, "kalman": False, "comms": False}, # Setup 1
    {"cnn": True,  "kalman": False, "comms": False}, # Setup 2
    {"cnn": True,  "kalman": False,  "comms": True}, # Setup 3
    {"cnn": True,  "kalman": True,  "comms": False},  # Setup 4
    {"cnn": True,  "kalman": True,  "comms": True}, # Setup 5 
]

# System configuration
def configure_system(setup):
    CONFIG["enable_sound_detection"] = setup["cnn"]
    CONFIG["enable_kalman"] = setup["kalman"]
    CONFIG["enable_bot_comms"] = setup["comms"]



def run_simulation_fixed_tests():
    battlefield_width = 1400
    battlefield_height = 850
    cell_size = 30
    num_bots = 3
    num_criminals = 3
    num_obstacles = 150

    battlefield = Battlefield(
        width=battlefield_width,
        height=battlefield_height,
        cell_size=cell_size,
        num_bots=num_bots,
        num_criminals=num_criminals,
        num_obstacles=num_obstacles,
        show_grid=False  # no need to draw when testing
    )
    battlefield.distribute_probability()

    start_time = time.time()
    MAX_DURATION = 120 

    success = False
    
    while len(battlefield.criminals) > 0:
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time > MAX_DURATION:
            print("Test has now timedout! Moving to the next test.")
            break

        for bot in battlefield.bots:
            bot.move()
        for criminal in battlefield.criminals:
            criminal.update_behavior()
            criminal.move()
        battlefield.check_bot_criminal_collisions()

    else:
        success = True  

    end_time = time.time()
    return end_time - start_time, success


#save results to a CSV
def save_result_fixed_tests(setup_id, setup, run_number, time_taken, success):
    file_exists = os.path.isfile('test_results.csv')
    with open('test_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Setup ID", 
                "CNN Enabled", 
                "Kalman Enabled", 
                "Bot Comms Enabled", 
                "Run Number", 
                "Time Taken (s)",
                "Outcome"
            ])
        writer.writerow([
            setup_id,
            setup["cnn"],
            setup["kalman"],
            setup["comms"],
            run_number,
            round(time_taken, 2),
            "Success" if success else "Timeout"
        ])


#  Testing loop
def run_fixed_tests():
    for setup_id, setup in enumerate(SETUPS, start=1):
        for run_number in range(1, 11):  
            print("Running Setup", setup_id, "Run", run_number)
            configure_system(setup)
            time_taken, success = run_simulation_fixed_tests()
            save_result_fixed_tests(setup_id, setup, run_number, time_taken, success)

            print("Time Completed in " + str(round(time_taken, 2)) + " seconds - " + (" Success" if success else " Timeout"))






def run_simulation_changing_numbers(num_bots, num_criminals):
    battlefield_width = 1400
    battlefield_height = 850
    cell_size = 30
    num_obstacles = 150

    # Always use same configs for this test
    CONFIG["enable_sound_detection"] = True
    CONFIG["enable_kalman"] = True
    CONFIG["enable_bot_comms"] = True
    CONFIG["enable_criminal_comms"] = True
    CONFIG["random_movement_only"] = False

    battlefield = Battlefield(
        width=battlefield_width,
        height=battlefield_height,
        cell_size=cell_size,
        num_bots=num_bots,
        num_criminals=num_criminals,
        num_obstacles=num_obstacles,
        show_grid=False
    )
    battlefield.distribute_probability()

    start_time = time.time()
    MAX_DURATION = 120  # seconds

    success = False

    while len(battlefield.criminals) > 0:
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time > MAX_DURATION:
            print("Timeout ! Moving to next test.")
            break

        for bot in battlefield.bots:
            bot.move()
        for criminal in battlefield.criminals:
            criminal.update_behavior()
            criminal.move()
        battlefield.check_bot_criminal_collisions()

    else:
        success = True

    end_time = time.time()
    return end_time - start_time, success



def save_changing_number_result(num_bots, num_criminals, run_number, time_taken, success):
    file_exists = os.path.isfile('changing_number_of_agents_results.csv')
    with open('changing_number_of_agents_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Num Bots", 
                "Num Criminals", 
                "Run Number", 
                "Time Taken (s)", 
                "Outcome"
            ])
        writer.writerow([
            num_bots,
            num_criminals,
            run_number,
            round(time_taken, 2),
            "Success" if success else "Timeout"
        ])



def run_changing_number_agents_tests():
    for num_bots in range(1, 4):  # 1, 2, 3
        for num_criminals in range(1, 4):  # 1, 2, 3
            for run_number in range(1, 11):  # 10 repeats
                print("Running", num_bots, "bots vs", num_criminals, "criminals - Run", run_number)
                time_taken, success = run_simulation_changing_numbers(num_bots, num_criminals)
                save_changing_number_result(num_bots, num_criminals, run_number, time_taken, success)
                if success:
                    print("Completed in", round(time_taken, 2), "seconds -  Success")
     

                else:
                    print("Completed in", round(time_taken, 2), "seconds -  Timeout")

if __name__ == "__main__":

    #testing options
    #retrain_on_misclassified()  # retrain the neural network once before tests to improve it adding intelligence

    #run_fixed_tests() #testing different inteligent levels of the bot for impact
    #run_changing_number_agents_tests() #changing diff number of bots and criminals 


    #print("All tests completed!")
    main() 

#main()


