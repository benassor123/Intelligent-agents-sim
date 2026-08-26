import heapq

def aStarSearch(battlefield, start, goal):

    def heuristic(a, b):  # Manhattan distance heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    open_list = []  
    heapq.heappush(open_list, (0, start))
    came_from = {}
    cost_so_far = {start: 0}



    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:

            break
        
        x, y = current
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

        for nx, ny in neighbors:
            if 0 <= nx < battlefield.num_cols and 0 <= ny < battlefield.num_rows:


                cell_value = battlefield.obstacle_grid[ny, nx]


                if cell_value == 3:  

                    continue  # Skip obstacle cells
                
                new_cost = cost_so_far[current] + 1
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    priority = new_cost + heuristic((nx, ny), goal)
                    heapq.heappush(open_list, (priority, (nx, ny)))
                    came_from[(nx, ny)] = current
    

    path = []
    if goal in came_from:
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path
    else:

        print(" No valid path found from " + str(start) + " to " + str(goal) + ". Bot will not move.")

        return []  
