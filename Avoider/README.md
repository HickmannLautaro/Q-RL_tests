# Avoider environment
Made more simple and discrete.

## Example
* Gray squares are stones
* Green square are when stones are avoided. 
* White rectangle is the Agent.
* White vertical lines are used as line separation

## Notation
- {n_1,n_2}: represents sets with elements n_1 and n_2  
- \[n_1,n_2\]: represents interval of real numbers between n_1 and n_2


## phase 1 (v_0)

### Played by hand
![](avoider_phase_1_v_0_demo.gif)

## Specifications
per step (independent of agents action) the obstacles go lower in the screen by the same amount (jump to next higher y coordinate). 
1 live.
### State space:
s = {s_1,s_2,s_3,s_4} with:
* s_1 ∈  {-1, 0, 1}={Left line, middle line, right lane} players x centre position.
* s_2 ∈  {1.00 , 0.73 , 0.47, 0.20, 0.00, -0.27, -1} = {Top of the screen line, intermediate position 1, intermediate position 2, intermediate position 3, bottom of the screen (catcher position), under catcher i.e. the obstacle was avoided, no obstacle in line} obstacle in first lane y centre position
* s_3 ∈  {1.00 , 0.73 , 0.47, 0.20, 0.00, -0.27, -1} = {Top of the screen line, intermediate position 1, intermediate position 2, intermediate position 3, bottom of the screen (catcher position), under catcher i.e. the obstacle was avoided, no obstacle in line} obstacle in second lane y centre position
* s_4 ∈  {1.00 , 0.73 , 0.47, 0.20, 0.00, -0.27, -1} = {Top of the screen line, intermediate position 1, intermediate position 2, intermediate position 3, bottom of the screen (catcher position), under catcher i.e. the obstacle was avoided, no obstacle in line} obstacle in third lane y centre position


## phase 2 (v_1)
 
Has some bugs depending on velocity if crashes are detected

### Played by hand
![](avoider_phase_1_v_1_demo.gif)

## Specifications
per step (independent of agents action) the obstacles  go lower in the screen depending on their speed. 
1 live.
### State space:
s = {s_1,s_2,s_3,s_4, s_5,s_6,s_7} with:
* s_1 ∈  {-1, 0, 1}={Left line, middle line, right lane} players x centre position.
-1.00 is the default value when no obstacle present in lane
* s_2 ∈  [-1.00 , 1.00] obstacle in first lane y centre position (if more than one obstacle in line the nearest to the agent is measured) 
* s_3 ∈  [0.00 , 1.00] U [-1.00] obstacle in first lane y velocity (if more than one obstacle in line the nearest to the agent is measured)
* s_4 ∈  [-1.00 , 1.00] obstacle in second lane y centre position (if more than one obstacle in line the nearest to the agent is measured) 
* s_5 ∈  [0.00 , 1.00] U [-1.00] obstacle in second lane y velocity (if more than one obstacle in line the nearest to the agent is measured)
* s_6 ∈  [-1.00 , 1.00] obstacle in third lane y centre position (if more than one obstacle in line the nearest to the agent is measured) 
* s_7 ∈  [0.00 , 1.00] U [-1.00] obstacle in third lane y velocity (if more than one obstacle in line the nearest to the agent is measured)



# Common to all

### Action space:
action_names = {left, stay (do nothing), right} coded as {97,None, 100}  
 * left: (if possible) jump one line to the left
 * stay: stay in the same lane
 * right: (if possible) jump one line to the right

### rewards
r ∈  {1.0,0.0,-10.0}
* 1 obstacle avoided (stone at same y position as agent but not same x position)
* 0 not crashed and not avoided
* -10 crashed into obstacle. Game Over

## Task solved criterium 

The variable `steps_per_episode` sets the maximal reward pro episode,  as `max_score = steps_per_episode / 5` (for now whit variable velocity something new should be defined)
If the episode cumulative reward averaged over the last 100 episodes is `≥ max_score * 0.98` for 100 consecutive episodes, stop training and set episode cumulative reward for the rest of the episodes to the average reward achieved until now.

Train for `n` steps independent of results. With `n_q` for Quantum and `n_c` for classic experiments. 


