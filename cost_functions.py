import numpy as np
from GP.gp_logistics import protected_pow, protected_log, protected_div

#Trentino
def trentino_first_cost_function(distance, steepness, is_water):
    term1 = protected_pow(distance,0.592)
    term_2 = np.where(is_water,protected_pow(distance,steepness-0.827), steepness)
    term_2 += 1.378+distance
    return term1 + term_2

def trentino_second_cost_function(distance, steepness, is_water):
    tmp_1= np.where(is_water, protected_log(distance, 0.628), steepness)
    temp2 = 1.837 + distance + protected_pow(distance,0.628)
    return tmp_1 + temp2

def trentino_third_cost_function(distance, steepness, is_water):
    tmp_1 = np.where(is_water, 1/distance, steepness)
    tmp_2 = 1.439++distance+protected_pow(distance,0.647)
    return tmp_1 + tmp_2

def trentino_fourth_cost_function(distance, steepness, is_water):
    tmp_1 = np.where(is_water,0.884, steepness)
    tmp_2 = protected_pow(distance,0.586)+1.380+distance
    return tmp_1 + tmp_2
#Naples

def naples_first_cost_function(distance, steepness, is_water):
    tmp_1 = np.where(is_water, 0.117, 0.11)
    tmp_2 = 0.084 *(steepness+distance)
    return  tmp_1 * tmp_2

def naples_second_cost_function(distance, steepness, is_water):
    tmp_1 = np.where(is_water, 0.517, steepness)
    tmp_2 = distance+0.803+protected_div(distance+steepness,0.153)
    return tmp_1 + tmp_2

def naples_third_cost_function(distance, steepness, is_water):
    tmp_1 =np.where(is_water, 0.215, steepness)
    return protected_pow(tmp_1+0.786+distance, 0.786)

def naples_fourth_cost_function(distance, steepness, is_water):
    #Not so good
    tmp_1 = np.where(is_water, 0.629, steepness)
    return 0.009* distance+tmp_1