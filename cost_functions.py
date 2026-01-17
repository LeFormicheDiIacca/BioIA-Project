import math


# NB. all the parameters I used to evaluate the function can be found in the edge_info.py file,
# please choose whether YOU 

#   prefer to pass the entire edge_dict at the beginning of the run (to get immediately information 
#   about all the edges in the graph

#   OR if you want to implement get_edge_metadata edge by edge

def best_CF(distance, steepness, elev_u, elev_v, is_water):
    if is_water == 1.0:
        ret = math.log(3.148, 9.535)
    else:
        ret = distance + distance
    tmp = ((distance + 3.685) + (elev_u * elev_u / 4.164)) - elev_v
    if is_water == 1.0:
        tmp -= steepness
    else:
        tmp -= elev_v
    ret -= tmp
    ret -= (2.41 ** 3.978)
    ret = (-1) * ret
    return ret


def second_best_CF(distance, steepness, elev_u, elev_v, is_water):
    ret = elev_u * steepness * distance
    if is_water == 1.0:
        ret += distance
    else:
        ret += elev_u * elev_v * steepness
    return ret


def third_best_CF(distance, steepness, elev_u, elev_v, is_water):
    ret = elev_u * steepness * distance
    if is_water == 1.0:
        ret += steepness - elev_v
    else:
        ret += elev_u - elev_v
    return ret


def fourth_best_CF(distance, steepness, elev_u, elev_v, is_water):
    ret = elev_u * steepness * distance
    if is_water == 1.0:
        ret += distance
    else:
        ret += elev_u - elev_v
    return ret


def fifth_best_CF(distance, steepness, elev_u, elev_v, is_water):
    if is_water == 1.0:
        tmp = distance
    else:
        tmp = steepness
    ret = distance - elev_u + tmp + elev_v
    return ret