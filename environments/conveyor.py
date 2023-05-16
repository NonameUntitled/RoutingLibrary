from environments import BaseEnvironment


class ConveyorEnvironment(BaseEnvironment, config_name='conveyor'):
    pass


POS_ROUND_DIGITS = 3
SOFT_COLLIDE_SHIFT = 0.2

BAG_RADIUS = 0.5
BAG_MASS = 20
BAG_DENSITY = 40

BELT_WIDTH = 1.5
BELT_UNIT_MASS = 15

# SPEED_STEP = 0.1
# SPEED_ROUND_DIGITS = 3

# MODEL_BELT_LEN = 313.25

_G = 9.8
_FRICTON_COEFF = 0.5
_THETA_1 = 1 / (6.48 * BELT_WIDTH * BELT_WIDTH * BAG_DENSITY)
_THETA_2_L = _G * _FRICTON_COEFF * BELT_UNIT_MASS
_k_3 = 50000
_THETA_3 = 0.0031
_THETA_4_L = _G * _FRICTON_COEFF
_k_2 = 1

_ETA = 0.8


# THETAS_ORIG = [2.3733e-4, 8566.3, 0.0031, 51.6804]

def _P_Zhang(length: float, V: float, T: float):
    return _THETA_1 * V * T * T + (_THETA_2_L * length + _k_3) * V + \
           _THETA_3 * T * T / V + (_THETA_4_L * length / 3.6 + _k_2) * T + V * V * T / 3.6


def consumption_Zhang(length, speed, n_bags):
    if speed == 0:
        return 0
    Q_g = n_bags * BAG_MASS / length
    T = Q_g * speed * 3.6
    return _P_Zhang(length, speed, T) / _ETA


def consumption_linear(length, speed, n_bags):
    return _G * _FRICTON_COEFF * BAG_MASS * n_bags * speed / _ETA
