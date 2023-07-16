# -*- coding: utf-8 -*-
"""
Template for the main solution code in Part 1 of the project.

The objective of the template files is to give you an idea of
the most important functions that you have to implement, what
input they will need and what output they should produce.

To make things work in practice you will have to add more
functionalities than the ones outlined, and you might have
to adapt the interfaces to fit your specific approach.

You are of course free (and encouraged) to structure the code
in any way you want!
"""
from typing import Tuple
import numpy as np
from numba import jit
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.space_mission import SpaceMission


def check_distance_0(particle_position: np.ndarray, 
                     coordinate: int, 
                     particle_velocity: np.ndarray) \
                     -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        particle_position (np.ndarray): _description_
        coordinate (int): _description_
        particle_velocity (np.ndarray): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """

    try:
        coordinate != 2

        distance = np.abs(particle_position[:, coordinate] - box_side_length)

        less_than_zero = np.where(particle_position[:, coordinate] < 0)

        particle_position[less_than_zero, coordinate] += distance[less_than_zero]
        particle_velocity[less_than_zero, coordinate] *= -1

        return particle_position, particle_velocity
    except:
        print("Error: coordinate must be 0 or 1")
        return None


def check_distance_and_escaped(particle_position: np.ndarray,
                               coordinate: int,
                               particle_velocity: np.ndarray,
                               velocity_constant: float,
                               box_hole_area:float) \
                               -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        particle_position (np.ndarray): _description_
        coordinate (int): _description_
        particle_velocity (np.ndarray): _description_
        velocity_constant (float): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """

    x = coordinate - 2
    y = coordinate - 1
    
    try:
        y > 0
    except:
        raise ValueError("Error: y coordinate must be 1")
    
    try:
        x > -1
    except:
        raise ValueError("Error: x coordinate must be 0")
        

    # * Check if particle has escaped
    # *
    # * hole in box is the square (L/2, L/2) to (3/2 L, 3/2 L), elsewhere the particle simply bounces back
    # * if particle escapes box, it is simply places at the top of the box with new random velocity

    sigma = 1 / (2 * np.pi * velocity_constant ** (2 / 3))

    distance = np.abs(particle_position[:, coordinate] - box_side_length)
    
    sides = np.sqrt(box_hole_area)
    
    
    
    

    """condition_for_inside_hole = (
        particle_position[:, x] <= 3 / 2 * box_side_length
        and particle_position[:, x] >= 1 / 2 * box_side_length
        and particle_position[:, y] <= 3 / 2 * box_side_length
        and particle_position[:, y] >= 1 / 2 * box_side_length
    )"""

    inside_hole = np.where(
        (particle_position[:, coordinate] < 0) & 
        (particle_position[:, x] <= box_side_length + sides) & 
        (particle_position[:, x] >= box_side_length - sides) & 
        (particle_position[:, y] <= box_side_length + sides) & 
        (particle_position[:, y] >= box_side_length - sides)
    )
    
    
    
    
    outside_hole = np.where(
        (particle_position[:, coordinate] < 0) & 
        (particle_position[:, x] >= box_side_length + sides) &
        (particle_position[:, x] <= box_side_length - sides) & 
        (particle_position[:, y] >= box_side_length + sides) & 
        (particle_position[:, y] <= box_side_length - sides)
    )

    particle_position[outside_hole, coordinate] += distance[outside_hole]
    particle_velocity[outside_hole, coordinate] *= -1

    particle_position[inside_hole, coordinate] = box_side_length
    """particle_velocity[inside_hole, :] = velocity_constant * np.random.normal(
        0, sigma, size=(len(inside_hole), 3)
    )"""

    escaped_particles = len(inside_hole[0])
   
    
    tot_escaped_momentum = np.abs(2*M_H2*np.sum(particle_velocity[inside_hole, coordinate]))

    return particle_position, particle_velocity, escaped_particles, tot_escaped_momentum


def check_distance_L(particle_position: np.ndarray,
                     coordinate: int,
                     particle_velocity: np.ndarray,
                     L: float,) \
                     -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        particle_position (np.ndarray): _description_
        coordinate (int): _description_
        particle_velocity (np.ndarray): _description_
        L (float): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """
    distance = np.abs(particle_position[:, coordinate] - box_side_length)

    less_than_L = np.where(particle_position[:, coordinate] > L)

    particle_position[less_than_L, coordinate] -= distance[less_than_L]
    particle_velocity[less_than_L, coordinate] *= -1

    return particle_position, particle_velocity

@jit(nopython=True)
def simulate_engine_performance(
    number_of_particles_in_box: int,
    box_side_length: float,
    temperature: float,
    box_hole_area: float,
) -> Tuple[float, float]:
    """
    First, each box has a a side, where the label is as follows:

    1: top
    2: bottom
    3: left
    4: right
    5: front
    6: back


    Args:
        number_of_particles_in_box (int): _description_
        box_side_length (float): _description_
        temperature (float): _description_
        box_hole_area (float): _description_

    Returns:
        Tuple[float, float]: _description_
    """

    
    sigma = np.sqrt(K_B * temperature / M_H2)
    tau = 1e-9
    velocity_constant = (1 / (2 * np.pi * sigma)) ** (3 / 2)

    particle_position = np.random.uniform(
        0, box_side_length, (number_of_particles_in_box, 3)
    )
    particle_velocity = velocity_constant * np.random.normal(
        0, sigma, (number_of_particles_in_box, 3)
    )

    tot_particles_escaped = 0
    tot_escaped_momentum = 0
    
    for t in range(1000):
        particle_position += particle_velocity * tau

        # * Check for wall overstep
        particle_position, particle_velocity = check_distance_0(
            particle_position, 0, particle_velocity
        )
        particle_position, particle_velocity = check_distance_0(
            particle_position, 1, particle_velocity
        )

        # * Check also for nuzzle
        particle_position, particle_velocity, escaped_particles, escaped_momentum = check_distance_and_escaped(
            particle_position, 2, particle_velocity, velocity_constant, box_hole_area
        )
        
        tot_particles_escaped += escaped_particles
        tot_escaped_momentum += escaped_momentum
        
        

        particle_position, particle_velocity = check_distance_L(
            particle_position, 0, particle_velocity, box_side_length
        )
        particle_position, particle_velocity = check_distance_L(
            particle_position, 1, particle_velocity, box_side_length
        )
        particle_position, particle_velocity = check_distance_L(
            particle_position, 2, particle_velocity, box_side_length
        )

    
    
    

    thrust_per_box = tot_escaped_momentum / 1e-12
    mass_loss_rate_per_box = M_H2*tot_particles_escaped
    return thrust_per_box, mass_loss_rate_per_box


def compute_fuel_mass_needed_for_boost(
    mission, 
    rocket_thrust, rocket_mass_loss_rate, initial_fuel_mass, target_delta_v
):
    """_summary_

    Args:
        mission (ast2000tools.space_mission.SpaceMission): Mission for the rocket
        rocket_thrust (float): Thrust in Newtons for rocket
        rocket_mass_loss_rate (float): Loss rate of mass for a given second
        initial_fuel_mass (float): Initial fuel mass for rocket at a given moment
        target_delta_v (float): Target delta v for rocket direction adjustment

    Returns:
        fuel_mass_used (float): Used fuel mass for rocket
    """
    
    SPACECRAFT_MASS = mission.spacecraft_mass
    exhaust_velocity = rocket_thrust / rocket_mass_loss_rate
    
    fuel_mass_used = (SPACECRAFT_MASS + initial_fuel_mass) * (1- np.exp(target_delta_v / exhaust_velocity))

    return fuel_mass_used

def coriolis_effect(rocket_velocity, planet_rotational_speed, rocket_position):
    """_summary_

    Args:
        rocket_velocity (_type_): _description_
        planet_rotational_speed (_type_): _description_
        rocket_position (_type_): _description_

    Returns:
        _type_: _description_
    """
    coriolis_force = 2 * np.cross(rocket_velocity, planet_rotational_speed)
    return coriolis_force

def simulate_rocket_launch(
    mission, rocket_thrust, rocket_mass_loss_rate, initial_fuel_mass
):
    
    
    
    
    GRAVI_ACCELERATION = const.G
    SUN_MASS = const.m_sun
    PLANETS_RADII = mission.system.radii
    SPACESHIP_MASS = mission.spacecraft_mass
    PLANETS_MASSES = mission.system.masses
    
    PLANET_RADII = PLANETS_RADII[0] #* In Km
    PLANET_MASS = PLANETS_MASSES[0]*SUN_MASS #* In Kg
    
    rotational_period_per_day = mission.system.rotational_periods[0]
    rotational_speed = 2*np.pi/(rotational_period_per_day*24*60*60)
    
    
    
    #* Integration of rocket path 
    
    T = 20*60 #* 20 minutes in seconds 
    N = int(1e4)
    dt = T / N 
    
    rocket_position = np.zeros((N-1, 2))
    rocket_position[0, 0] = PLANET_RADII*1000 #* In Meters
    rocket_velocity = np.zeros((N-1, 2))
    
    
    v_escape = lambda r: np.sqrt((2*GRAVI_ACCELERATION*PLANET_MASS)/(r))
    grav_acc = lambda r: -(GRAVI_ACCELERATION*PLANET_MASS)/(r)**2
    
    #* Using Leapfrog method:
    
    rocket_mass_update = lambda rocket_fuel: SPACESHIP_MASS + rocket_fuel 
    
    SPACESHIP_MASS_TOTAL = rocket_mass_update(initial_fuel_mass)
    
    t = 0
    for i in range(N):
        distance_from_planet_center_curr = np.linalg.norm(rocket_position)
        coriolis_force = coriolis_effect(rocket_velocity[i, :], np.array([0, rotational_speed]), rocket_position[i, :])
        
        total_accelleration = (np.array([0, grav_acc(distance_from_planet_center_curr)]) + np.array([0, rocket_thrust]) + np.array([0, coriolis_force]))/SPACESHIP_MASS_TOTAL
    
        rocket_position[i+1, :] = rocket_position[i, :] + rocket_velocity[i, :]*dt + 0.5*total_accelleration*dt**2
        
        distance_from_planet_center_next = np.linalg.norm(rocket_position)
        
        delta_v = np.sqrt((total_accelleration[0])**2 + (total_accelleration[1])**2)*dt
        
        fuel_consumed = compute_fuel_mass_needed_for_boost(mission, rocket_thrust, rocket_mass_loss_rate, initial_fuel_mass, delta_v)
        print(fuel_consumed)
        initial_fuel_mass -= fuel_consumed
        print("Fuel mass", initial_fuel_mass)
        SPACESHIP_MASS_TOTAL = rocket_mass_update(initial_fuel_mass)
        
        total_accelleration_next = (np.array([0, -grav_acc(distance_from_planet_center_next)]) + np.array([0, rocket_thrust]) + np.array([0, coriolis_force]))/SPACESHIP_MASS_TOTAL
        
        rocket_velocity[i+1, :] = rocket_velocity[i, :] + 0.5*(total_accelleration + total_accelleration_next)*dt
        
        
        v = np.linalg.norm(rocket_velocity)
        
        print("V: {:g} M: {:g} G: {:g}".format(v_escape(distance_from_planet_center_next)-v, SPACESHIP_MASS_TOTAL, grav_acc(distance_from_planet_center_next)))

        
        t += dt
    
    
    final_height_above_surface = 0
    final_upward_speed = 0
    fuel_mass_after_launch = 0
    launch_duration = 0
    
    return (
        final_height_above_surface,
        final_upward_speed,
        fuel_mass_after_launch,
        launch_duration,
    )


def convert_to_solar_system_frame_simple(
    system, final_height_above_surface, final_upward_speed, launch_duration
):
    """
    Here you can implement the coordinate system conversion for
    challenge F of Part 1 of the project.
    """

    # Insert awesome code here...

    # You will probably also need these quantities:
    # const.AU
    # const.day
    # const.yr
    # system.radii
    # system.initial_positions
    # system.initial_velocities
    # system.rotational_periods

    # Return quantities in the solar system units and frame of reference
    return (
        rocket_position_before_launch,
        rocket_position_after_launch,
        rocket_velocity_after_launch,
        time_after_launch,
    )


def run_engine_and_launch_simulation(
    mission,
    number_of_boxes,
    number_of_particles_in_box,
    box_side_length,
    temperature,
    box_hole_area,
    initial_fuel_mass,
):
    """
    This function executes the simulation code for the rocket engine and launch,
    and returns the results.
    """

    # Simulate engine
    thrust_per_box, mass_loss_rate_per_box = simulate_engine_performance(
        number_of_particles_in_box, box_side_length, temperature, box_hole_area
    )

    # Compute rocket_thrust and rocket_mass_loss_rate here..
    # 
    rocket_thrust = thrust_per_box * number_of_boxes
    rocket_mass_loss_rate = mass_loss_rate_per_box * number_of_boxes
    
    initial_fuel_mass = number_of_boxes * number_of_particles_in_box * M_H2
    
    print(rocket_thrust,rocket_mass_loss_rate,initial_fuel_mass)

    # Simulate launch
    (
        final_height_above_surface,
        final_upward_speed,
        fuel_mass_after_launch,
        launch_duration,
    ) = simulate_rocket_launch(
        mission, rocket_thrust, rocket_mass_loss_rate, initial_fuel_mass
    )

    """return (
        thrust_per_box,
        mass_loss_rate_per_box,
        final_height_above_surface,
        final_upward_speed,
        fuel_mass_after_launch,
        launch_duration,
    )"""
    return None


# Prevent the following code from executing when calling `import part_1`
if __name__ == "__main__":
    # Print a message if a newer version of ast2000tools is available
    # * utils.check_for_newer_version()

    # Construct SpaceMission instance for my mission
    seed = utils.get_seed("09012023")
    mission = SpaceMission(seed)

    # Extract associated SolarSystem object
    system = mission.system

    global K_B, M_H2
    
    K_B = const.k_B
    M_H2 = const.m_H2
   
    box_side_length = 5e-6  # * Meters
    box_hole_area = 1/16 * box_side_length * box_side_length
    number_of_particles_in_box = int(1e6)
    temperature = 4e3   # * Kelvin
    number_of_boxes = 60*1e21
    initial_fuel_mass = number_of_boxes * number_of_particles_in_box * M_H2

    # Run engine and launch simulations and get results relative to planet
    """thrust_per_box,             \
    mass_loss_rate_per_box,     \
    final_height_above_surface, \
    final_upward_speed,         \
    fuel_mass_after_launch,     \
    launch_duration             \ 
    =
    """
    run_engine_and_launch_simulation(mission,
                                         number_of_boxes,
                                         number_of_particles_in_box,
                                         box_side_length,
                                         temperature,
                                         box_hole_area,
                                         initial_fuel_mass)

    """
    # Convert simulated launch results to the solar system frame,
    # assuming launch along the x-direction at time = 0 years
    rocket_position_before_launch, \
    rocket_position_after_launch,  \
    rocket_velocity_after_launch,  \
    time_after_launch              \
      = convert_to_solar_system_frame_simple(system,
                                             final_height_above_surface,
                                             final_upward_speed,
                                             launch_duration)
    time_of_launch = 0.0

    # Perform real launch
    mission.set_launch_parameters(thrust_per_box*number_of_boxes,
                                  mass_loss_rate_per_box*number_of_boxes,
                                  initial_fuel_mass,
                                  launch_duration,
                                  rocket_position_before_launch,
                                  time_of_launch)
    mission.launch_rocket()

    # Verify simulated launch results
    mission.verify_launch_result(rocket_position_after_launch)
    """
