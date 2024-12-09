#use :  source .venv/bin/activate to activate enviornment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

def lla_to_ecef(phi, lam, h):
    """
    Converts latitude, longitude, and altitude to ECEF coordinates.
    
    Parameters:
        phi (float): Latitude in degrees.
        lam (float): Longitude in degrees.
        h (float): Altitude in meters.
        
    Returns:
        list: ECEF coordinates [x, y, z] in meters.
    """
    # Convert degrees to radians
    phi = np.radians(phi)
    lam = np.radians(lam)
    
    # Earth parameters
    Re = 6378.137 * 1000  # Radius of the Earth in meters
    f = 1 / 298.257223563  # Flattening parameter of the ellipsoid
    e_sqrd = 2 * f - f**2  # Square of eccentricity of the ellipsoid
    
    # Curvature of Earth at the location
    Ce = Re / np.sqrt(1 - e_sqrd * (np.sin(phi))**2)
    
    # ECEF coordinates
    x = (Ce + h) * np.cos(phi) * np.cos(lam)
    y = (Ce + h) * np.cos(phi) * np.sin(lam)
    z = (Ce * (1 - e_sqrd) + h) * np.sin(phi)
    
    return [x, y, z]

lat = -50
lon = -90
observerxyz = np.array(lla_to_ecef(lat,lon,16))/1000 #(52, -100, 68, 16, 42, 30), new(-50, -90, 35, 9, 10, 17)


def load_file(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data

import numpy as np

def ecef_to_enu(lat_deg, lon_deg):
    """
    Computes the transformation matrix from ECEF to ENU coordinates.

    Parameters:
        lat_deg (float): Latitude in degrees.
        lon_deg (float): Longitude in degrees.

    Returns:
        numpy.ndarray: 3x3 transformation matrix.
    """
    # Convert degrees to radians
    φ = np.deg2rad(lat_deg)
    λ = np.deg2rad(lon_deg)
    
    # Compute the transformation matrix
    C = np.array([
        [-np.sin(λ), np.cos(λ), 0],
        [-np.sin(φ) * np.cos(λ), -np.sin(φ) * np.sin(λ), np.cos(φ)],
        [np.cos(φ) * np.cos(λ), np.cos(φ) * np.sin(λ), np.sin(φ)]
    ])
    
    return C

eceftoenu = ecef_to_enu(lat,lon) #rotation matrix #(34, 10, 63, 46, 55, 53)

positions = load_file("positions.json")
positions_ENU = positions
#turn positions into enu
def apply_rotation_to_all_depths(positions, rotation_matrix):
    """
    Applies a rotation matrix to all depths in a 3D list of positions.

    Parameters:
        positions (list of list of list): The 3D list of positions (e.g., satellites at depths).
        rotation_matrix (numpy.ndarray): 3x3 rotation matrix.

    Returns:
        list of list of list: Transformed positions with the same structure as the input.
    """
    rotated_positions = []
    
    for satellite_positions in positions:
        # Apply the rotation to each depth for the current satellite
        rotated_satellite_positions = [np.dot(rotation_matrix, position).tolist() for position in satellite_positions]
        rotated_positions.append(rotated_satellite_positions)
    
    return rotated_positions

positions_ENU = apply_rotation_to_all_depths(positions, eceftoenu)

visibility = load_file("visibility.json")

def plot_satellite_positionsECEF(positions, visibility, userxyz,coordsys,depth):
    """
    Plots satellite positions in 3D with a sphere representing the Earth.
    
    Parameters:
        positions (list): List of satellite positions in the form [[x, y, z], ...].
        visibility (list): List indicating visibility of satellites (1 for visible, 0 for not visible).
        userxyz (list): User's position [x, y, z].
        coordsys: (string) of ENU or ECEF
        depth (int): depth at which you want to plot, 0 for first time step
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Define the WGS84 ellipsoid parameters

    a = 6378.137  # semi-major axis (meters)
    f = 1 / 298.257223563  # flattening
    b = a * (1 - f)  # semi-minor axis
    # Generate u, v for parametric plot of the ellipsoid
    u = np.linspace(0, 2 * np.pi, 100)  # azimuthal angle
    v = np.linspace(0, np.pi, 100)  # polar angle
    # Parametric equations for the ellipsoid
    x = a * np.outer(np.cos(u), np.sin(v))
    y = a * np.outer(np.sin(u), np.sin(v))
    z = b * np.outer(np.ones(np.size(u)), np.cos(v))
    # Create a 3D plot
    # Plot the ellipsoid
    ax.plot_surface(x, y, z, alpha=0.2)
    ## Plot all satellite positions (at the first time step) in red
    all_positions = np.array([pos[depth] for pos in positions])  # Extract depth time step
    ax.scatter(all_positions[:, 0], all_positions[:, 1], all_positions[:, 2], color='red', label='All Satellites')

    # # Plot only visible satellites in green
    visible_positions = np.array([positions[i][depth] for i in range(len(visibility)) if visibility[i][depth] == 1])
    ax.scatter(visible_positions[:, 0], visible_positions[:, 1], visible_positions[:, 2], marker='s',s=100,color='green', label='Visible Satellites')

    # Plot user's position in a blue square
    ax.scatter(userxyz[0], userxyz[1], userxyz[2], color='blue', marker='s', s=100, label='User Position')

    # Add labels and legend
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.legend()
    ax.set_title(f"3D Satellite Positions {coordsys} at tstep {depth}")

    # Make the plot interactive
    plt.show()

def plot_satellite_positionsENU(positions, visibility, userxyz,coordsys,depth):
    """
    Plots satellite positions in 3D with a sphere representing the Earth.
    
    Parameters:
        positions (list): List of satellite positions in the form [[x, y, z], ...].
        visibility (list): List indicating visibility of satellites (1 for visible, 0 for not visible).
        userxyz (list): User's position [x, y, z].
        coordsys: (string) of ENU or ECEF
        depth (int): depth at which you want to plot, 0 for first time step
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all satellite positions (at the first time step) in red
    all_positions = np.array([pos[depth] for pos in positions])  # Extract depth time step
    ax.scatter(all_positions[:, 0], all_positions[:, 1], all_positions[:, 2], color='red', label='All Satellites')

    # # Plot only visible satellites in green
    # visible_positions = np.array([positions[i][depth] for i in range(len(visibility)) if visibility[i][depth] == 1])
    # ax.scatter(visible_positions[:, 0], visible_positions[:, 1], visible_positions[:, 2], marker='d',s=75,color='green', label='Visible Satellites')

    # Plot user's position in a blue square
    ax.scatter(userxyz[0], userxyz[1], userxyz[2], color='blue', marker='s', s=100, label='User Position')

    # Add labels and legend
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.legend()
    ax.set_title(f"3D Satellite Positions {coordsys} at tstep {depth}")

    # Make the plot interactive
    plt.show()

observerENU = np.dot(eceftoenu,observerxyz)
print(observerxyz)
print(observerENU)
#plot ENU
#plot_satellite_positionsENU(positions_ENU, visibility,observerENU,'ENU',0 )
#plot ECEF
plot_satellite_positionsECEF(positions, visibility,observerxyz,'ECEF',0 )