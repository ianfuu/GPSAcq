using SatelliteToolbox
using Plots
using StaticArrays
using SatelliteToolboxSgp4
using SatelliteToolboxTle
using Meshes
using GeometryBasics
using LinearAlgebra
using Combinatorics
using POMDPs
using DataFrames
using CSV
using LinearAlgebra
using StaticArrays
using MCTS
using SatelliteToolbox
using Plots
using StaticArrays
using SatelliteToolboxSgp4
using SatelliteToolboxTle
using Meshes
using GeometryBasics
using LinearAlgebra
using Combinatorics
using Random
import QuickPOMDPs: QuickPOMDP, QuickMDP
import POMDPTools: ImplicitDistribution,Deterministic,Uniform
using POMDPLinter
import POMDPLinter: show_requirements
using Dates

#First run sgp4 propagation on the satellites and get their positions/visibility
#user_xyz::Vector{Float64},
#satellite_positions::Dict{Int, Vector{Vector{Float64}}},
#visibility::Dict{Int, Vector{Bool}}
#these are the data types. 

#Load in TLES
tles = read_tles_from_file("gp.txt")
#tles
# Initialize a dictionary to store positions for each TLE
tle_indices = 1:1:length(tles)  # Indices for every 50th entry
# Random.seed!(1337)
# tle_indices = sample(1:length(tles), min(150, length(tles)), replace=false) #random indices
tle_count = length(tle_indices)  # Count of the selected TLEs
#tle_count = min(100, length(tles))  # Limit to the available number of TLEs
positions_by_tle = Vector{Vector{Vector{Float64}}}(undef,tle_count)

# Simulation parameters
Depth = 21  # steps in our simulation
start = 20  # min after epoch
End = 40  # min after epoch
step_size = (End - start) / (Depth - 1)

# Loop through the selected TLEs (in TEME so will need to convert)
#for converting from teme to ITRF
eop_IAU1980  = fetch_iers_eop()

#get direct cosine matrix to go from eci(TEME) to ECEF (ITRF)
for (i, idx) in enumerate(tle_indices)
    tle = tles[idx]
    sgp4d = sgp4_init(tle)  # Initialize the SGP4 propagator

    jul_date = tle_epoch(tle) #this is the julian day of the given TLE
    TEME_ITRF = r_eci_to_ecef(DCM, TEME(), ITRF(), jul_date, eop_IAU1980) #matrix to turn TEME to ITRF for given tle
    # Store positions for the current TLE
    positions = Vector{Vector{Float64}}()  # Temporary vector to store this TLE's positions
    for t in start:step_size:End
        r_teme, _ = sgp4!(sgp4d, t)  # Propagate
        #transform to ECEF
        r_ecef = TEME_ITRF * r_teme
        push!(positions, r_ecef)  # Store position
    end
    positions_by_tle[i] = positions  # Assign to the vector
end


function save_to_file(data, filename::String)
    open(filename, "w") do file
        write(file, JSON.json(data))
    end
end
save_to_file(positions_by_tle,"positions.json")

function get_color(index)
    colors = palette(:tab10)  # Use the tab10 color palette
    return colors[mod1(index, length(colors))]
end

function lla_to_ecef(phi, λ, h)
    # Convert degrees to radians
    phi = phi * (π / 180)
    λ = λ * (π / 180)
    
    # Earth parameters
    Re = 6378.137 * 1000  # Radius of the Earth in meters
    f = 1 / 298.257223563  # Flattening parameter of the ellipsoid
    eSqrd = 2 * f - f^2    # Square of eccentricity of the ellipsoid
    
    # Curvature of Earth at some location
    Ce = Re / sqrt(1 - eSqrd * (sin(phi))^2)
    
    # ECEF coordinates
    x = (Ce + h) * cos(phi) * cos(λ)
    y = (Ce + h) * cos(phi) * sin(λ)
    z = ((Ce * (1 - eSqrd) + h) * sin(phi))
    
    return [x, y, z]
end
lat =-45
lon =0
observerxyz =  lla_to_ecef(lat,lon,16)/1000 #to get in km counts (37, -25, 63, 40, 56, 56) #curret (52, -100, 68, 16, 42, 30)

function unitVec(v::Vector{})
    # Calculate the magnitude (or norm) of the vector
    mag = sqrt(sum(v .^ 2))  # This is the Euclidean norm (L2 norm)
    
    # Check if the magnitude is not zero to avoid division by zero
    if mag == 0
        throw(ArgumentError("Cannot normalize a zero vector"))
    end
    
    # Normalize the vector by dividing each component by the magnitude
    return v / mag
end
#Computes ECEF to ENU Rotation Matrix
function ecef_to_enu(lat_deg, lon_deg)
    φ = deg2rad(lat_deg)  # radians
    λ = deg2rad(lon_deg)  # radians
    
    C = [
        -sin(λ)        cos(λ)         0;
        -sin(φ) * cos(λ)  -sin(φ) * sin(λ)  cos(φ);
        cos(φ) * cos(λ)   cos(φ) * sin(λ)  sin(φ)
    ]
    
    return C
end

# Compute LOS unit vector from user to sat in ENU
function compute_los_enu(user_ecef, sat_ecef)
    # Convert ECEF to LLA
    #user ecef given in km so make m
    user_lla = ecef_to_lla(user_ecef[1]*1000, user_ecef[2]*1000, user_ecef[3]*1000)  # LLA in [m]
    lat = user_lla[1]
    lon = user_lla[2]
    
    # Compute the transformation matrix from ECEF to ENU
    C_ecef_to_enu = ecef_to_enu(lat, lon)
    
    # Calculate the LOS vector in ECEF
    los_ecef = sat_ecef .- user_ecef  # Element-wise subtraction
    
    # Normalize the LOS vector to get unit vector
    unit_los_ecef = unitVec(los_ecef)  # Assuming unitvec returns the magnitude and unit vector
    
    # Compute LOS in ENU by applying the transformation matrix
    los_enu = C_ecef_to_enu * unit_los_ecef  # Transpose the unit vector to match dimensions
    
    return los_enu
end

# Compute Az,El given user and satelitte posiions in ECEF
function compute_az_el_range(user_ecef, sat_ecef)
    los_enu = compute_los_enu(user_ecef, sat_ecef)  # unit vector pointing from user to satellite
    e, n, u = los_enu[1], los_enu[2], los_enu[3]
    
    az = atan(e, n)  # atan2(y, x) in Julia
    az = rad2deg(az)  # convert to degrees
    
    el = atan(u, sqrt(e^2 + n^2))  # elevation angle
    el = rad2deg(el)  # convert to degrees
    
    #range = norm(sat_ecef - user_ecef)  # compute the range
    
    return [az, el] #,range
end

# Compute LLA from ECEF
function ecef_to_lla(x, y, z)
    rho = sqrt(x^2 + y^2)
    r = sqrt(x^2 + y^2 + z^2)
    oldGuess = 0.0
    newGuess = asin(z / r)
    Re = 6378.137 * 1000  # meters
    f = 1 / 298.257223563  # Flattening parameter
    eSqrd = 2 * f - f^2   # Eccentricity squared

    # Find the latitude with tolerance
    while abs(oldGuess - newGuess) > 1e-8
        oldGuess = newGuess
        Ce = Re / sqrt(1 - eSqrd * (sin(oldGuess)^2))  # Curvature of Earth at location
        newGuess = atan((z + Ce * eSqrd * sin(oldGuess)) / rho)
    end

    # Define lat, lon, and height values
    phi = newGuess
    lat = rad2deg(phi)
    lambda_ = atan(y, x)
    lon = rad2deg(lambda_)
    Ce = Re / sqrt(1 - eSqrd * (sin(phi)^2))
    h = (rho / cos(phi)) - Ce
    return [lat, lon, h]
end

# Initialize azel ENU 
function azel_from_pos(observerxyz,positions_by_tle)
    azel_ENU = Vector{Vector{Vector{Float64}}}(undef, length(positions_by_tle))
    #calculate az el  in ENU but positions_by_tle is in ECEF
    # Loop through the positions_by_tle vector to calculate az/el
    for (idx, positions) in enumerate(positions_by_tle)
        azel_ENU[idx] = Vector{Vector{Float64}}(undef, length(positions))
        # For each position in the current satellite's trajectory
        for (i, pos) in enumerate(positions)
            # Call calculate_azel and get azimuth and elevation
            azel = compute_az_el_range(observerxyz, pos) #observerxyz and pos both in ECEF
            
            # Store the azimuth and elevation
            azel_ENU[idx][i] = azel
        end
    end
    return azel_ENU
end

#New Visibility
function sat_visibility(azel::Vector{Vector{Vector{Float64}}},tolerance)
    #tolerance is the angle above our horizon that we classify "visible" as
    # Output matrix for visibility, meant to see which satellites are visible at each time step
    visible_satellites = Vector{Vector{Bool}}(undef, length(azel)) 

    # Loop through each satellite and time step
    for (satellite_i, positions) in enumerate(azel)
        visible_satellites[satellite_i] = Bool[]  # Initialize visibility for this satellite
        for satellite_azel in positions
            # Check if the satellite is above tolerance° from the horizon
            is_visible = satellite_azel[2] >= tolerance  #if elevation greater than tolerance then its in view
            push!(visible_satellites[satellite_i], is_visible)  # Append visibility to the satellite
        end
    end
    return visible_satellites
end

#Calculate the visibility Vector
azel_ENU = azel_from_pos(observerxyz,positions_by_tle)
visibility = sat_visibility(azel_ENU,10)
save_to_file(visibility,"visibility.json")

visible_sats = [sat for sat in 1:length(visibility) if any(visibility[sat][1:13])]
sats_at1 = length(find_visible_satellites_at_depth(visibility, 1))
sats_at5 = length(find_visible_satellites_at_depth(visibility, 5))
sats_at10 = length(find_visible_satellites_at_depth(visibility, 10))
println((lat, lon, length(visible_sats),sats_at1,sats_at5,sats_at10))

function find_visible_satellites_at_depth(visibility::Vector{Vector{Bool}}, d::Int)
    # Find which columns (keys) in the visibility dictionary have a 1 at the d-th entry
    visible_satellites = []

    # Iterate through the visibility dictionary
    for (sat_id, visibility_vector) in enumerate(visibility)
        # Check if the d-th entry is 1
        if visibility_vector[d] == 1
            push!(visible_satellites, sat_id)
        end
    end

    return Vector{Int}(visible_satellites) #will be a vector that holds [30,32,3,1,4] which are keys of satellites in dict
end

# function find_best_location(positions_by_tle)
#     max_sats = 0
#     best_lat_lon = (0, 0)

#     # Loop over latitude and longitude combinations
#     for lati in 0:1:45
#         for long in 0:-5:-130
#             # Calculate observer location in ECEF
#             observer_loc = geodetic_to_ecef(lati, long, 16.0)  # Altitude 16.0 meters
#             observerxyz = collect(observer_loc) / 1000  # Convert to km

#             # Compute satellite visibility
#             visibility = satellite_visibility(observerxyz, positions_by_tle)

#             # Sum total visibility over all time steps
#             total_sats = 0
            
#             total_sats = [sat for sat in 1:length(visibility) if any(visibility[sat][1:15])] 
#             #how many sats ever become visible in first 15 steps

#             # Check if this combination has the maximum visibility
#             if total_sats > max_sats
#                 max_sats = total_sats
#                 best_lat_lon = (lati, long)
#             end
#         end
#     end

#     # Return the best latitude and longitude combination and maximum visibility
#     return best_lat_lon, max_sats
# end
function find_best_locations(positions_by_tle)
    results = []
    # Loop over latitude and longitude combinations
    for lati in 0:-5:-90
        for long in 90:-10:-90
            # Calculate observer location in ECEF
            observerxyz =  lla_to_ecef(lati,long,16)/1000 # Convert to km
            azel_ENU = azel_from_pos(observerxyz,positions_by_tle)

            # Compute satellite visibility
            visibility = sat_visibility(azel_ENU,10)

            # Count satellites ever visible in the first 10 steps
            total_sats = length([sat for sat in 1:length(visibility) if any(visibility[sat][1:13])])
            sats_at1 = length(find_visible_satellites_at_depth(visibility, 1))
            sats_at5 = length(find_visible_satellites_at_depth(visibility, 5))
            sats_at10 = length(find_visible_satellites_at_depth(visibility, 10))

            # Store the result
            push!(results, (lati, long, total_sats,sats_at1,sats_at5,sats_at10))
        end
    end
    # Sort results by max_sats in descending order
    sorted_results = sort(results, by=x->x[3], rev=true)
    # Return the sorted list
    return sorted_results
end

# Call the function and print results
lat_loncombos = find_best_locations(positions_by_tle)
# println("Best location: latitude = ", best_lat_lon[1], ", longitude = ", best_lat_lon[2])
# println("Maximum total visibility: ", max_sats)

# #plotting
# #Plot the first vector (red dot)
# plt = scatter([observerxyz[1]], [observerxyz[2]], [observerxyz[3]], color=:blue, marker=:circle, label="observerxyz")

# # Plot positions for each time step with different colors
# # for (i, time_set) in enumerate(positions_by_tle[1:30])
# #     color = get_color(i)  # Generate a unique color for each time set
# #     for pos in time_set
# #         scatter!([pos[1]], [pos[2]], [pos[3]], color=color, marker=:square, label="", legend=false)
# #     end
# # end
# visat0 = find_visible_satellites_at_depth(visibility, 1)
# # for (i,idx) in enumerate(visat0)
# #     color = get_color(i)  # Generate a unique color for each time set
# #     scatter!([positions_by_tle[idx][1][1]], [positions_by_tle[idx][1][2]], [positions_by_tle[idx][1][2]], color="green", marker=:square, label="", legend=false)
# # end
# for i in 1:length(positions_by_tle)
#     print([positions_by_tle[i][1][1]], [positions_by_tle[i][1][2]], [positions_by_tle[i][1][3]])
#     scatter!([positions_by_tle[i][1][1]], [positions_by_tle[i][1][2]], [positions_by_tle[i][1][3]], color="red", marker=:square, label="", legend=false)
# end
# # # Show the plot
# display(plt)