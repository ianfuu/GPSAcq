#GPS Acquisition MDP for AA228 Project

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
using GeometryBasics
using LinearAlgebra
using Combinatorics
using Random
import QuickPOMDPs: QuickPOMDP, QuickMDP
import POMDPTools: ImplicitDistribution,Deterministic,Uniform
using POMDPTools
using POMDPLinter
import POMDPLinter: show_requirements
using Dates
using D3Trees
using Colors
using JSON
using StatsBase


#Load in TLES
tles = read_tles_from_file("gp.txt")
# Initialize a List to store positions for each TLE
#tle_indices = 1:50:length(tles)  # Indices for every 50th entry
tle_indices=1:1:length(tles)
tle_count = length(tle_indices)  # Count of the selected TLEs
positions_by_tle = Vector{Vector{Vector{Float64}}}(undef,tle_count)

# Simulation parameters
Depth = 21  # steps in our simulation
start = 25  # min after epoch
End = 35  # min after epoch
step_size = (End - start) / (Depth - 1)

tle_count = min(100, length(tles))  # Limit to the available number of TLEs

# Loop through the selected TLEs (in TEME so will need to convert)
eop_IAU1980  = fetch_iers_eop()

#get direct cosine matrix (Rotation Matrix) to go from eci(TEME) to ECEF (ITRF)
for (i, idx) in enumerate(tle_indices)
    tle = tles[idx]
    sgp4d = sgp4_init(tle)  # Initialize the SGP4 propagator

    jul_date = tle_epoch(tle) #this is the julian day of the given TLE
    TEME_ITRF = r_eci_to_ecef(DCM, TEME(), ITRF(), jul_date, eop_IAU1980) #Gives rotation Matrix from TEME to ECEF
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

#get observer location in ECEF
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
lat = -50
lon = -90
observerxyz =  lla_to_ecef(lat,lon,16)/1000 #to get in km counts -50, -90, 35, 9, 10, 17)


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
    #get positions in m
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
function azel_from_pos(observerxyz::Vector{},positions_by_tle::Vector{Vector{Vector{Float64}}})
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
azel_ENU = azel_from_pos(observerxyz,positions_by_tle)
#calculate az el  in ENU but positions_by_tle is in ECEF

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
visibility = sat_visibility(azel_ENU,10)
save_to_file(visibility,"visibility.json")

function calculate_DOP(user_position::Vector{Float64},positions::Vector{Vector{Vector{Float64}}},s::Vector{Int64})
    #Calculate _DOP from the current configuration
    #s is the current state, a 5 tuple holding step and index of currently acquired satellites
    #d is current depth/time step
    d = s[1]
    state = s[2:end]
    # Extract the satellite positions at the current timestep `d` for satellites `s` as |s|x1 matrix
    satellite_positions = reshape([positions[sat][d] for sat in state], :, 1)
    # Calculate the unit vectors to each selected satellite in ENU
    unitvectors = [compute_los_enu(user_position, sat_pos) for sat_pos in satellite_positions]
    # Convert the list of unit vectors into an n x 3 matrix (each row is a unit vector)
    unit_vectors = hcat(unitvectors...)'

    dop_matrix = inv(unit_vectors' * unit_vectors)
    # The DOP can be approximated by the trace of the DOP matrix
    DOP = sqrt(tr(dop_matrix))
    return DOP
end

function DOP_to_Reward(DOP)
    #function to turn a given DOP into an explicit reward (reward small DOPS)
    if DOP <= 4
        return 100.0
    elseif DOP >= 15
        return -100.0
    else
        # Linear interpolation for DOP between 4 and 15
        return 100.0 - (DOP - 4) * (200.0 / 11.0)
    end
end

#To find weakest satellite contribution in current state
function weakest_satellite_contribution(observerxyz::Vector{Float64}, positions::Vector{Vector{Vector{Float64}}}, visibility::Vector{Vector{Bool}},s::Vector{Int64})
    # Get the DOP for the first combination
    d = s[1] + 1 #because weakest at next time step
    sats = s[2:end]

    #visible sats at next time step
    visible_sats = find_visible_satellites_at_depth(visibility, d)
    sats_nolongervis = setdiff(sats, visible_sats)
    #println(d, "    ", sats_nolongervis)
    if !isempty(sats_nolongervis)
        weakest_satellite = sats_nolongervis[1] #only take first entry, hopefully its not longer
        return weakest_satellite
    end
    
    #if all satelittes are in view in the next step, do weaskest swapping
    first_comb = first(Combinatorics.combinations(sats, 3))
    sample = [d; first_comb[:]][:]
    best_dop = calculate_DOP(observerxyz, positions, sample)
    weakest_satellite = setdiff(sats, first_comb)[1]  # Initially set the weakest satellite to the one not in the first combination
    
    # Iterate over all possible combinations of 3 satellites (out of 4)
    for comb in Combinatorics.combinations(sats, 3)
        sample = [d; comb[:]][:]
        DOP_value = calculate_DOP(observerxyz, positions, sample)
        
        if DOP_value < best_dop  # We are looking for the combination with the lowest DOP (best dop)
            best_dop = DOP_value
            # Find which satellite is not in this combination
            weakest_satellite = setdiff(sats, comb)[1]
        end
    end
    return weakest_satellite #index of weakest satellite
end

#get index of which satellitees are visible
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

function generate_possible_states(s::Vector{Int}, visible_sats::Vector{Int}, weak_sat::Int)
    # Initialize the result list to store the possible next states
    possible_states = []

    s = s[2:end]

    index = findall(x -> x == weak_sat, s)
    used_sats = filter(x->x≠weak_sat,s) #satellites used in other channels
    poss_sats = setdiff(visible_sats,used_sats) #see which sats we can switch to (including our the current one)
    # Iterate over all possible acquisitions
    for sat in poss_sats
        # Create a copy of the current state
        new_state = copy(s)
        # replace weakest sat with a possible sat in view
        new_state[index[1]] = sat
        # Add the new state to the list of possible states
        push!(possible_states, new_state)
    end
    # Rearrange so the state where `state[0]` is `weak_sat` comes first (so action 0 is nothing)
    possible_states = sort(possible_states, by = x -> (x[1] == weak_sat ? 0 : 1))
    
    return possible_states
end

function get_sp(s::Vector{Int}, visibility::Vector{Vector{Bool}}, observerxyz::Vector{Float64}, positions::Vector{Vector{Vector{Float64}}})
    # Assuming that our action is swap, this is how we get our new sp
    # want look at next time step and swap the worst for best in that time
    d = s[1]+1  # Get the depth of the current step +1

    # Figure out the weakest satellite from the visible ones
    visible_sats = find_visible_satellites_at_depth(visibility, d)  # At current depth
    weak_sat = weakest_satellite_contribution(observerxyz, positions,visibility, s)
    # Index of weakest satellite in s
    index = findall(x -> x == weak_sat, s)[end] #so if the time step is 5 and sat 5 in view, we replace the later

    # Remove satellites already in s[2:end] from visible_sats
    available_sats = filter(sat -> !in(sat, s[2:end]), visible_sats)
    # Find the sat with the best contribution
    best_dop = 100  # Initialize variable to store best DOP
    best_config = copy(s)  # Start with the current configuration as the best
    # Now make d get one larger (one step in the future)
    best_config[1] += 1

    for sat in available_sats
        # Create a new configuration by replacing the weakest satellite
        new_config = copy(s)
        new_config[1]+=1
        new_config[index] = sat
        # Calculate the DOP for this new configuration
        dop = calculate_DOP(observerxyz, positions, new_config)

        # Update the best configuration if the new one has a lower DOP
        if dop < best_dop
            best_dop = dop
            best_config = new_config
        end
    end

    # Now enforce that states have to be in ascending order to reduce state space
    sp = [best_config[1]; sort(best_config[2:end])]
    return sp
end


function pos_of_state(positions::Vector{Vector{Vector{Float64}}},s::Vector{Int64})
    #extract coordinates of the state from current states
    d=s[1]
    state = s[2:end]
    # Extract the satellite positions at the current timestep `d` for satellites `s` as |s|x1 matrix
    satellite_positions = reshape(
        [Vector{Float64}([positions[sat][d][1], positions[sat][d][2], positions[sat][d][3]]) for sat in state],
        :, 1)
    
    return satellite_positions
end


function create_statespaceold(visibility::Vector{Vector{Bool}}, depth::Int)
    #function takes in our visibility matrix from before and extracts which satellites are viewable at each step 
    #to create out full state space
    #now enforcing that states must be in ascending order 
    d_of_search = depth # how many steps into the sim we want to create states for
    possible_states = Vector{Vector{Int}}()  # To store all possible states
    step = d_of_search # step we want

    # Find the satellites visible at this step
    visible_sats = [sat for sat in 1:length(visibility) if visibility[sat][step]]
    
    # Generate all combinations of 4 satellites (if there are at least 4 visible satellites)
    if length(visible_sats) >= 4
        for combination in combinations(visible_sats, 4)
            # Sort satellites in ascending order
            sorted_combination = sort(collect(combination))
            # Create the state vector [step, sat1, sat2, sat3, sat4]
            state = [step; sorted_combination]
            push!(possible_states, state)
        end
    end

    return possible_states
end

function create_statespace(visibility::Vector{Vector{Bool}}, depth::Int)
    # Function to create the state space based on satellite visibility over `depth` steps
    #function takes in our visibility matrix from before and our state space
    #to deal with issues where we you could choose stay and a satellite goes out of view so its not in old StateSpace
    #we now just take all satellites viewable at any time, and compute their state combos (at all time steps) even though many are impossible
    #should help with our bounds errors of states not existing
    #now enforcing that states must be in ascending order 
    # Step 1: Find all satellites that come into view at any point over the given depth
    visible_sats = [sat for sat in 1:length(visibility) if any(visibility[sat][1:depth])]

    # Step 2: Generate all 4-satellite combinations from the visible satellites
    possible_combinations = Vector{Vector{Int}}()
    if length(visible_sats) >= 4
        for combination in combinations(visible_sats, 4)
            push!(possible_combinations, sort(collect(combination)))  # Sort for ascending order
        end
    end

    # Step 3: Generate states for all steps and the possible satellite combinations
    possible_states = Vector{Vector{Int}}()
    for step in 1:depth
        for combination in possible_combinations
            state = [step; combination]  # Create the state vector [step, sat1, sat2, sat3, sat4]
            push!(possible_states, state)
        end
    end
    
    return possible_states
end


all_s0s = create_statespaceold(visibility,1)


𝒜 = [:Nothing, :Swap]  # Create an array of Action enum values
𝒮 = unique(create_statespace(visibility, 13)) #
#if current config, 53 come into view over 13 steps, ~292825 states per step, ~3806725 total state space size for 13 steps
#this 𝒮 will blow up the more satellites that are ever visible so tle propagation is important

mdp = QuickMDP(
    initialstate = Uniform(all_s0s), #allow all initial states   Uniform([s0]),#
    actions = 𝒜,
    actiontype = Symbol,
    states = 𝒮,
    statetype = Vector{Int64},
    discount = 0.95,
    isterminal = s -> s[1] >= 13, #terminal state when we reach last time step of interest

    transition = function (s, a)        
        if a == :Nothing
            sp = copy(s)
            sp[1] += 1
            return Uniform([sp])
        elseif a == :Swap
            #then we swap weakest for strongest with 100% certainty
            sp = get_sp(s,visibility,observerxyz,positions_by_tle) #global definition of these so can just be function of s,a
            return Uniform([sp])
        else
            error("Invalid action: $a")
        end
    end,

    reward = function (s, a)
        if a == :Nothing #do nothing case
            cost = 0
        elseif a == :Swap
            cost = -10
        end
        #penalize ones going out of view
        visible_sats = find_visible_satellites_at_depth(visibility, s[1])
        sats_nolongervis = setdiff(s[2:end], visible_sats)
        cost2 = 0
        if !isempty(sats_nolongervis)
            num_not_in_view = length(sats_nolongervis)
            cost2 = -100*num_not_in_view
        end

        #now DOP contribution
        scale_factor = 1 #scales the DOP reward contribution
        DOP  = calculate_DOP(observerxyz,positions_by_tle,s)
        return cost + cost2 + scale_factor*DOP_to_Reward(DOP)
        
    end
)

#Now we can run our methods
solver = MCTSSolver(max_time=20.0, depth=10, exploration_constant=4.0,enable_tree_vis=true) #depth = depth-1 bc tree search starts at 0

planner = solve(solver, mdp,) #gives use the policy 

#example to visualize a tree
# seed = 42
# Random.seed!(seed)
# visible_sats = find_visible_satellites_at_depth(visibility,1) #at depth 1
# random_sats = Vector{Int}(sample(visible_sats, 4, replace=false))
# rand_s0 = [1; sort(random_sats[:])][:]
# a, info = action_info(planner, rand_s0)
# #get action values
# tree = planner.tree
# t = D3Tree(info[:tree], init_expand=2)
# inchrome(t)



#Now doing it with best state
#want to start with whatever state has the highest initial DOP and take actions from there
function find_best_s0(all_s0s, observerxyz, positions_by_tle)
    bestDOP = 1000.0  # A very large initial value for DOP
    best_s0 = [1, 1, 1, 1, 1]  # Example initial state
    # Loop through all possible states and calculate the DOP for each
    for state in all_s0s
        dop = calculate_DOP(observerxyz, positions_by_tle, state)
        # If the current DOP is smaller than the best one found so far, update the best state
        if dop < bestDOP
            bestDOP = dop
            best_s0 = state
        end
    end
    # Return the best state and its corresponding DOP value
    return best_s0, bestDOP
end

#Now we are ready to generate a policy
function generate_policy(all_s0s, observerxyz, positions_by_tle, visibility, planner,random::Bool)
    #random says if we want a random s0 or the best one
    #get initial state
    if random == false
        # Step 1: Find the initial best state (best_s0) and corresponding DOP
        best_s0, bestDOP = find_best_s0(all_s0s, observerxyz, positions_by_tle)
    elseif random == true
        #generate random initial state from all all_s0s
        seed = 42
        Random.seed!(seed)
        visible_sats = find_visible_satellites_at_depth(visibility,1) #at depth 1
        random_sats = Vector{Int}(sample(visible_sats, 4, replace=false))
        best_s0 = [1; sort(random_sats[:])][:]
    end

    # Initialize the state list and action list, DOP sum list
    states = [best_s0]
    initialdop = calculate_DOP(observerxyz,positions_by_tle,best_s0)
    dops = [initialdop]
    actions = []
    Q_vals = []
    # Simulate the process for 10 time steps
    for step in 1:10
        # Take action using the current state
        a, info = action_info(planner, best_s0)
        tree = planner.tree
        t = D3Tree(info[:tree], init_expand=2)
        #inchrome(t)
        # Store the action
        push!(actions, a)
        #store_bestQ
        push!(Q_vals, info.best_Q)
        # Update state based on the action
        if a == :Nothing
            # If the action is :Nothing, just increment the depth and keep the same satellites
            sp = copy(best_s0)
            sp[1] += 1  # Increment the depth to the next time step
        elseif a == :Swap
            # If the action is :Swap, swap the weakest satellite with the strongest
            sp = get_sp(best_s0, visibility, observerxyz, positions_by_tle)
        end
        # Store the new state
        push!(states, sp)
        # Update the current state for the next iteration
        best_s0 = sp
        #calcualte dop of new state
        newdop = calculate_DOP(observerxyz,positions_by_tle,best_s0)
        push!(dops, newdop)
    end
    # Return the policy as a tuple of states and actions
    return states, actions, dops, Q_vals
end

#Now we are ready to generate a policy
function generate_set_policy(all_s0s, observerxyz, positions_by_tle, visibility, planner,random::Bool)
    #random says if we want a random s0 or the best one
    #get initial state
    if random == false
        # Step 1: Find the initial best state (best_s0) and corresponding DOP
        best_s0, bestDOP = find_best_s0(all_s0s, observerxyz, positions_by_tle)
    elseif random == true
        #generate random initial state from all all_s0s
        seed = 42
        Random.seed!(seed)
        visible_sats = find_visible_satellites_at_depth(visibility,1) #at depth 1
        random_sats = Vector{Int}(sample(visible_sats, 4, replace=false))
        best_s0 = [1; sort(random_sats[:])][:]
    end

    # Initialize the state list and action list, DOP sum list
    states = [best_s0]
    initialdop = calculate_DOP(observerxyz,positions_by_tle,best_s0)
    dops = [initialdop]
    actions = []
    Q_vals = []
    # Simulate the process for 10 time steps
    for step in 1:10
        # Take action using the current state
        a, info = action_info(planner, best_s0)
        tree = planner.tree
        t = D3Tree(info[:tree], init_expand=2)
        #inchrome(t)
        a = :Swap
        # Store the action
        push!(actions, a)
        push!(Q_vals,0)
        # Update state based on the action
        if a == :Nothing
            # If the action is :Nothing, just increment the depth and keep the same satellites
            sp = copy(best_s0)
            sp[1] += 1  # Increment the depth to the next time step
        elseif a == :Swap
            # If the action is :Swap, swap the weakest satellite with the strongest
            sp = get_sp(best_s0, visibility, observerxyz, positions_by_tle)
        end
        # Store the new state
        push!(states, sp)
        # Update the current state for the next iteration
        best_s0 = sp
        #calcualte dop of new state
        newdop = calculate_DOP(observerxyz,positions_by_tle,best_s0)
        push!(dops, newdop)
    end
    # Return the policy as a tuple of states and actions
    return states, actions, dops, Q_vals
end

#policy assuming we start in best DOP initial state
println("Policy for Best s0")
policy = generate_policy(all_s0s,observerxyz,positions_by_tle,visibility,planner,false)
#print out policy
for step in 1:length(policy[2])
    println("State at Step",step," is: ", policy[1][step], " with DOP: ", policy[3][step])
    println("Action at Step",step," is: ", policy[2][step], " with Q_val: ", policy[4][step])
end
println("Final State: ",policy[1][end]," with DOP: ", policy[3][end] )
println("Sum of DOP: ",sum(policy[3]) )
println("Sum of Q_val: ",sum(policy[4]) )

#policy assuming we start in random s0 from possible ones
println("")
println("Policy for Random s0")
policy = generate_policy(all_s0s,observerxyz,positions_by_tle,visibility,planner,true)
#print out policy
for step in 1:length(policy[2])
    println("State at Step",step," is: ", policy[1][step], " with DOP: ", policy[3][step])
    println("Action at Step",step," is: ", policy[2][step], " with Q_val: ", policy[4][step])
end
println("Final State: ",policy[1][end]," with DOP: ", policy[3][end] )
println("Sum of DOP: ",sum(policy[3]) )
println("Sum of Q_val: ",sum(policy[4]) )


#policy assuming we start in best DOP initial state, and always swap
println("Policy for Best s0, Always Swap")
policy = generate_set_policy(all_s0s,observerxyz,positions_by_tle,visibility,planner,false)
#print out policy
for step in 1:length(policy[2])
    println("State at Step",step," is: ", policy[1][step], " with DOP: ", policy[3][step])
    println("Action at Step",step," is: ", policy[2][step])
end
println("Final State: ",policy[1][end]," with DOP: ", policy[3][end] )
println("Sum of DOP: ",sum(policy[3]) )
println("Sum of Q_val: 5529.19 " )
#need a metric to compare with a 'random policy'

#plotting:
using Plots
function circleShape(h,k,r)
    ang = LinRange(0,2*pi,500)
    h .+ r*sin.(ang), k .+ r*cos.(ang)
end

function plotPolar(azel::Vector{Vector{Vector{Float64}}}, visibility::Vector{Vector{Bool}}, s::Vector)
    
    #get visible satellites at depth
    depth = s[1]
    visibility_at_depth = find_visible_satellites_at_depth(visibility,depth)
    #all of them
    azimuths = [azel[idx][depth][1] for idx in visibility_at_depth]  # Azimuth angles in degrees
    elevations = [azel[idx][depth][2] for idx in visibility_at_depth]  # Elevation distances in degrees
    
    #selected ones
    states = s[2:end]
    azimuths_sel = [azel[sat][depth][1] for sat in states]
    elv_sel = [azel[sat][depth][2] for sat in states]

    sz = length(azimuths)
    xs = fill(NaN,sz,1)
    ys = fill(NaN,sz,1)
    bestxs = fill(NaN,4,1)
    bestys = fill(NaN,4,1)

    for i in 1:sz
        local az, el, r, theta
        az = azimuths[i];
        el = elevations[i];
        r = 90 - el
        theta = az*(pi/180)
        xs[i,1] = r*sin(theta)
        ys[i,1] = r*cos(theta)
    end

    for i in 1:4
        local az, el, r, theta
        az = azimuths_sel[i];
        el = elv_sel[i];
        r = 90 - el
        theta = az*(pi/180)
        bestxs[i,1] = r*sin(theta)
        bestys[i,1] = r*cos(theta)
    end

    # Create a scatter plot
    scatter(xs, ys, aspect_ratio=1, m = 2, axis = false,dpi =500)
    scatter!(bestxs, bestys, aspect_ratio=1, m = 4, axis = false,dpi =500)

    labels = []
    for (i, idx) in enumerate(visibility_at_depth)
        push!(labels,string(idx))
    end
    annotate!.(xs,ys.+5,text.(labels,6))

    plot!(circleShape(0,0,80), seriestype = [:shape,], lw = 0.5, linecolor = :blue, 
    legend = false, fillalpha = 0, aspect_ratio = 1, axis = false,dpi =500)# this is the elevation cutoff
    radii= [30,60,90]
    for r in radii
        plot!(circleShape(0, 0, r), seriestype=[:shape], lw=0.5, linecolor=:black, 
            legend=false, fillalpha=0, aspect_ratio=1, axis=false,dpi =500)
        annotate!(6, -r + 5, text(string(r, "°"), 8, :black))  # Label the radius
    end

    # Add horizontal and vertical lines
    max_radius = 90  # Maximum radius of the plot

    # Add horizontal, vertical, and diagonal lines
    max_radius = 90  # Maximum radius of the plot
    angles = 0:30:330  # Angles for diagonal lines
    for angle in angles
        theta = angle * (pi / 180)
        x = max_radius * sin(theta)
        y = max_radius * cos(theta)
        plot!([0, x], [0, y], lw=0.5, linecolor=:black, legend=false,dpi =500)  # Diagonal line

        # Add angle labels slightly outside the circle
        label_x = (max_radius + 5) * sin(theta)
        label_y = (max_radius + 5) * cos(theta)
        if angle == 0 || angle == 180
            annotate!(label_x, label_y, text(string(angle, "°"), 8, :black, halign=:center))
        elseif angle<180
            annotate!(label_x+3, label_y, text(string(angle, "°"), 8, :black, halign=:center))
        elseif angle>180
            annotate!(label_x-3, label_y, text(string(angle, "°"), 8, :black, halign=:center))
        end
    end

    title!("Satellite Sky at Time Step:  $depth")
    filename = string("skp",depth,".png")
    savefig(filename)
    
end

#now plot it
best_s0, bestDOP = find_best_s0(all_s0s, observerxyz, positions_by_tle)

# for step in 1:length(policy[1])
#     plotPolar(azel_ENU, visibility, policy[1][step])
# end
#plotPolar(azel_ENU, visibility, rand_s0)


