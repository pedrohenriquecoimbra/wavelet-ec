import numpy as np

def vapour_deficit_pressure(T, RH):
    if np.nanquantile(T, 0.1) < 100 and np.nanquantile(T, 0.9) < 100:
        T = T + 274.15

    # Saturation Vapor Pressure (es)
    #es = 0.6108 * np.exp(17.27 * T / (T + 237.3))
    es = (T **(-8.2)) * (2.7182)**(77.345 + 0.0057*T-7235*(T**(-1)))

    # Actual Vapor Pressure (ea)
    ea = es * RH / 100

    # Vapor Pressure Deficit (Pa)
    return (es - ea)# * 10**(-3)

def calculate_mean_wind_direction(wind_speeds, wind_directions):
    """
    Calculates the mean wind direction given wind speeds and wind directions.

    Args:
        wind_speeds (np.ndarray): Array of wind speeds.
        wind_directions (np.ndarray): Array of wind directions in degrees (0° to 360°).

    Returns:
        float: Mean wind direction in degrees (0° to 360°).
    """
    # Convert wind directions to radians
    wind_dir_rad = np.radians(wind_directions)

    # Compute eastward and northward components
    V_east = np.nanmean(wind_speeds * np.sin(wind_dir_rad))
    V_north = np.nanmean(wind_speeds * np.cos(wind_dir_rad))

    # Calculate the mean wind direction
    mean_WD = np.arctan2(V_east, V_north) * (180 / np.pi)

    # Ensure the result is in the range [0°, 360°]
    mean_WD = (360 + mean_WD) % 360

    return mean_WD