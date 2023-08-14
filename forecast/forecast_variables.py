# Copyright 2024 Crown in Right of Canada
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Define dataset variables that need special treatment
# transcribed from graphcast.graphcast
ATMO_VARIABLES = (
    "potential_vorticity",
    "specific_rain_water_content",
    "specific_snow_water_content",
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "vertical_velocity",
    "vorticity",
    "divergence",
    "relative_humidity",
    "ozone_mass_mixing_ratio",
    "specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content",
    "fraction_of_cloud_cover",
)
FIXED_VARIABLES = ['land_sea_mask','geopotential_at_surface'] # Variables that do not change within a batch
DERIVED_VARIABLES = ['year_progress_sin','year_progress_cos','day_progress_sin','day_progress_cos','toa_incident_solar_radiation'] # Variables that we can add ourselves if necessary
# Input variables that need to be treated as vectors during regridding
WIND_VARIABLES_U = ['u_component_of_wind','10m_u_component_of_wind']
WIND_VARIABLES_V = ['v_component_of_wind','10m_v_component_of_wind']
