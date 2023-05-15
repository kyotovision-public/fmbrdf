# first row of reflectometry
python estimate_BRDF.py phenol 4
python estimate_BRDF_rendering.py phenol 5

# second row of reflectometry
python estimate_BRDF.py epoxyglass2 4
python estimate_BRDF_rendering.py epoxyglass2 2

# third row of reflectometry
python estimate_BRDF.py horse_reflectometry 5
python estimate_BRDF_rendering.py horse_reflectometry 11

# fourth row of reflectometry
python estimate_BRDF.py human 2
python estimate_BRDF_rendering.py human 0

