# left fitting result of object with rough mesogeometry
python save_intensities_plane.py sponge_sandpaper -0.05235987755 10 8 0.014 0.011 0 3 35 32 -s 7 -e 7
python mesogeometry_fitting.py sponge_sandpaper

# right fitting result of object with rough mesogeometry
python save_intensities_plane.py texture_sand -0.05235987755 10 8 0.017 0.013 0 3 35 32 -s 5 -e 7
python mesogeometry_fitting.py texture_sand

