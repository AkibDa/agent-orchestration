from global_land_mask import globe
# Kochi is on land (coastal city)
print("Kochi (9.93, 76.26):", globe.is_land(9.93, 76.26))
# Offshore 
print("Offshore (5.0, 70.0):", globe.is_land(5.0, 70.0))
