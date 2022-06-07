
from tomo_encoders import Patches

vol_shape = (1000, 2100, 2100)
patch_size = (64, 64, 64)

p = Patches(vol_shape, initialize_by = 'regular-grid', patch_size = patch_size)
p = p.filter_by_cylindrical_mask(mask_ratio = 0.9, height_ratio = 1.0) # tomo volume contains nothing outside the cylindrical mask

p_1 = p.perturb(5)


print(p['points'][:5])
print(p_1['points'][:5])

print("Are all perturbed points within the volume?", p_1._check_valid_points() is None)
