from core.reid.backbones.osnet import osnet_ibn_x1_0, osnet_x0_25, osnet_x1_0
from core.reid.backbones.vehicle_osnet import vehicle_osnet_x1_0

MODEL_FACTORY = {
    "osnet_x0_25": osnet_x0_25,
    "osnet_x1_0": osnet_x1_0,
    "osnet_ibn_x1_0": osnet_ibn_x1_0,
    "vehicle_osnet_x1_0": vehicle_osnet_x1_0,
}
