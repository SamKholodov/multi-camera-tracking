from core.reid.backbones.osnet import osnet_ibn_x1_0, osnet_x0_25

MODEL_FACTORY = {
    "osnet_x0_25": osnet_x0_25,
    "osnet_ibn_x1_0": osnet_ibn_x1_0,
}
