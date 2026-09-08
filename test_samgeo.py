from samgeo import SamGeo

sam = SamGeo(
    model_type="vit_h",
    checkpoint="sam_vit_h_4b8939.pth",
    automatic=True,
)
print("SAM-Geo initialized successfully.")