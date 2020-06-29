import pandas as pd
import numpy as np
import fire


def calc_px_per_deg(
        x_pixels = 1280,  # px
        y_pixels = 800,  # px
        x_mm = 5,
        y_mm = 3.1,
        mouse_fov = 120,  # degrees
        mouse_retinal_arc = 4.9  # mm
        ):
    # calc
    um_per_deg = mouse_retinal_arc / mouse_fov * 1000
    x_um = x_mm*1000
    y_um = y_mm*1000
    x_pixels_per_um = x_pixels/x_um
    y_pixels_per_um = y_pixels/y_um

    x_pixels_per_deg = x_pixels_per_um * um_per_deg
    y_pixels_per_deg = y_pixels_per_um * um_per_deg
    print(f"x pixels per degree: {x_pixels_per_deg}")
    print(f"y pixels per degree: {y_pixels_per_deg}")

    logMARs = np.arange(0,3,0.1)
    # divide by 60 to convert minutes to degree
    x = np.array(list(map(lambda logmar: 10**logmar / 60 * x_pixels_per_deg, logMARs))).reshape(-1,1)
    y = np.array(list(map(lambda logmar: 10**logmar / 60 * y_pixels_per_deg, logMARs))).reshape(-1,1)
    data = np.hstack([x,y])

    table = pd.DataFrame(data=data, columns=["x (px)", "y (px)"], index=logMARs)
    table.index.name = 'logMAR'
    print(table)
    
if __name__ == '__main__':
    fire.Fire(calc_px_per_deg)