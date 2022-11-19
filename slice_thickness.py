from videos import Video
import matplotlib.pyplot as plt
import numpy as np


def plotter(x, y):
    plt.plot(x, y)
    plt.xlabel("Depth/cm")
    plt.ylabel("Slice thickness/cm")
    # plt.ylim(0.2, 1.6)


def main():
    viddata = {"filename": "vid10.mp4", "start_deep": True,
               "total_depth_cm": 12, "roi": [632, 111, 232, 552],
               "total_depth_pixels": 552}
    vid = Video(viddata)
    bkgd = vid.get_bkgd()
    widths, depths = vid.get_slice_thickness_data(25)
    """
    widths = []
    depths = []
    width_indices = np.arange(0, vid.frame_count, 25)
    for index in width_indices:
        width, depth_cm = vid.analyseFrame(index, subtract_bkgd=True)
        if width != 0:
            widths.append(width)
            depths.append(depth_cm)

    width_diffs = np.sqrt(np.diff(widths)**2)
    threshold = max(widths) / 5
    bad_indices = []
    for i, diff in enumerate(width_diffs):
        if diff >= threshold:
            bad_indices.append(i)
    depths = np.array(depths)
    widths = np.array(widths)
    mask = np.ones(len(widths), dtype=bool)
    mask[bad_indices] = False
    depths = depths[mask]
    widths = widths[mask]

    depth_diffs = np.diff(depths)
    bads = [0]
    for i, diff in enumerate(depth_diffs):
        if diff > 0:
            bads.append(i)
    mask2 = np.ones(len(depths), dtype=bool)
    mask2[bads] = False
    depths = depths[mask2]
    widths = widths[mask2]
    """
    # x = np.linspace(0, len(bkgd), len(bkgd))

    # plt.plot(x, bkgd)
    # plt.plot(depths, widths)
    plotter(depths, widths)
    plt.show()


main()
