import pandas as pd
import pickle
import numpy as np
import sys

FILENAME = sys.argv[1]
OUT_FILENAME = sys.argv[2]

cols = ["videoID", "t", "x1", "y1", "x2", "y2", "class", "tracklet"]

df = pd.read_csv(FILENAME, names=cols)


result_dict = {}

for name, group in df.groupby("videoID"):
    for t, group2 in group.groupby("t"):
        frame_id = ",".join((str(name), str(t).zfill(4)))
        prop_fn = lambda r: [r["x1"], r["y1"], r["x2"], r["y2"], 0.99]
        proposals = list(group2.apply(prop_fn, axis=1))
        result_dict[frame_id] = np.vstack(proposals)

with open(OUT_FILENAME, "wb") as file:
    pickle.dump(result_dict, file)
