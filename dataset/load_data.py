"""Download real interferometer background data from GWOSC.

Adapted from chreissel/GWDatasetGeneration. This toy dataset focuses on a single
detector (H1 by default), so ``config.general.ifos`` should contain exactly one
interferometer. The strain background is real; the witness channel is synthesised
later in ``witness.py`` (GWOSC open data does not expose auxiliary channels).
"""

import operator
from functools import reduce
from pathlib import Path

import requests
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from gwpy.segments import Segment, SegmentList
from tqdm import tqdm

from utils import load_config

BASE_URL = "https://gwosc.org/api/v2/runs/{run}/timelines"


def fetch_segments(run: str, detector: str = "H1"):
    timeline = f"{detector}_DATA"
    url = f"{BASE_URL.format(run=run)}/{timeline}/segments"

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    segments = []
    for seg in data["results"]:
        start = int(seg["start"])
        end = int(seg["stop"])
        segments.append(Segment(start, end))

    return SegmentList(segments)


def load_data(config, data_dir: str):

    data_dir = Path(data_dir)
    background_dir = data_dir / "background_data"
    background_dir.mkdir(parents=True, exist_ok=True)

    run = getattr(config.general, "run", "O3a")
    # Cap on how many seconds of background to download (None -> all segments).
    max_seconds = getattr(config.general, "max_background_seconds", None)

    segments = {}
    for ifo in config.general.ifos:
        segments[ifo] = fetch_segments(run, ifo)
    network_segments = reduce(operator.and_, segments.values())

    downloaded = 0
    for (start, end) in tqdm(network_segments, desc="Fetching background"):
        duration = end - start
        if duration < config.general.waveform_duration:
            continue

        fname = background_dir / f"background-{start}-{duration}.hdf5"
        if not fname.exists():
            ts_dict = TimeSeriesDict()
            for ifo in config.general.ifos:
                ts_dict[ifo] = TimeSeries.fetch_open_data(ifo, start, end, cache=True)
            ts_dict = ts_dict.resample(config.general.sample_rate)
            ts_dict.write(fname, format="hdf5")

        downloaded += duration
        if max_seconds is not None and downloaded >= max_seconds:
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download GWOSC background data")
    parser.add_argument("--config", type=str, default="configs/config_H1.yaml")
    parser.add_argument("--data", type=str, default="./data")
    args = parser.parse_args()

    config = load_config(config_path=args.config)
    load_data(config, args.data)
