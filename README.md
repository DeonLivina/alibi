# alibi — Embedding witness channel information for gravitational-wave data analysis

This repository provides the guideline to answer the question:

> **Does embedding witness channel information alongside the signal-carrying
> strain channel help a model tell real astrophysical signals apart from instrumental
> glitches?**

We approach this questions by generating a labelled **toy dataset**. It combines real interferometer background with simulated
transients, and adds a **synthetic witness channel** that is coupled to glitches but blind to astrophysical signals.

## The idea

In a real interferometer, a witness (auxiliary) sensor records instrumental/environmental
disturbances but does **not** respond to gravitational waves. That asymmetry is the
physical basis of glitch vetoing, and it is what this dataset encodes across three classes:

| Class          | Strain channel (H1)                    | Witness channel                         |
|----------------|----------------------------------------|-----------------------------------------|
| **Signal**     | real noise + injected gravitational wave (GW) signal        | noise only (a GW does not couple here)  |
| **Glitch**     | real noise + injected glitch transient | noise + coupled copy of the glitch  |
| **Background** | real noise only                        | noise only                              |

The signal and glitch transients are placed at the same time location, so the witness, not the arrival time, is the discriminator.

## What is simulated, and how

* **Signals**: ml4gw CBC waveforms (`IMRPhenomD`), projected onto the detector and rescaled
  to a target network SNR (`ml4gw.gw.reweight_snrs`). We start by considering shorter signals from binary black hole merging.
* **Glitches**: ml4gw ad-hoc `SineGaussian` transients. The glitch source is what the
  witness sees; it couples into the strain through an *LTI Butterworth filter*. Only a
  fraction `alpha` of the strain-glitch power is coherent with the witness (the rest is an
  independent strain-only component) — so `alpha` directly sets how informative the witness
  is.
* **Background**: real H1 strain from GWOSC O3a; the witness background is synthesised
  Gaussian noise.
* **Whitening**: per-channel PSD estimation + `ml4gw.transforms.Whiten`.

The design loosely follows [`chreissel/GWDatasetGeneration`](https://github.com/chreissel/GWDatasetGeneration),
extended with the witness channel and the glitch class. WE consider only a one-detector setup (H1) for now.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. **Download real background** (needs network access to GWOSC):

   ```bash
   cd dataset
   python load_data.py --config configs/config_H1.yaml --data ./data
   ```
   This writes `./data/background_data/background-*.hdf5`. 
   It can be omitted when working on the FASRC cluster since the O3a dataset is already downloaded and available following the path: `/n/holystore01/LABS/iaifi_lab/Lab/creissel/SparseBank/background_data/`.

2. **Generate the dataset**:

   ```bash
   python main.py --config configs/config_H1.yaml --data ./data/background_data --out ./out
   ```
   This writes `background.h5`, `signal.h5`, `glitch.h5` into `./out`.

## Output format

One HDF5 file per class, each containing:

* `data`  — `(N, 2, T)` float array, channel order **`[strain, witness]`** (whitened),
  with `T = waveform_duration * sample_rate`.
* `label` — `(N,)` int: `0=background, 1=signal, 2=glitch`.
* one dataset per sampled parameter (e.g. `snr`, `chirp_mass` for signals; `frequency`,
  `quality`, `strain_snr`, `witness_snr` for glitches).
* attrs: `label` (the class id) and `channels` (`[strain, witness]`).

## Key config knobs (`dataset/configs/config_H1.yaml`)

* `general` — detector, sample rate, window duration, counts, GWOSC run.
* `waveform` / `snr_reweighting` — CBC prior and signal SNR distribution.
* `glitch.prior` / `glitch.snr` — SineGaussian parameter prior and per-channel SNRs.
* `witness.coupling.alpha` — strain↔witness coherence (the main "how useful is the witness"
  knob); `witness.coupling.filter` — the Butterworth coupling band.
