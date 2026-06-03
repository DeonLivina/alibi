# alibi — witness-channel toy dataset for gravitational-wave embeddings

This repository explores embeddings of gravitational-wave (GW) data, with a focus on
one question:

> **Does embedding witness / auxiliary-channel information alongside the signal-carrying
> strain channel help a model tell real astrophysical signals apart from instrumental
> glitches?**

The `dataset/` folder generates a labelled **toy dataset** designed to study exactly
that. It combines **real interferometer background** (from GWOSC) with **ml4gw**-simulated
transients, and adds a **synthetic witness channel** that is coupled to glitches but blind
to astrophysical signals.

## The idea

In a real interferometer, a witness (auxiliary) sensor records instrumental/environmental
disturbances but does **not** respond to gravitational waves. That asymmetry is the
physical basis of glitch vetoing, and it is what this dataset encodes across three classes:

| Class          | Strain channel (H1)                    | Witness channel                         |
|----------------|----------------------------------------|-----------------------------------------|
| **BBH signal** | real noise + injected CBC chirp        | noise only (a GW does not couple here)  |
| **Glitch**     | real noise + injected glitch transient | noise + **coupled copy** of the glitch  |
| **Background** | real noise only                        | noise only                              |

A model with access to the witness can learn: *strain transient + correlated witness
transient → glitch; strain transient with no witness counterpart → signal.* A model
restricted to the strain channel cannot make that distinction as cleanly. The 2-channel
output makes the ablation trivial: train on `data[:, 0:1]` (strain only) vs
`data[:, 0:2]` (strain + witness).

The signal and glitch transients are placed at the **same time location**, so the witness —
not timing — is the discriminator.

## What is simulated, and how

* **Signals**: ml4gw CBC waveforms (`IMRPhenomD`), projected onto the detector and rescaled
  to a target network SNR (`ml4gw.gw.reweight_snrs`).
* **Glitches**: ml4gw ad-hoc `SineGaussian` transients. The glitch *source* is what the
  witness sees; it couples into the strain through an **LTI Butterworth filter**. Only a
  fraction `alpha` of the strain-glitch power is coherent with the witness (the rest is an
  independent strain-only component) — so `alpha` directly sets how informative the witness
  is.
* **Background**: real H1 strain from GWOSC O3a; the witness background is synthesised
  Gaussian noise.
* **Whitening**: per-channel PSD estimation + `ml4gw.transforms.Whiten`.

The design loosely follows [`chreissel/GWDatasetGeneration`](https://github.com/chreissel/GWDatasetGeneration),
extended with the witness channel and the glitch class. A single strain detector (H1) is
used for now.

## Layout

```
README.md            # this file
requirements.txt     # dependencies
dataset/
  configs/config_H1.yaml   # all knobs (signal/glitch priors, witness coupling, whitening)
  load_data.py             # download real H1 background from GWOSC
  waveforms.py             # ml4gw CBC signals + SineGaussian glitch sources
  witness.py               # witness-noise synthesis + glitch->strain/witness coupling
  injections.py            # per-class injection -> whitened [strain, witness] batch
  main.py                  # orchestration -> writes one HDF5 per class
  distributions.py utils.py set_seed.py  # prior/config/seed helpers
  test_smoke.py            # offline test (no network): shapes + witness asymmetry
```

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

2. **Generate the dataset**:

   ```bash
   python main.py --config configs/config_H1.yaml --data ./data/background_data --out ./out
   ```
   This writes `background.h5`, `signal.h5`, `glitch.h5` into `./out`.

If GWOSC access is blocked in your environment, point `--data` at any folder of HDF5 files
that each contain an `H1` dataset (a strain time series) longer than the analysis window;
the witness channel is synthesised regardless.

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

## Test

```bash
cd dataset && python test_smoke.py
```
Runs fully offline (synthesises a fake background), checks output shapes, and verifies the
core asymmetry: the witness is correlated with the strain transient for glitches but not for
signals.
