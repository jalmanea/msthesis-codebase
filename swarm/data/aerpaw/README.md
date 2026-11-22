# AERPAW Datasets for Channel Model Validation

This directory contains air-to-ground channel measurement data from the [AERPAW](https://aerpaw.org/) testbed for validating our drone communication channel model.

## Recommended Datasets

### 1. Primary: A2G Channel Sounding (244 MB) - RECOMMENDED

| Field | Value |
|-------|-------|
| **DOI** | [10.5061/dryad.7h44j105p](https://doi.org/10.5061/dryad.7h44j105p) |
| **Size** | 244 MB |
| **Format** | SigMF (I/Q + metadata) |
| **Frequency** | CBRS band (~3.5 GHz) |
| **Features** | GPS coordinates, altitude, multipath measurements |
| **License** | CC0 1.0 Universal |

**Download:**
```bash
# Option 1: Direct download
wget -O A2G_Channel_Measurements.zip \
  "https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.7h44j105p/download"

# Option 2: Manual
# Visit https://datadryad.org/dataset/doi:10.5061/dryad.7h44j105p
# Click "Download dataset"
```

**Citation:**
```bibtex
@misc{gurses2025aerpaw,
  author = {Gürses, Anil and Sichitiu, Mihail L.},
  title = {{AERPAW Air-to-ground channel sounding and multipath measurement with a UAV}},
  year = {2025},
  publisher = {Dryad},
  doi = {10.5061/dryad.7h44j105p}
}
```

---

### 2. Alternative: Multi-Altitude Signal Data (150 GB)

| Field | Value |
|-------|-------|
| **DOI** | [10.5061/dryad.2z34tmpvv](https://doi.org/10.5061/dryad.2z34tmpvv) |
| **Size** | 150.84 GB |
| **Format** | SigMF |
| **Frequency** | 3.32 GHz |
| **Altitudes** | 40m, 70m, 100m |
| **License** | CC0 1.0 Universal |

**Note:** Large dataset. Use if you need systematic altitude variation analysis.

**Citation:**
```bibtex
@misc{dickerson2024aerpaw,
  author = {Dickerson, Cole and Fahim Raouf, Amir Hossein and Ozdemir, Ozgur and others},
  title = {{AERPAW UAV-based signal data collected at varying altitudes and sampling rates}},
  year = {2024},
  publisher = {Dryad},
  doi = {10.5061/dryad.2z34tmpvv}
}
```

---

### 3. Alternative: IEEE DataPort LTE I/Q (8.8 GB)

| Field | Value |
|-------|-------|
| **DOI** | [10.21227/0p43-0d72](https://doi.org/10.21227/0p43-0d72) |
| **Size** | 8.8 GB |
| **Format** | SigMF |
| **Frequency** | 3.51 GHz (LTE) |
| **Altitudes** | 30m, 50m, 70m, 90m, 110m |
| **Access** | IEEE membership required |

**Associated Paper:** [IEEE Communications Magazine](https://doi.org/10.1109/MCOM.005.2200784)

**Code:** [Code Ocean](https://doi.org/10.24433/CO.6586124.v1)

**Citation:**
```bibtex
@article{maeng2023lte,
  author = {Maeng, Sung Joon and Ozdemir, Ozgur and Guvenc, Ismail and Sichitiu, Mihail and Dutta, Rudra},
  title = {{LTE I/Q Data Set for UAV Propagation Modeling, Communication, and Navigation Research}},
  journal = {IEEE Communications Magazine},
  year = {2023},
  doi = {10.1109/MCOM.005.2200784}
}
```

---

## Data Format

All datasets use **SigMF** (Signal Metadata Format):

- `.sigmf-meta` - JSON metadata (frequency, sample rate, GPS, gains)
- `.sigmf-data` - Binary I/Q samples

**Python loading:**
```python
import json
import numpy as np

# Load metadata
with open('recording.sigmf-meta') as f:
    meta = json.load(f)

# Load I/Q samples (complex float32)
iq = np.fromfile('recording.sigmf-data', dtype=np.complex64)
```

---

## Frequency Adjustment

Our model uses 2.0 GHz; AERPAW data is at ~3.3-3.5 GHz.

**Path loss adjustment:**
```
ΔPL = 20·log₁₀(f_measured / f_model)
    = 20·log₁₀(3.3 GHz / 2.0 GHz)
    = 4.3 dB
```

This offset is handled automatically in the validation notebook.

---

## Directory Structure

After downloading:
```
data/aerpaw/
├── README.md                          # This file
├── A2G_Channel_Measurements/          # Primary dataset
│   ├── 2023-12-15_15_41/
│   │   ├── recording_001.sigmf-meta
│   │   ├── recording_001.sigmf-data
│   │   └── ...
│   └── ...
├── fitted_channel_params.json         # Output from validation
└── *.zip                              # Downloaded archives
```

---

## Validation Notebook

Run the validation with:
```bash
cd dronecomm/notebooks
jupyter notebook validation_aerpaw.ipynb
```

The notebook will:
1. Load AERPAW measurements
2. Extract path loss vs distance/elevation
3. Compare against Al-Hourani model predictions
4. Fit environment parameters to real data
5. Generate validation plots and statistics
