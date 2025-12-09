# SABV (Signature-Agnostic Binary Visualizer)

A PE/binary-to-image converter designed to visualize any binary file.  
Optionally integrates a fuzzy-inference system (FIS) inspired by the SAGMAD methodology.

SABV allows:
- Conversion of arbitrary binary files into structured image representations  
- Optional fuzzy-inference–enhanced visualizations  
- Configurable sampling, resolution, and threading

---

# Installation

```bash
git clone https://github.com/AquaCoder0010/SABV.git
pip install -r requirements.txt
```

---
# Example Usage

## Basic Visualization (FIS disabled)

```python
from SABV import SABV
import cv2

# visualization without fuzzy inference
sabv = SignatureAgnosticBinaryVisualizer()
img = sabv.process_file("ENTER-FILE-PATH")

cv2.imwrite("IMAGE_PATH.png", img)
```
## Visualization With Fuzzy Inference System (FIS)

```python
from SABV import SABV
import cv2

sabv_with_fis = SABV(FIS_ENABLED=True, N=3, sample=0.05, FIS_THREADING_ENABLED=True)
img = sabv.process_file("ENTER-FILE-PATH")

cv2.imwrite("IMAGE_PATH.png", img)
```

# Benchmark
<img src="images/Figure_1.png" alt="alt text" width>

# Example Images
## with FIS
<img src="images/sabv-FIS.png" alt="alt text" width="300">

## without FIS
<img src="images/sabv-no-FIS.png" alt="alt text" width="300">

# Citation
Saridou, B.; Rose, J. R.; Shiaeles, S.; Papadopoulos, B.  
*SAGMAD—A Signature Agnostic Malware Detection System Based on Binary Visualisation and Fuzzy Sets.*  
**Electronics**, 2022, **11**, 1044.  
https://doi.org/10.3390/electronics11071044

