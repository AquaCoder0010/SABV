# SABV

PE file to Image converter
Converts any PE file (or any file) and visualizes it through fuzzy inference


# Installation

```bash
git clone https://github.com/AquaCoder0010/SABV.git
pip install -r requirements.txt



# Example

```python
sabv = SignatureAgnosticBinaryVisualizer(N=10)
img = sabv.process_file("ENTER-FILE-PATH")
cv2.imwrite(img, "IMAGE_PATH")


