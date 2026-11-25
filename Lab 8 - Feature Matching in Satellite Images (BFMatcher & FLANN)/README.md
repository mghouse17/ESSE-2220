# Lab 8: Feature Matching in Satellite Images (BFMatcher & FLANN)

This lab compares feature matching in pre- and post-disaster satellite images using OpenCV's Brute-Force Matcher (BFMatcher) and FLANN.

The implementation supports SIFT and ORB feature detection and demonstrates:

- Raw BFMatcher and FLANN matching
- Distance-threshold filtering
- Lowe's ratio test
- Symmetry checking with cross-check matching

## Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the experiment from this lab directory:

```bash
python Lab8_temp.py
```

The repository includes the input satellite images and the generated matching visualizations.
