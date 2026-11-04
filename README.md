# Human Activity Recognition: How Well Does It Work on New People?

Human activity recognition (HAR) on the UCI HAR dataset. The main question is how much of the reported accuracy holds up when a model is tested on a person it has never seen.

This is my term project for CS6140 Machine Learning at Northeastern University.

## Why this matters

The data is cut into short windows that overlap by 50%. If windows are shuffled randomly into train and test sets, the test set contains near-copies of training windows from the same person, so accuracy can look better than it really is. This project measures that effect and separates it from the harder problem of generalizing to new people.

## Dataset

[UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) (Anguita et al., 2013)

- 30 people wearing a smartphone on the waist, 6 activities (walking, upstairs, downstairs, sitting, standing, laying)
- 10,299 windows of 2.56 s (128 samples at 50 Hz), 50% overlap
- 561 precomputed features per window, plus raw accelerometer and gyroscope signals (9 channels)
- Official split by person: 21 people in train (7,352 windows), 9 in test (2,947 windows), no overlap
- Classes are roughly balanced (1,406 to 1,944 windows each)

The notebook downloads the dataset automatically.

## Progress so far

- [x] Data downloaded, loaded, and checked (window counts, person split, class balance)
- [x] Baseline models trained on the official person-wise split
- [x] Proposal submitted and feedback received
- [ ] Split experiments to separate overlap leakage from person shift
- [ ] 1D-CNN on raw signals
- [ ] Error analysis and per-person results
- [ ] Learning curve over number of training people

### Baseline results (official split, untuned)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Logistic regression | 95.5% | 95.5% |
| SVM (RBF kernel) | 95.5% | 95.4% |
| Random forest | 92.5% | 92.3% |
| Gradient boosting | 93.3% | 93.3% |

For comparison, the original paper reports 96% test accuracy with a tuned SVM on the same split.

## Next steps

Based on proposal feedback, the plan now separates the random-split vs. person-split gap into distinct effects:

1. Random windows vs. random non-overlapping windows (overlap leakage)
2. Non-overlapping windows vs. same-person time-based split (closeness in time)
3. Same-person split vs. person-disjoint split (person shift)

Other planned work: tuning with person-wise cross-validation only, reporting variation across people (not just overall accuracy), and repeating random person subsets in the learning curves.

## How to run

1. Open `har_project.ipynb` in [Google Colab](https://colab.research.google.com) (File → Upload notebook).
2. Run the cells from top to bottom. The first cells download and load the data.
3. Results and plots are saved to the `results/` folder.

A free Colab CPU is enough for the classical models. A GPU helps for the CNN.

## Reference

D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz. "A Public Domain Dataset for Human Activity Recognition Using Smartphones." ESANN 2013.
