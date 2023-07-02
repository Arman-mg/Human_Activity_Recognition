# **Human Activity Recognition Using Inertial Sensors**

Welcome to the Human Activity Recognition repository, a project dedicated to monitoring, identifying, and categorizing human activities using body-worn sensors. The project employs advanced sensor technologies and offers a comprehensive comparative analysis of various strategies to categorize human activity based on sensor data.

## **Project Description**

This project involves collecting data from multiple sensors attached to various parts of a subject's body, such as the torso, right arm, left arm, right leg, and left leg. The data is then preprocessed and categorized using the K-Means clustering technique. The performance of the model and the results of the analysis are subsequently evaluated.

The data is collected from 3-axis accelerometers, gyroscopes, and magnetometers, providing 45 values for each sample. Eight subjects, comprised of four females and four males aged between 20 and 30, participated in the study. Each subject performed 19 different activities, each for a duration of 5 minutes.
Data Analysis

The data were collected at a frequency of 25 Hz from the various sensors on the body, with the signals divided into 5-second segments. This resulted in a total of 480 segments for each activity.
Feature Extraction

A discrete-time sequence was obtained after capturing the signals as described above. The first set of features were derived from the minimum and maximum values, the mean, variance, skewness, kurtosis, autocorrelation sequence, and the peaks of the Discrete Fourier Transform (DFT) of components at the relevant frequencies.
