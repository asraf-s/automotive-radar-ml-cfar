Real-Time FMCW Automotive Radar Detection Using ML-Assisted Adaptive 2D CA-CFAR

Overview

This project presents an intelligent automotive radar prototype for real-time target detection using Frequency Modulated Continuous Wave (FMCW) radar principles. The system estimates target range (distance) and velocity (speed) using Range-Doppler processing and improves detection robustness using a machine learning assisted adaptive CFAR thresholding method.

The final adaptive model uses an Extra Trees Regressor to predict the optimal CFAR offset for each radar frame.

Key Features
	•	FMCW radar signal simulation
	•	Range FFT for distance estimation
	•	Doppler FFT for velocity estimation
	•	2D Range-Doppler map generation
	•	Adaptive 2D CA-CFAR detection
	•	Machine learning threshold prediction
	•	Real-time frame processing
	•	Quantitative performance evaluation

Final Results
	•	Correct Detections: 132
	•	Recall: 72.5%
	•	Precision: 46.3%
	•	F1 Score: 0.565
	•	Average Range Error: 0.337 m
	•	Average Velocity Error: 1.196 m/s
	•	Average Runtime: 30.8 ms/frame

  Machine Learning Comparison

The following models were evaluated:
	•	Linear Regression
	•	Random Forest
	•	SVR
	•	Neural Network (MLP)
	•	Extra Trees Regressor ✅ Best

Best model performance:
	•	MAE: 0.323
	•	R²: 0.812
Future Work
	•	Real radar dataset validation (CARRADA / RadarScenes)
	•	Multi-target tracking
	•	Embedded deployment
	•	Deep learning on radar maps


Author

ASRAF S
