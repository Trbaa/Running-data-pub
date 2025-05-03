# 🏃‍♂️ Running Time Prediction

This project analyzes running data and predicts the remaining time to finish a run, based on GPS data from previous activities (GPX files).

## 💡 Why I built this
I was recently asked if I had a project I could share. I didn’t.  
That motivated me to sit down and build something useful from scratch — combining two things I care about: running and learning to code.

## 📁 What it does

- Loads GPX files from a folder
- Parses all location, time, and elevation data
- Calculates distance, speed, grade, and cumulative metrics
- Predicts how much time is left until the end of the run using a simple machine learning model (Linear Regression)

## 🔧 Tech

- Python 🐍
- Pandas & NumPy for data handling
- scikit-learn for prediction
- geopy for distance calculations

## 🧠 What's next

- Real-time tracking and prediction
- Web or mobile interface for live feedback while running
- Smarter models and personalized predictions

## Try it yourself!

- Add your data in 'data' folder 
- Start the program
- Enjoj :)