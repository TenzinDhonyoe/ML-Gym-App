<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>🏋️‍♂️ ML Gym Tracker</h1>

<p>A machine learning application to track your gym workouts using computer vision and pose estimation. The project uses Mediapipe for pose detection and a custom-trained model to classify and count repetitions of exercises.</p>

<h2>📁 Files</h2>
<ul>
    <li><code>app.py</code> - The main application code that sets up the GUI, video capture, and pose detection.</li>
    <li><code>landmarks.py</code> - Contains the list of landmarks used for pose detection.</li>
    <li><code>deadlift.pkl</code> - A pickled file containing the trained model data for detecting deadlift exercise stages.</li>
</ul>

<h2>🚀 Getting Started</h2>

<h3>Prerequisites</h3>
<ul>
    <li>Python 3.8 or higher</li>
    <li>Required Python packages (install using <code>pip</code>):</li>
    <ul>
        <li><code>tkinter</code></li>
        <li><code>customtkinter</code></li>
        <li><code>pandas</code></li>
        <li><code>numpy</code></li>
        <li><code>mediapipe</code></li>
        <li><code>opencv-python</code></li>
        <li><code>Pillow</code></li>
    </ul>
</ul>

<h3>Installation</h3>
<ol>
    <li>Clone the repository:
        <pre><code>git clone https://github.com/yourusername/ML-Gym-Tracker.git
cd ML-Gym-Tracker</code></pre>
    </li>
    <li>Install the required packages:
        <pre><code>pip install -r requirements.txt</code></pre>
    </li>
    <li>Place the <code>deadlift.pkl</code> file in the project directory.</li>
</ol>

<h3>Running the Application</h3>
<ol>
    <li>Execute the main application script:
        <pre><code>python app.py</code></pre>
    </li>
    <li>The application window will open, displaying the video feed and tracking information.</li>
</ol>

<h2>📊 Project Structure</h2>
<ul>
    <li><strong>Main Window Setup</strong>: The <code>tkinter</code> window is initialized with a title, size, and appearance mode set to dark.</li>
    <li><strong>Labels and Buttons</strong>: Various labels are created to display the current stage, repetition count, and probability of the detected pose. A reset button is provided to reset the repetition counter.</li>
    <li><strong>Video Frame Setup</strong>: A frame is set up to display the video feed from the webcam.</li>
    <li><strong>Pose Detection</strong>: Mediapipe is used for pose detection, and the trained model is loaded to classify the detected poses and count repetitions.</li>
    <li><strong>Detection Loop</strong>: A loop function <code>detect</code> captures video frames, processes them for pose detection, and updates the GUI with the current state and counts.</li>
</ul>

</body>
</html>
