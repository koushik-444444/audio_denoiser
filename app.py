import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy import signal
from tensorflow.keras.models import load_model
import librosa

app = Flask(__name__)

# Folder paths for file uploads and results
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Load the Keras model (Make sure you have 'my_model.keras' in the same folder or provide the path)
model = load_model('my_model.keras')

# Function to check if the uploaded file is a valid audio file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the index page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'audio_file' not in request.files:
            return render_template('index.html', error="No file part")
        file = request.files['audio_file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Denoise the audio using the model
            denoised_filepath = os.path.join(app.config['RESULTS_FOLDER'], f"denoised_{filename}")
            denoised_audio = denoise_audio(filepath, denoised_filepath)

            # Plot original and denoised spectrograms
            original_plot = plot_spectrogram(filepath, 'Original')
            denoised_plot = plot_spectrogram(denoised_filepath, 'Denoised')

            return render_template(
                'index.html',
                orig_audio=filepath,
                orig_plot=original_plot,
                denoised_audio=denoised_audio,
                denoised_plot=denoised_plot
            )
    return render_template('index.html')

# Function to denoise audio using the Keras model
def denoise_audio(input_file, output_file):
    # Step 1: Read audio file using librosa (this ensures mono format and resampling)
    audio, sr = librosa.load(input_file, sr=16000, mono=True)

    # Step 2: Preprocess audio for the model (reshape to model's expected input shape)
    audio_input = np.reshape(audio, (1, len(audio), 1))  # Modify this depending on your model's input shape
    audio_input = audio_input.astype('float32')

    # Step 3: Denoise the audio using the model
    denoised_audio = model.predict(audio_input)

    # Step 4: Postprocess (flatten and convert to int16 for wav format)
    denoised_audio = np.reshape(denoised_audio, (-1,))
    denoised_audio = np.int16(denoised_audio * 32767)  # Rescale to 16-bit PCM format

    # Step 5: Save the denoised audio to file
    wav.write(output_file, sr, denoised_audio)

    return output_file

# Function to plot the spectrogram of an audio file
def plot_spectrogram(audio_file, title):
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    f, t, Sxx = signal.spectrogram(audio, sr)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, np.log(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(f'{title} Spectrogram')

    plot_filename = f'{title.lower()}_spectrogram.png'
    plot_filepath = os.path.join(app.config['RESULTS_FOLDER'], plot_filename)
    plt.savefig(plot_filepath)
    plt.close()

    return plot_filename

# Route to serve uploaded and result files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
