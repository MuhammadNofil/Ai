from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt  # Add this import
import os
import threading
import tensorflow as tf
import numpy as np
import librosa

sess = threading.local()

def get_session():
    if not hasattr(sess, 'session'):
        current_directory = os.path.dirname(__file__)
        model_path = os.path.join(current_directory, "model.tflite")
        print(model_path)
        sess.session = tf.lite.Interpreter(model_path=model_path)
        sess.session.allocate_tensors()
    return sess.session

@csrf_exempt
def api(request):
    print(request)
    session = get_session()
    if request.method == 'POST' and request.FILES.get('mp3_file'):
        mp3_file = request.FILES['mp3_file']
        print(mp3_file)
        # Initialize lists to store audio chunks and sample rates
        audio_chunks = []
        sample_rates = []

        # Load the MP3 file using librosa
        audio_data, sample_rate = librosa.load(mp3_file, sr=None, mono=True)

        # Resize audio data to match expected input shape (if needed)
        expected_length = 44032
        if len(audio_data) < expected_length:
            audio_data = np.pad(audio_data, (0, expected_length - len(audio_data)), mode='constant')
        else:
            audio_data = audio_data[:expected_length]

        # Perform inference on the entire audio clip
        input_data = audio_data.reshape(1, -1).astype(np.float32)
        input_details = session.get_input_details()
        session.set_tensor(input_details[0]['index'], input_data)
        session.invoke()
        output_details = session.get_output_details()
        output_data = session.get_tensor(output_details[0]['index'])

        # Calculate softmax probabilities
        softmax_output = tf.nn.softmax(output_data[0]).numpy()

        # Define the label descriptions based on your provided labels
        label_descriptions = {
            0: 'Background Noise',
            1: 'Belt',
            2: 'Belt Tensioner',
            3: 'Knocking',
            4: 'Rattling'
        }

        # Create a dictionary to store label percentages
        label_percentages = {}

        # Calculate confidence percentage for each label and round to remove decimal
        for label_index, confidence in enumerate(softmax_output):
            label_description = label_descriptions.get(label_index, 'Unknown')
            label_percentages[label_description] = round(confidence * 100)

        # Return the predicted labels and their confidence percentages
        return JsonResponse({'label_percentages': label_percentages, 'sample_rate': sample_rate})

    else:
        return HttpResponse(status=400, content='Bad request. Please provide a valid MP3 file.')
