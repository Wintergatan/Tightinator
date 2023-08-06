import os
import uuid
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, jsonify

app = Flask(__name__)

UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def generate_csv(input_wav, output_csv, threshold):
    # Run your Python script here to generate the CSV file based on the uploaded WAV file
    # Example: Call main.py with filled arguments
    os.system(f"python3 main.py -f {input_wav} -o {output_csv} -t {threshold}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.wav'):
            unique_id = str(uuid.uuid4())
            csv_directory = os.path.join(app.config['UPLOAD_FOLDER'], unique_id)
            os.makedirs(csv_directory)  # Create a directory for the UUID
            wav_path = os.path.join(csv_directory, f'{unique_id}.wav')
            file.save(wav_path)  # Save the uploaded WAV file

            return jsonify({'status': 'success', 'uuid': unique_id})

    return render_template('index.html')

@app.route('/config/<uuid>', methods=['GET'])
def config(uuid):
    # Call generate_csv here after the user is redirected to the config endpoint
    #generate_csv(f'static/upload/{uuid}/{uuid}.wav', output_csv, threshold)

    return render_template('config.html', uuid=uuid)

@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    unique_id = str(uuid.uuid4())
    output_filename = f'{unique_id}.csv'

    # Run your Python script here to generate the CSV file based on the uploaded WAV file
    # Example: generate_csv_from_wav(filename, output_filename)

    # For demonstration purposes, we'll create an empty CSV file with the generated filename
    csv_directory = os.path.join(app.config['UPLOAD_FOLDER'], unique_id)
    os.makedirs(csv_directory)  # Create a directory for the UUID
    csv_path = os.path.join(csv_directory, output_filename)
    open(csv_path, 'w').close()

    return render_template('download.html', filename=output_filename)

@app.route('/downloads/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory('', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

