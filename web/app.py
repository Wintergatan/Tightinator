import os
import uuid
import subprocess
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, jsonify, send_file, Response

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

@app.route('/run/<uuid>', methods=['POST'])
def run(uuid):
    threshold = float(request.form['threshold'])
    rounding = int(request.form['rounding'])

    try:
        # Run the Python script with subprocess in the background
        subprocess.Popen(['python3', 'main.py', '-f', "static/upload/{}/{}.wav".format(uuid, uuid), '-o', "static/upload/{}/{}.csv".format(uuid, uuid)])

        # Redirect to the "Running..." page
        return redirect(url_for('running', uuid=uuid))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/running/<uuid>')
def running(uuid):
    return render_template('running.html', uuid=uuid)

@app.route('/download/<uuid>')
def download(uuid):
    csv_path = os.path.join(app.root_path, 'static', 'upload', uuid, f'{uuid}.csv')

    if os.path.exists(csv_path):
        return send_file(csv_path, as_attachment=True, download_name=f'{uuid}.csv')
    else:
        return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True)

