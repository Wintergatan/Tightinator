import os
import uuid
import time
import subprocess
import threading
from threading import Thread
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, jsonify, send_file, Response

app = Flask(__name__)

UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

processes = {}

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
            wav_path = os.path.join(csv_directory, file.filename)
            file.save(wav_path)  # Save the uploaded WAV file

            return jsonify({'status': 'success', 'uuid': unique_id, 'filename': file.filename})

    return render_template('index.html')

@app.route('/config/<uuid>/<filename>', methods=['GET'])
def config(uuid, filename):
    # Call generate_csv here after the user is redirected to the config endpoint
    #generate_csv(f'static/upload/{uuid}/{uuid}.wav', output_csv, threshold)
    output_filename = filename[:-4]+".csv"

    return render_template('config.html', uuid=uuid, filename=filename, output_filename=output_filename)

@app.route('/run/<uuid>/', methods=['POST'])
def run(uuid):
    filename = str(request.form['filename'])
    output_filename = str(request.form['output_filename'])
    threshold = float(request.form['threshold'])
    rounding = int(request.form['rounding'])

    debug_command = [
        'python3',
        'main.py',
        '-f', "static/upload/{}/{}".format(uuid, filename),
        '-o', "static/upload/{}/{}".format(uuid, output_filename)
    ]

    print("Debug Command:", " ".join(debug_command))

    try:
        process = subprocess.Popen(debug_command, text=True)

        return redirect(url_for('running', uuid=uuid, output_filename=output_filename))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/running/<uuid>/<output_filename>')
def running(uuid, output_filename):
    output_path = os.path.join(app.root_path, 'static', 'upload', uuid, output_filename)

    while not os.path.exists(output_path):
        return render_template('running.html', uuid=uuid, output_filename=output_filename)
        time.sleep(1)  # Wait for 1 second before checking again

    return redirect(url_for('result', uuid=uuid, output_filename=output_filename))

'''
@app.route('/running/<uuid>/<output_filename>')
def running(uuid, output_filename):

    csv_path = os.path.join(app.root_path, 'static', 'upload', uuid, output_filename)

    if os.path.exists(csv_path):
        return redirect(url_for('result', uuid=uuid, output_filename=output_filename))  # Redirect to the "Process Complete" page
    else:
        return render_template('running.html', uuid=uuid, output_filename=output_filename)   # Continue rendering the "Running..." page

    if process is None:
        return render_template('result.html', uuid=uuid, output="Process not found")

    if process.poll() is None:
        return render_template('running.html', uuid=uuid, output_filename=output_filename, process_output=process_output)
    else:
        return render_template('result.html', uuid=uuid, output_filename=output_filename, output=process_output)
'''
def wait_for_process(uuid, process):
    process.wait()

@app.route('/result/<uuid>/<output_filename>')
def result(uuid, output_filename):
    failure = request.args.get('failure', default=False, type=bool)
    return render_template('result.html', uuid=uuid, output_filename=output_filename, failure=failure)

@app.route('/download/<uuid>/<output_filename>')
def download(uuid,output_filename):
    csv_path = os.path.join(app.root_path, 'static', 'upload', uuid, output_filename)

    if os.path.exists(csv_path):
        return send_file(csv_path, as_attachment=True, download_name=output_filename)
    else:
        return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True)

