import os
import json
import glob
import uuid
import time
#import logging
import subprocess
import threading
from threading import Thread
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, jsonify, send_file, Response

app = Flask(__name__)

UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

processes = {}

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
    processes[uuid] = filename
    print("Writing processes[{}]={}".format(uuid, filename))
    #logging.info("Writing processes[{}]={}".format(uuid, filename))
    save_processes_to_file(processes)
    output_filename = filename[:-4]+".csv"
    output_filename = output_filename.replace(" ", "_")

    return render_template('config.html', uuid=uuid, filename=filename, output_filename=output_filename)

@app.route('/config/<uuid>/', methods=['GET'])
def rerun(uuid):
    processes = load_processes_from_file()
    filename = processes[uuid]
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    try:
        os.rename("static/upload/{}/summary.html".format(uuid), "static/upload/{}/summary-{}.html".format(uuid, timestamp))
    except Exception as e:
        print(e)

    output_filename = filename[:-4]+".csv"
    output_filename = output_filename.replace(" ", "_")

    return render_template('config.html', uuid=uuid, filename=filename, output_filename=output_filename)


@app.route('/run/<uuid>/', methods=['POST'])
def run(uuid):
    filename = str(request.form['filename'])
    output_filename = str(request.form['output_filename'])
    alg = str(request.form['algSelect'])
    bpm = str(request.form['bpm'])
    bpmWin = str(request.form['bpmWin'])
    threshold = str(request.form['threshold'])
    sample_size = str(request.form['sample'])
    rounding = str(request.form['rounding'])
    channel = str(request.form['channelNumber'])
    downsample_rate = str(request.form['downsampleRate'])

    processes[uuid] = filename

    debug_command = [
        'python3',
        'main.py',
        '--work-dir', "static/upload/{}/".format(uuid),
        '-f', filename,
        '-o', output_filename,
        '-d', downsample_rate,
        '-a', alg,
        '-b', bpm,
        '-bw', bpmWin,
        '-t', threshold,
        '-l', sample_size,
        '-r', rounding,
        '-w',
        '-v'
    ]

    if request.form['webMode']:
        web_mode = [
            '-x', str(request.form['xwide']),
            '-y', str(request.form['yhigh'])
            ]
        debug_command = debug_command + web_mode


    print("Running {}".format(debug_command))

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
        time.sleep(1)

    return redirect(url_for('result', uuid=uuid, output_filename=output_filename))

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

@app.route('/summaries/<uuid>')
def get_previous_summaries(uuid):
    summaries = []
    pattern = os.path.join('static', 'upload', uuid, 'summary-*.html')
    for filename in glob.glob(pattern):
        display_name = os.path.basename(filename).replace('summary-', '').replace('.html', '')
        summaries.append({"filename": filename, "display_name": display_name})
    return jsonify(summaries)

def save_processes_to_file(processes):
    with open('processes.json', 'w') as f:
        json.dump(processes, f)

def load_processes_from_file():
    try:
        with open('processes.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static/assets'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

def create_app(logger_override=None):
    app = Flask(__name__)

    if logger_override:
        # working solely with the flask logger
        app.logger.handlers = logger_override.handlers
        app.logger.setLevel(logger_override.level)

        # for logger in (app.logger, logger.getLogger('main')):
        #     logger.handlers = logger_override.handlers
        #     logger.setLevel(logger_override.level)

    return app

if __name__ == '__main__':
    app.run(debug=(os.getenv("DEBUG_MODE", "False") == "True"))
    processes = load_processes_from_file()

