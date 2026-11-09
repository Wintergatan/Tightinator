import os
import sys
import glob
import time
import uuid as uuidlib
import subprocess
import threading
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, jsonify, abort
from werkzeug.utils import secure_filename

# Only static/assets is served as public static content. User uploads and
# results live under static/upload and are served only through the routes
# below, which validate the job ID first.
app = Flask(__name__, static_folder='static/assets', static_url_path='/static/assets')

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'upload')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Reject uploads over 150 MB (returns HTTP 413).
app.config['MAX_CONTENT_LENGTH'] = 150 * 1024 * 1024

MAIN_SCRIPT = os.path.join(app.root_path, 'main.py')

# Analysis jobs are tracked in-process. The Docker image runs a single
# gunicorn worker, so these limits are global for the container.
MAX_CONCURRENT_JOBS = int(os.getenv('MAX_CONCURRENT_JOBS', '2'))
JOB_TIMEOUT_SECONDS = int(os.getenv('JOB_TIMEOUT_SECONDS', '900'))
_jobs = {}
_jobs_lock = threading.Lock()

MAX_NAME_LENGTH = 200


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def is_valid_uuid(value):
    try:
        return str(uuidlib.UUID(value)) == value
    except (ValueError, TypeError, AttributeError):
        return False


def job_dir(job_id):
    """Return the directory for a job, or 404 if the ID is malformed or unknown."""
    if not is_valid_uuid(job_id):
        abort(404)
    path = os.path.join(UPLOAD_FOLDER, job_id)
    if not os.path.isdir(path):
        abort(404)
    return path


def clean_filename(name, extension):
    """Reduce a user-supplied name to a safe basename with the given extension."""
    name = secure_filename(name or '')
    if not name.lower().endswith(extension):
        name = name + extension
    if name == extension or len(name) > MAX_NAME_LENGTH:
        abort(400)
    return name


def uploaded_wav(path):
    """The single .wav file that was uploaded into a job directory."""
    wavs = sorted(f for f in os.listdir(path) if f.lower().endswith('.wav'))
    if not wavs:
        abort(404)
    return wavs[0]


def output_csv_name(wav_name):
    return clean_filename(wav_name[:-4], '.csv')


def form_number(name, cast, default, low, high):
    """Parse a numeric form field, falling back to the default when blank."""
    raw = request.form.get(name, '').strip()
    if raw == '':
        return default
    try:
        value = cast(raw)
    except ValueError:
        abort(400)
    if not (low <= value <= high):  # also rejects NaN
        abort(400)
    return value


# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------

def _watch_job(job_id, process):
    """Wait for a job so it doesn't linger as a zombie; kill it if it overruns."""
    try:
        process.wait(timeout=JOB_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        print("Job {} exceeded {}s, killing it".format(job_id, JOB_TIMEOUT_SECONDS))
        process.kill()
        process.wait()
    finally:
        with _jobs_lock:
            _jobs.pop(job_id, None)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not file.filename or not file.filename.lower().endswith('.wav'):
            return jsonify({'status': 'error', 'error': 'Please upload a .wav file.'}), 400

        filename = clean_filename(file.filename, '.wav')
        unique_id = str(uuidlib.uuid4())
        path = os.path.join(UPLOAD_FOLDER, unique_id)
        os.makedirs(path)
        file.save(os.path.join(path, filename))

        return jsonify({'status': 'success', 'uuid': unique_id, 'filename': filename})

    return render_template('index.html')


@app.route('/config/<uuid>/<filename>', methods=['GET'])
def config(uuid, filename):
    path = job_dir(uuid)
    # The filename always comes from disk; the URL segment is only cosmetic.
    wav_name = uploaded_wav(path)
    return render_template('config.html', uuid=uuid, filename=wav_name, output_filename=output_csv_name(wav_name))


@app.route('/config/<uuid>/', methods=['GET'])
def rerun(uuid):
    path = job_dir(uuid)
    wav_name = uploaded_wav(path)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    summary = os.path.join(path, 'summary.html')
    if os.path.exists(summary):
        os.rename(summary, os.path.join(path, 'summary-{}.html'.format(timestamp)))

    return render_template('config.html', uuid=uuid, filename=wav_name, output_filename=output_csv_name(wav_name))


@app.route('/run/<uuid>/', methods=['POST'])
def run(uuid):
    path = job_dir(uuid)
    filename = uploaded_wav(path)
    output_filename = clean_filename(request.form.get('output_filename'), '.csv')

    bpm = form_number('bpm', float, 0.0, 0, 1000)
    bpm_win = form_number('bpm_win', float, 0.0, 0, 1000)
    chunk_size = form_number('chunk_size', float, 8.4, 0.1, 1000)
    threshold = form_number('threshold', float, 0.1, 0, 1)
    sample_size = form_number('sample', int, 100, 1, 100000)
    rounding = form_number('rounding', int, 6, 0, 15)
    norm_p = form_number('norm_p', float, 2.0, 0.1, 100)

    command = [
        sys.executable,
        MAIN_SCRIPT,
        '--work-dir', path + os.sep,
        '-f', filename,
        '-o', output_filename,
        '-b', str(bpm),
        '-bw', str(bpm_win),
        '-cz', str(chunk_size),
        '-t', str(threshold),
        '-l', str(sample_size),
        '-r', str(rounding),
        '-p', str(norm_p),
        '-w',
        '-v'
    ]

    if request.form.get('web_mode'):
        command += [
            '-w',
            '-x', str(form_number('xwide', int, 2000, 100, 20000)),
            '-y', str(form_number('yhigh', int, 1340, 300, 20000)),
        ]

    if request.form.get("correlation"):
        command.append('-cp')

    with _jobs_lock:
        if uuid in _jobs:
            return jsonify({'error': 'An analysis for this file is already running.'}), 409
        if len(_jobs) >= MAX_CONCURRENT_JOBS:
            return jsonify({'error': 'The server is busy. Please try again in a minute.'}), 429

        print("Running {}".format(command))
        try:
            process = subprocess.Popen(command, text=True)
        except OSError as e:
            print("Failed to start job {}: {}".format(uuid, e))
            return jsonify({'error': 'Could not start the analysis.'}), 500
        _jobs[uuid] = process

    threading.Thread(target=_watch_job, args=(uuid, process), daemon=True).start()
    return redirect(url_for('running', uuid=uuid, output_filename=output_filename))


@app.route('/running/<uuid>/<output_filename>')
def running(uuid, output_filename):
    path = job_dir(uuid)
    output_filename = clean_filename(output_filename, '.csv')

    if not os.path.exists(os.path.join(path, output_filename)):
        return render_template('running.html', uuid=uuid, output_filename=output_filename)

    return redirect(url_for('result', uuid=uuid, output_filename=output_filename))


@app.route('/result/<uuid>/<output_filename>')
def result(uuid, output_filename):
    job_dir(uuid)
    output_filename = clean_filename(output_filename, '.csv')
    failure = request.args.get('failure', default=False, type=bool)
    return render_template('result.html', uuid=uuid, output_filename=output_filename, failure=failure)


@app.route('/download/<uuid>/<output_filename>')
def download(uuid, output_filename):
    path = job_dir(uuid)
    # send_from_directory refuses names that escape the directory.
    if not os.path.isfile(os.path.join(path, secure_filename(output_filename))):
        return "Long request", 503
    return send_from_directory(path, secure_filename(output_filename), as_attachment=True)


@app.route('/files/<uuid>/<name>')
def job_file(uuid, name):
    """Serve a job's plot pages (summary.html and earlier summary-*.html) inline."""
    path = job_dir(uuid)
    if not (name == 'summary.html' or (name.startswith('summary-') and name.endswith('.html'))):
        abort(404)
    return send_from_directory(path, name)


@app.route('/summaries/<uuid>')
def get_previous_summaries(uuid):
    path = job_dir(uuid)
    summaries = []
    for filename in sorted(glob.glob(os.path.join(glob.escape(path), 'summary-*.html'))):
        name = os.path.basename(filename)
        display_name = name.replace('summary-', '').replace('.html', '')
        summaries.append({"url": url_for('job_file', uuid=uuid, name=name), "display_name": display_name})
    return jsonify(summaries)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static/assets'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


# Bokeh plots load from cdn.bokeh.org and compile CustomJS callbacks at
# runtime ('unsafe-eval'); the templates use inline scripts ('unsafe-inline').
CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.bokeh.org",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob: https://github.githubassets.com",
    "font-src 'self' data:",
    "connect-src 'self'",
    "worker-src 'self' blob:",
    "frame-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'self'",
])


@app.after_request
def set_security_headers(response):
    response.headers.setdefault('Content-Security-Policy', CONTENT_SECURITY_POLICY)
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'SAMEORIGIN')
    response.headers.setdefault('Referrer-Policy', 'same-origin')
    return response


def create_app(logger_override=None):
    app = Flask(__name__)

    if logger_override:
        # working solely with the flask logger
        app.logger.handlers = logger_override.handlers
        app.logger.setLevel(logger_override.level)

    return app


if __name__ == '__main__':
    app.run(debug=(os.getenv("DEBUG_MODE", "False") == "True"))
