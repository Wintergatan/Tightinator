FROM python:latest

WORKDIR /opt

# Add requirements and install dependencies
COPY requirements.txt /opt/requirements.txt
RUN pip3 install -r requirements.txt

# Copy files
COPY web /opt/web/
COPY main.py /opt/web/main.py

# We'll run everything under a regular user acct, not root.
RUN adduser wilson --home /opt --disabled-password
RUN chown wilson:wilson /opt -R
USER wilson

# Start up the app!
WORKDIR /opt/web
EXPOSE 5000/tcp
CMD [ "gunicorn", "--bind", "127.0.0.1:5000", "app:app"]
