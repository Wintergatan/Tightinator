from app import app
import logging

gunicorn_logger = logging.getLogger('gunicorn.error')
application = create_app(logger_override=gunicorn_logger)

if __name__ == "__main__":
    application.run()
