# utils.py - some utility functions for use across this application
import json
import logging

def log_http_403_errors(text: str, url: str = None, data = None, logger = None) -> None:
    """
    We sometimes have text that will repeatably trigger a 403 error (probably because it hits up against something
    in the RENCI ingress). This function is intended to log the text that triggered these errors so that we can
    complain about them.

    :param text: The text that is attempting to be annotated.
    :param url: The URL is being used to annotate the text.
    :param data: The full request to the server (if applicable/available).
    :param logger: The logger to use (will default to a logger named after this module).
    :return:
    """

    if not logger:
        logger = logging.getLogger(__name__)

    if not url:
        url = "unknown"

    if '"' in text:
        # Escape double quotes in the text.
        text = text.replace('"', '\\"')

    if data:
        logger.warning(
            f"Received HTTP 403 error when sending data to URL {url}: text=\"{text}\", data={json.dumps(data, indent=2)}"
        )
    else:
        logger.warning(
            f"Received HTTP 403 error when sending text to URL {url}: \"{text}\""
        )
