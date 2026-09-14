"""HTTP helpers shared by the services."""

import json
import logging

import requests
from urllib3.util import Retry

logger = logging.getLogger(__name__)


def make_session(retries: int = 10, backoff_factor: float = 0.1) -> requests.Session:
    """
    A requests.Session that retries GET and POST requests on 5xx responses and connection
    errors, with exponential backoff. Pass it to any service as `requests_session`.
    """
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods={"GET", "POST"},
    )
    session = requests.Session()
    session.mount("http://", requests.adapters.HTTPAdapter(max_retries=retry))
    session.mount("https://", requests.adapters.HTTPAdapter(max_retries=retry))
    return session


def forbidden(response: requests.Response, text, data=None) -> bool:
    """
    True if the response is an HTTP 403, after logging the text that triggered it.

    Some texts repeatably get a 403 from the RENCI ingress (probably a web application
    firewall rule). We log them so they can be reported, and callers return an empty
    result rather than abort a whole batch. Any other error status raises HTTPError.

    :param response: The response to check.
    :param text: The text (or identifiers) being annotated, for the log.
    :param data: The request payload, for the log.
    """
    if response.status_code == 403:
        logger.warning(
            f"HTTP 403 from {response.url} for text {json.dumps(text)}; request: {json.dumps(data)}"
        )
        return True
    response.raise_for_status()
    return False
