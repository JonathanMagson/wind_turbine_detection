"""Retrying HTTP fetch for the public archives.

A single run makes thousands of requests against S3 and GCS, so a transient
mid-transfer failure is a matter of when rather than whether -- one
``IncompleteRead`` two hours into a five-AOI job loses the whole run.

404 is not retried: in these archives a missing key means the granule genuinely
lacks that file, which is a normal condition worth failing fast on.
"""

import http.client
import socket
import time
import urllib.error
import urllib.request

RETRYABLE = (urllib.error.URLError, http.client.IncompleteRead,
             http.client.RemoteDisconnected, ConnectionError, socket.timeout,
             TimeoutError)


def fetch(url, timeout=90, attempts=4, backoff=2.0):
    """Bytes at ``url``, retrying transient failures with exponential backoff."""
    last = None
    for i in range(attempts):
        try:
            return urllib.request.urlopen(url, timeout=timeout).read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            last = exc
        except RETRYABLE as exc:                             # noqa: PERF203
            last = exc
        if i < attempts - 1:
            time.sleep(backoff ** i)
    raise last


def fetch_text(url, **kw):
    return fetch(url, **kw).decode()
