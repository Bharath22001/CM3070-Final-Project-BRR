"""
Checks that the system runs with no internet connection.

Every outbound connection and DNS lookup to anything other than this machine is
blocked while the real serving path runs. Loopback is allowed, because Ollama
is a local server on 127.0.0.1:11434. "Offline" means after setup: installing
packages, pulling the model and the checkpoints needed a network once.

The first test checks the guard itself by trying to reach a public address. If
it fails, ignore every other result in this file.

Run from the project root:
    python -m pytest tests/test_offline.py -q
"""

from __future__ import annotations

import io
import ipaddress
import socket
from contextlib import contextmanager

import numpy as np
import pytest

from cuneiform.api import service

# Reuse the same "is Ollama up" check as test_api, so there is only one definition.
from test_api import requires_ollama  # noqa: E402


class OutboundBlocked(OSError):
    """Raised instead of connecting out, so the caller sees a network failure."""


def _is_loopback(addr) -> bool:
    """Is this address this machine talking to itself?"""
    try:
        host = addr[0] if isinstance(addr, (tuple, list)) else addr
    except Exception:                                            # noqa: BLE001
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in ("localhost", "localhost.localdomain")


class NetworkGuard:
    """Record what was attempted; allow loopback; refuse everything else."""

    def __init__(self) -> None:
        self.loopback: list[str] = []
        self.blocked: list[str] = []

    @property
    def reached_out(self) -> bool:
        return bool(self.blocked)


@contextmanager
def no_network():
    """
    Block every non-loopback connection and DNS lookup for the duration, by
    patching socket.socket.connect and socket.getaddrinfo (httpx, requests and
    urllib3 all go through them). Restored in a finally so a failure can't
    break networking for later tests.
    """
    guard = NetworkGuard()
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def connect(self, address, *args, **kwargs):
        if _is_loopback(address):
            guard.loopback.append(str(address))
            return real_connect(self, address, *args, **kwargs)
        guard.blocked.append(f"connect {address!r}")
        raise OutboundBlocked(f"offline gate: refused connection to {address!r}")

    def getaddrinfo(host, *args, **kwargs):
        # A DNS lookup is an outbound packet too, and it's how ultralytics checks connectivity.
        if host is None or _is_loopback(host):
            return real_getaddrinfo(host, *args, **kwargs)
        guard.blocked.append(f"dns {host!r}")
        raise socket.gaierror(f"offline gate: refused DNS lookup for {host!r}")

    socket.socket.connect = connect
    socket.getaddrinfo = getaddrinfo
    try:
        yield guard
    finally:
        socket.socket.connect = real_connect
        socket.getaddrinfo = real_getaddrinfo


TABLET = "P335940"


# The guard, before anything that depends on it

def test_the_guard_actually_blocks():
    """
    If this fails, ignore every result below it. It tries the very thing the
    other tests say never happens and requires it to be refused and recorded.
    """
    with no_network() as guard:
        with pytest.raises((OutboundBlocked, socket.gaierror, OSError)):
            socket.getaddrinfo("example.com", 80)
        with pytest.raises((OutboundBlocked, OSError)):
            socket.create_connection(("93.184.216.34", 80), timeout=2)

    assert len(guard.blocked) == 2, guard.blocked
    assert any("example.com" in b for b in guard.blocked)

    # Loopback must still work, or the Domain 3 test would pass for the wrong reason.
    with no_network() as guard:
        assert _is_loopback(("127.0.0.1", 11434))
        assert not _is_loopback(("8.8.8.8", 53))
        assert guard.blocked == []


# Domains 1 and 2: the reading

def test_analyse_makes_no_outbound_connection():
    """The whole reading path with the network blocked, including the Domain 2 checkpoint load."""
    with no_network() as guard:
        service.warm()
        result = service.analyse_tablet(TABLET)

    assert guard.blocked == [], f"the reading path reached out: {guard.blocked}"
    assert result["counts"]["slots"] > 0
    assert result["lines"], "no reading produced, so nothing was really exercised"


# Domain 1: the detector

def test_detector_makes_no_outbound_connection():
    """
    Ultralytics looks up public DNS servers at import to decide on update
    checks and analytics. service._detector() sets YOLO_OFFLINE before the
    import so no packet is sent. I wrote this test first and saw it fail.
    """
    with no_network() as guard:
        model = service._detector()
        model.predict(
            source=[np.zeros((512, 512, 3), dtype=np.uint8)],
            imgsz=512, conf=0.05, device="cpu", verbose=False,
        )

    assert guard.blocked == [], (
        "the detector reached out: " + repr(guard.blocked) +
        " -- YOLO_OFFLINE is not taking effect before ultralytics is imported"
    )


# Domain 3: the language model

@requires_ollama
def test_summarise_makes_no_outbound_connection():
    """
    Domain 3 with the network blocked. I check both that nothing reached the
    internet and that the payload shows a real generation happened. I don't
    look for a new socket to Ollama, because httpx reuses pooled connections.
    """
    # regenerate=True skips the cache so this is a real model call.
    with no_network() as guard:
        payload, cached = service.summarise_tablet(TABLET, regenerate=True)

    assert guard.blocked == [], f"Domain 3 reached out: {guard.blocked}"
    assert not cached, "served from cache, so no generation was exercised"
    assert payload["summary"]["tentative_reading"].strip(), (
        "no prose came back, so the generative stage did not really run")
    assert payload["seconds"] > 0, (
        "a generation that took no measurable time did not happen")
    assert payload["attempts"] >= 1


# The whole promise, in one test

def test_an_upload_works_end_to_end_with_no_network():
    """
    Upload a photograph with no network: Domain 1 detects, Domain 2 fuses and
    nothing reaches the internet. Upload is the only route that runs
    ultralytics inside a request, so this is where a regression would show.
    """
    from PIL import Image

    from cuneiform import config

    src = Image.open(config.IMAGES_DIR / "P335940.jpg").convert("RGB")
    small = src.resize((src.width // 2, src.height // 2), Image.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=88)

    tablet = None
    try:
        with no_network() as guard:
            rec = service.ingest_upload(buf.getvalue(), "offline_test.jpg")
            tablet = rec["tablet"]
            result = service.analyse_tablet(tablet)

        assert guard.blocked == [], (
            "the upload path reached out: " + repr(guard.blocked))
        assert rec["n_detections"] > 0, "nothing was detected, so nothing ran"
        assert result["counts"]["slots"] > 0

        # And the 30 evaluation tablets are untouched.
        assert len(service.list_tablets()) == 30
    finally:
        if tablet:
            for p in config.UPLOADS_DIR.glob(tablet + ".*"):
                p.unlink(missing_ok=True)
