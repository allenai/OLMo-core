#!/usr/bin/env python3
"""Minimal threaded HTTP/HTTPS (CONNECT) forward proxy.

Run on a host WITH internet (hermione); reverse-tunnel its port onto an
air-gapped host (lambda) so pip/huggingface/git there can egress through here.

Usage: python http_connect_proxy.py [bind_host] [port]   (default 127.0.0.1 8899)
"""
import socket
import select
import sys
import threading
from urllib.parse import urlsplit


def pipe(a, b):
    socks = [a, b]
    try:
        while True:
            r, _, _ = select.select(socks, [], [], 120)
            if not r:
                break
            for s in r:
                data = s.recv(65536)
                if not data:
                    return
                (b if s is a else a).sendall(data)
    finally:
        for s in socks:
            try:
                s.close()
            except OSError:
                pass


def handle(client):
    try:
        client.settimeout(60)
        req = b""
        while b"\r\n\r\n" not in req:
            chunk = client.recv(4096)
            if not chunk:
                client.close()
                return
            req += chunk
        line = req.split(b"\r\n", 1)[0].decode("latin1")
        method, target, _ = line.split(" ")
        if method == "CONNECT":
            host, port = target.rsplit(":", 1)
            remote = socket.create_connection((host, int(port)), timeout=30)
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            pipe(client, remote)
        else:
            u = urlsplit(target)
            host = u.hostname
            port = u.port or 80
            path = u.path or "/"
            if u.query:
                path += "?" + u.query
            remote = socket.create_connection((host, port), timeout=30)
            rest = req.split(b"\r\n", 1)[1]
            remote.sendall(f"{method} {path} HTTP/1.1\r\n".encode("latin1") + rest)
            pipe(client, remote)
    except Exception:
        try:
            client.close()
        except OSError:
            pass


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8899
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(256)
    print(f"http-connect proxy listening on {host}:{port}", flush=True)
    while True:
        c, _ = srv.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()


if __name__ == "__main__":
    main()
