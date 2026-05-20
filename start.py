#!/usr/bin/env python3
"""
Единая точка запуска ARTi Forge.
Поддерживает режимы:
  monolith – всё в одном процессе
  split    – локальные микросервисы + API Gateway
  gateway  – только API Gateway, обращается к внешним сервисам
"""
import argparse
import os
import subprocess
import sys
import time
import signal
import requests
from urllib.parse import urljoin

def wait_for_health(url, timeout=120):
    start = time.time()
    health_url = urljoin(url, "/health")
    while time.time() - start < timeout:
        try:
            resp = requests.get(health_url, timeout=2)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False

def run_monolith(args):
    env = os.environ.copy()
    if args.profile:
        env["PROFILE"] = args.profile
    cmd = [sys.executable, "server.py", "--host", args.host, "--port", str(args.port)]
    subprocess.run(cmd, env=env, check=True)

def run_split(args):
    env = os.environ.copy()
    if args.profile:
        env["PROFILE"] = args.profile

    t2i_port = args.text2image_port
    i2s_port = args.image2shape_port

    # Запускаем Text2Image микросервис
    t2i_cmd = [sys.executable, "microservices/text2image_server.py",
               "--host", args.host, "--port", str(t2i_port)]
    print(f"Запуск Text2Image на порту {t2i_port}...")
    t2i_proc = subprocess.Popen(t2i_cmd, env=env)

    # Запускаем Image2Shape микросервис
    i2s_cmd = [sys.executable, "microservices/image2shape_server.py",
               "--host", args.host, "--port", str(i2s_port)]
    print(f"Запуск Image2Shape на порту {i2s_port}...")
    i2s_proc = subprocess.Popen(i2s_cmd, env=env)

    t2i_url = f"http://{args.host}:{t2i_port}"
    i2s_url = f"http://{args.host}:{i2s_port}"

    print("Ожидание готовности Text2Image...")
    if not wait_for_health(t2i_url):
        print("Text2Image не запустился за отведённое время")
        sys.exit(1)
    print("Ожидание готовности Image2Shape...")
    if not wait_for_health(i2s_url):
        print("Image2Shape не запустился за отведённое время")
        sys.exit(1)

    # Запускаем API Gateway
    gw_env = env.copy()
    gw_env["TEXT2IMAGE_SERVICE_URL"] = t2i_url
    gw_env["IMAGE2SHAPE_SERVICE_URL"] = i2s_url
    gw_cmd = [sys.executable, "server.py", "--host", args.host, "--port", str(args.port)]
    print(f"Запуск API Gateway на порту {args.port}...")
    gw_proc = subprocess.Popen(gw_cmd, env=gw_env)

    def handler(sig, frame):
        print("\nЗавершение...")
        for p in [t2i_proc, i2s_proc, gw_proc]:
            p.terminate()
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    try:
        while all(p.poll() is None for p in [t2i_proc, i2s_proc, gw_proc]):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in [t2i_proc, i2s_proc, gw_proc]:
            if p.poll() is None:
                p.terminate()

def run_gateway(args):
    env = os.environ.copy()
    if not args.text2image_url and not os.environ.get("TEXT2IMAGE_SERVICE_URL"):
        print("Ошибка: укажите TEXT2IMAGE_SERVICE_URL")
        sys.exit(1)
    if not args.image2shape_url and not os.environ.get("IMAGE2SHAPE_SERVICE_URL"):
        print("Ошибка: укажите IMAGE2SHAPE_SERVICE_URL")
        sys.exit(1)

    env["TEXT2IMAGE_SERVICE_URL"] = args.text2image_url or env.get("TEXT2IMAGE_SERVICE_URL")
    env["IMAGE2SHAPE_SERVICE_URL"] = args.image2shape_url or env.get("IMAGE2SHAPE_SERVICE_URL")

    cmd = [sys.executable, "server.py", "--host", args.host, "--port", str(args.port)]
    subprocess.run(cmd, env=env, check=True)

def main():
    parser = argparse.ArgumentParser(description="ARTi Forge Launcher")
    parser.add_argument("--mode", choices=["monolith", "split", "gateway"], default="monolith")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--text2image-port", type=int, default=8001)
    parser.add_argument("--image2shape-port", type=int, default=8002)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--text2image-url", default=None)
    parser.add_argument("--image2shape-url", default=None)
    args = parser.parse_args()

    if args.mode == "monolith":
        run_monolith(args)
    elif args.mode == "split":
        run_split(args)
    elif args.mode == "gateway":
        run_gateway(args)

if __name__ == "__main__":
    main()