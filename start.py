#!/usr/bin/env python3
"""
Единая точка запуска ARTi Forge.
Поддерживает режимы:
  monolith – всё в одном процессе
  split    – локальные микросервисы + API Gateway
  gateway  – только API Gateway, обращается к внешним сервисам
"""
import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from urllib.parse import urljoin

import requests

# Временная настройка логирования (будет уточнена позже)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("launcher")


def wait_for_health(url, timeout=120, service_name="Service"):
    """Ожидание HTTP /health эндпоинта."""
    start = time.time()
    health_url = urljoin(url, "/health")
    logger.info("Ожидание готовности %s (%s)...", service_name, health_url)
    while time.time() - start < timeout:
        try:
            resp = requests.get(health_url, timeout=2)
            if resp.status_code == 200:
                logger.info("%s готов (статус %d)", service_name, resp.status_code)
                return True
            else:
                logger.debug("%s вернул статус %d, повтор через 2с", service_name, resp.status_code)
        except requests.RequestException as e:
            logger.debug("%s недоступен: %s", service_name, e)
        time.sleep(2)
    logger.error("%s не запустился за %d секунд", service_name, timeout)
    return False


def run_monolith(args):
    from config import settings  # импортируем уже загруженный объект настроек

    env = os.environ.copy()
    if args.profile:
        env["PROFILE"] = args.profile

    host = args.host or settings.host or "localhost"
    port = args.port or settings.port or 8000
    cmd = [sys.executable, "server.py", "--host", host, "--port", str(port)]
    logger.info("Запуск в монолитном режиме: %s", " ".join(str(x) for x in cmd))
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        logger.error("Монолит завершился с ошибкой: %s", e)
        sys.exit(e.returncode)


def run_split(args):
    from config import settings

    env = os.environ.copy()
    if args.profile:
        env["PROFILE"] = args.profile

    host = args.host or settings.host or "localhost"
    t2i_port = args.text2image_port or settings.text2image_port or 8001
    i2s_port = args.image2shape_port or settings.image2shape_port or 8002
    gateway_port = args.port or settings.port or 8000

    # Запускаем Text2Image микросервис
    t2i_cmd = [
        sys.executable, "microservices/text2image_server.py",
        "--host", host, "--port", str(t2i_port)
    ]
    logger.info("Запуск Text2Image: %s", " ".join(str(x) for x in t2i_cmd))
    t2i_proc = subprocess.Popen(t2i_cmd, env=env)

    # Запускаем Image2Shape микросервис
    i2s_cmd = [
        sys.executable, "microservices/image2shape_server.py",
        "--host", host, "--port", str(i2s_port)
    ]
    logger.info("Запуск Image2Shape: %s", " ".join(str(x) for x in i2s_cmd))
    i2s_proc = subprocess.Popen(i2s_cmd, env=env)

    t2i_url = f"http://{host}:{t2i_port}"
    i2s_url = f"http://{host}:{i2s_port}"

    if not wait_for_health(t2i_url, service_name="Text2Image"):
        logger.error("Text2Image не отвечает на /health, завершение")
        sys.exit(1)
    if not wait_for_health(i2s_url, service_name="Image2Shape"):
        logger.error("Image2Shape не отвечает на /health, завершение")
        sys.exit(1)

    # Запускаем API Gateway
    gw_env = env.copy()
    gw_env["TEXT2IMAGE_SERVICE_URL"] = t2i_url
    gw_env["IMAGE2SHAPE_SERVICE_URL"] = i2s_url
    gw_cmd = [sys.executable, "server.py", "--host", host, "--port", str(gateway_port)]
    logger.info("Запуск API Gateway: %s", " ".join(str(x) for x in gw_cmd))
    gw_proc = subprocess.Popen(gw_cmd, env=gw_env)

    def handler(sig, frame):
        logger.info("Получен сигнал %s, завершаем процессы...", sig)
        for p in (t2i_proc, i2s_proc, gw_proc):
            if p.poll() is None:
                p.terminate()
                logger.debug("Процесс %d завершён", p.pid)
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    try:
        while all(p.poll() is None for p in (t2i_proc, i2s_proc, gw_proc)):
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Получено прерывание от клавиатуры")
    finally:
        for p in (t2i_proc, i2s_proc, gw_proc):
            if p.poll() is None:
                logger.info("Принудительное завершение процесса %d", p.pid)
                p.terminate()


def run_gateway(args):
    from config import settings

    env = os.environ.copy()

    t2i_url = args.text2image_url or settings.text2image_url or env.get("TEXT2IMAGE_SERVICE_URL")
    i2s_url = args.image2shape_url or settings.image2shape_url or env.get("IMAGE2SHAPE_SERVICE_URL")

    if not t2i_url:
        logger.error("Не указан TEXT2IMAGE_SERVICE_URL")
        sys.exit(1)
    if not i2s_url:
        logger.error("Не указан IMAGE2SHAPE_SERVICE_URL")
        sys.exit(1)

    env["TEXT2IMAGE_SERVICE_URL"] = t2i_url
    env["IMAGE2SHAPE_SERVICE_URL"] = i2s_url

    host = args.host or settings.host or "localhost"
    port = args.port or settings.port or 8000
    cmd = [sys.executable, "server.py", "--host", host, "--port", str(port)]
    logger.info("Запуск API Gateway в режиме gateway: %s", " ".join(str(x) for x in cmd))
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        logger.error("API Gateway завершился с ошибкой: %s", e)
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(description="ARTi Forge Launcher")
    parser.add_argument("--mode", choices=["monolith", "split", "gateway"], default="monolith")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--text2image-port", type=int, default=None)
    parser.add_argument("--image2shape-port", type=int, default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--text2image-url", default=None)
    parser.add_argument("--image2shape-url", default=None)
    parser.add_argument("--log-level", default="DEBUG",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Уровень детализации логов")
    args = parser.parse_args()

    # Настройка логирования
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    global logger
    logger = logging.getLogger("launcher")
    logger.debug("Аргументы командной строки: %s", args)

    # Инициализируем глобальные настройки с указанным конфигом
    from config import Settings
    Settings(config_path=args.config)

    # Переопределяем настройки из аргументов командной строки (приоритет)
    from config import settings
    if args.host is not None:
        settings.host = args.host
    if args.port is not None:
        settings.port = args.port
    if args.text2image_port is not None:
        settings.text2image_port = args.text2image_port
    if args.image2shape_port is not None:
        settings.image2shape_port = args.image2shape_port
    if args.profile is not None:
        settings.profile = args.profile
    if args.text2image_url is not None:
        settings.text2image_url = args.text2image_url
    if args.image2shape_url is not None:
        settings.image2shape_url = args.image2shape_url

    # Запуск в выбранном режиме
    if args.mode == "monolith":
        run_monolith(args)
    elif args.mode == "split":
        run_split(args)
    elif args.mode == "gateway":
        run_gateway(args)
    else:
        logger.error("Неизвестный режим: %s", args.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()