#!/usr/bin/env python3

from threading import Lock
import subprocess
import signal
import requests
import time
from pathlib import Path


class ModelManager:

    def __init__(self, model_config):

        self.lock = Lock()
        self.config = model_config
        self.model_name = model_config["name"]
        self.process = None
        self.current_model = None

    ###############################################################

    def ensure(self):

        with self.lock:
            if (
                self.current_model == self.model_name
                and self.process is not None
                and self.process.poll() is None
            ):
                return

            self.stop()

            self.start()

    ###############################################################

    def start(self):

        command = self.build_command(self.config)

        print(f"[ModelManager] Starting {self.model_name}")

        self.process = subprocess.Popen(
            command,
        )


        self.wait_until_ready(self.config["port"])

        self.current_model = self.model_name

        print(f"[ModelManager] {self.model_name} ready.")

    ###############################################################

    def stop(self):

        if self.process is None:
            return

        print("[ModelManager] Stopping model...")

        self.process.send_signal(signal.SIGTERM)

        try:

            self.process.wait(timeout=20)

        except subprocess.TimeoutExpired:

            print("[ModelManager] Force killing model.")

            self.process.kill()

            self.process.wait()

        self.process = None
        self.current_model = None

    ###############################################################

    def wait_until_ready(self, port):

        url = f"http://{self.config['client_host']}:{port}/health"

        timeout = time.time() + 120

        while time.time() < timeout:

            if self.process.poll() is not None:

                raise RuntimeError("llama-server exited unexpectedly.")

            try:

                r = requests.get(url, timeout=1)

                if r.status_code == 200:
                    return

            except Exception:
                pass

            time.sleep(0.25)

        raise RuntimeError("Timed out waiting for llama-server.")

    ###############################################################

    def build_command(self, cfg):

        command = [

            str(Path(cfg["llama_dir"]) / "build/bin/llama-server"),

            "-m",
            cfg["model"],

            "--port",
            str(cfg["port"]),

            "--host",
            cfg["host"],

            "-c",
            str(cfg["ctx"]),

            "-ngl",
            str(cfg["ngl"])
        ]

        command.extend(cfg.get("args", []))

        return command

    ###############################################################

    def running(self):

        return self.current_model
