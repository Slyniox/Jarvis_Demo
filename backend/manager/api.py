#!/usr/bin/env python3

from proxy import app, config, manager


@app.get("/status")
def status():

    return {

        "running": manager.running()

    }


@app.post("/stop")
def stop():

    manager.stop()

    return {

        "status": "stopped"

    }


@app.get("/models")
def models():

    return {

        "available":

            [config["llama"]["name"]]

    }
