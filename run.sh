#!/usr/bin/env bash
set -e

#echo "[1/3] Create venv"
#python -m venv venv
## shellcheck disable=SC1091
#source venv/bin/activate
#
#echo "[2/3] Install requirements"
#pip install --upgrade pip
#pip install -r requirements.txt
#
#echo "[3/3] Run server"
uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload
