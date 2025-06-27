#!/bin/bash
pip install -r requirements.txt
export ENV=${1:-stage}
uvicorn app:app --host 0.0.0.0 --port 8000 --reload &
python3 -m kafka.consumers.frames_consumer &

