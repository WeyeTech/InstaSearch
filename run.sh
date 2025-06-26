#!/bin/bash
pip install -r requirements.txt
export ENV=${1:-dev}
uvicorn app:app --host 0.0.0.0 --port 8000 --reload &
python kafka/consumers/frames_consumer.py &

