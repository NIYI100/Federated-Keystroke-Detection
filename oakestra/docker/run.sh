#!/bin/bash
#cd /app
#PYTHONPATH=/app python3 model/train.py &
python3 /backend/main.py &
httpd-foreground