#!/bin/bash
#cd /app
#PYTHONPATH=/app python3 model/train.py &
PYTHONUNBUFFERED='True' python3 /backend/main.py &
httpd-foreground