#!/bin/bash
#cd /app
#PYTHONPATH=/app python3 model/train.py &
#PYTHONUNBUFFERED='True' python3 /backend/main.py &
sed -i -e "s/http:\/\/localhost:8080/http:\/\/keystrokedetector/g" /usr/local/apache2/htdocs/main.dart.js
sed -i -e "s/Listen 80/Listen 8000/g" /usr/local/apache2/conf/httpd.conf
cd /backend
/usr/local/bin/uwsgi --pythonpath /ai_model --ini /backend/wsgi.ini &
nginx &
httpd-foreground