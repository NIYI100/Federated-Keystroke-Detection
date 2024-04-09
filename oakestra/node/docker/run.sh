#!/bin/bash
sed -i -e "s/http:\/\/localhost:8080/http:\/\/$HOSTNAME.oakestra/g" /usr/local/apache2/htdocs/main.dart.js
cd /backend
/usr/local/bin/uwsgi --pythonpath /ai_model --ini /backend/wsgi.ini &
nginx &
httpd-foreground