#!/bin/bash
set -e
sudo cp "$(dirname "$0")/aiiab.conf" /etc/nginx/conf.d/aiiab.conf
sudo nginx -t
sudo systemctl reload nginx
echo "App accessible at http://127.0.0.1/"
