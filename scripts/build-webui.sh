#!/bin/bash
# Rebuild the webui after editing source files in webui-src/src/
set -e

cd "$(dirname "$0")/../webui-src"

echo "Installing dependencies..."
npm install

echo "Building..."
npm run build

echo "Copying output..."
cp .svelte-kit/output/prerendered/pages/index.html ../webui/index.html
rm -rf ../webui/_app
cp -r .svelte-kit/output/client/_app ../webui/

echo "Done. Restart app.py to see changes."
