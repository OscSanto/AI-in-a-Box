cp ../public/index.html ../webui/index.html;
rm -rf ../webui/_app;
cp -r ../public/_app ../webui/_app;
rm -rf ../public/_app;
rm -f ../public/favicon.svg;
rm -f ../public/index.html;
echo "Copying output..."
echo "Done. Restart app.py to see changes."
