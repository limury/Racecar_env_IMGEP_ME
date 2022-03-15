#!/bin/bash

cd ../src

./test_random.py ./results/ 100000 100 10000 curiosity polynomial
cp -r results/ ../analysis/decompress/
cd ../analysis/decompress
mv results/ IMGEP-results
./rename.sh
cd ..
python analysis.py
python graph.py
cd ../src
