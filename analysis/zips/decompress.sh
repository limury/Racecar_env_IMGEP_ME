#!/bin/bash

f=$1
tar xvzf $f -C ../decompress
cd ../decompress
./rename.sh
cd ../zips
