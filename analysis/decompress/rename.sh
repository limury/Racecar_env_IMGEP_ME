#!/bin/bash

i=0

dirs=$(find -name "racecar*" -type d)
for dir in $dirs        
do 
  cd $dir
  for f in *
  do
    mv -- "$f" "$i-$f"
  done
  cd ..
  ((i=i+1))

  mv $dir/* ../data/
  rm -rf $dir

done 
