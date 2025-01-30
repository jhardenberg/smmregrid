#!/bin/bash

cdo -V
DIR=tests/data

cdo remapcon,r360x180 $DIR/onlytos-ipsl.nc $DIR/tos-ipsl-remapcon.nc
echo 'YES!'

cdo gencon,r360x180 $DIR/onlytos-ipsl.nc $DIR/weights.nc
cdo remap,r360x180,$DIR/weights.nc $DIR/onlytos-ipsl.nc $DIR/onlytos-ipsl-remap0.nc
echo 'YES!'

export REMAP_AREA_MIN=0.5
cdo remapcon,r360x180 $DIR/onlytos-ipsl.nc $DIR/tos-ipsl-remapcon.nc
echo 'YES!'
cdo gencon,r360x180 $DIR/onlytos-ipsl.nc $DIR/weights.nc
cdo remap,r360x180,$DIR/weights.nc $DIR/onlytos-ipsl.nc $DIR/onlytos-ipsl-remap5.nc
echo 'YES!'
