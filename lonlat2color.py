#!/usr/bin/env python

#
# Copyright (c) 2012, USC/ISI
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: 
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer. 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution. 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import math

# hsl2rgb(h, s, l): convert from HSL to RGB
# h in [0..360]
# s in [0..1]
# l in [0..1]
def hsl2rgb(h, s, l):
  c  =(1.0 - abs(2*l-1))*s
  hh = h/60.0
  rgb = []
  x  = c*(1.0 - abs(hh - 2*int(hh/2.0) - 1.0))
  if 0<=hh and hh<1:
    rgb = [c, x, 0]
  if 1<=hh and hh<2:
    rgb = [x, c, 0]
  if 2<=hh and hh<3:
    rgb = [0, c, x]
  if 3<=hh and hh<4:
    rgb = [0, x, c]
  if 4<=hh and hh<5:
    rgb = [x, 0, c]
  if 5<=hh and hh<6:
    rgb = [c, 0, x]

  m = l-c/2.0
  r = int(255*(rgb[0]+m))
  g = int(255*(rgb[1]+m))
  b = int(255*(rgb[2]+m))
  return [r,g,b]

def hsl2rgba(h, s, l):
  c  =(1.0 - abs(2*l-1))*s
  hh = h/60.0
  rgb = []
  x  = c*(1.0 - abs(hh - 2*int(hh/2.0) - 1.0))
  if 0<=hh and hh<1:
    rgb = [c, x, 0]
  if 1<=hh and hh<2:
    rgb = [x, c, 0]
  if 2<=hh and hh<3:
    rgb = [0, c, x]
  if 3<=hh and hh<4:
    rgb = [0, x, c]
  if 4<=hh and hh<5:
    rgb = [x, 0, c]
  if 5<=hh and hh<6:
    rgb = [c, 0, x]

  m = l-c/2.0
  r = rgb[0]+m
  g = rgb[1]+m
  b = rgb[2]+m
  return [r,g,b,1]

# lonlat2rgb(lon, lat): convert coordinates to RGB triplet
# input:  latitude  in [ -90;  90]
# input:  longitude in [-180; 180]
# output: [r g b]
def lonlat2rgb(lon, lat):
  lat = -lat; #north should be dark
  lon = math.fmod(lon+3600, 360)
  (LATL, LATH) = (-115, 80)  #normalize to this range
  if lat > LATH:
    lat = LATH
  if lat < LATL:
    lat = LATL
  return hsl2rgb(lon, 1, (lat - LATL)/(LATH-LATL))

def lonlat2rgba(lon, lat):
  lat = -lat; #north should be dark
  lon = math.fmod(lon+3600, 360)
  (LATL, LATH) = (-115, 80)  #normalize to this range
  if lat > LATH:
    lat = LATH
  if lat < LATL:
    lat = LATL
  return hsl2rgba(lon, 1, (lat - LATL)/(LATH-LATL))
