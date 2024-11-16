#!/bin/sh

while inotifywait thesis.tex -e modify; do 
	optex thesis.tex;
done
