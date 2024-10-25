#!/bin/sh

while inotifywait thesis.tex; do 
	optex thesis.tex;
done &

okular thesis.pdf &
