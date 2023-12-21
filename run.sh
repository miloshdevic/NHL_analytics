#!/bin/bash

# echo "TODO: fill in the docker run command"
docker run -p 8000:8000 --env COMET_API_KEY=$COMET_API_KEY -it model-ift6758