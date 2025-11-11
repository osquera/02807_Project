#!bin/bash

gcloud auth activate-service-account --key-file=google-service-account-key.json

ACCESS_TOKEN="$(gcloud auth print-access-token)"
cat > ~/.netrc <<EOF
machine europe-west6-python.pkg.dev
  login oauth2accesstoken
  password ${ACCESS_TOKEN}
EOF
chmod 600 ~/.netrc
