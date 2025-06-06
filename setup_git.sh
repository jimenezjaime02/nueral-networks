#!/bin/bash
# Setup Git identity, credential helper, and remote.

git config --global user.name "jimenezjaime02"
git config --global user.email "jimenezjaime02@gmail.com"
# Store credentials so you don't have to re-enter them after the first push
# (you will still need to supply your GitHub username and personal access token once)
git config --global credential.helper store

# Set the remote named origin, replacing if it already exists
if git remote | grep -q '^origin$'; then
    git remote set-url origin https://github.com/jimenezjaime02/nueral-networks.git
else
    git remote add origin https://github.com/jimenezjaime02/nueral-networks.git
fi

echo "Git configuration complete. Remote 'origin' set to https://github.com/jimenezjaime02/nueral-networks.git"

