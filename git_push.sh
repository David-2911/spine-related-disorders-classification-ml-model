#!/bin/bash
# Script to add, commit, and push changes to git

# Check if a commit message was provided as an argument
if [ -z "$1" ]; then
  echo "Usage: ./git_push.sh \"Your commit message\""
  exit 1
fi

git add .
git commit -m "$1"
git push
