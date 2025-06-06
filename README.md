# nueral-networks

This repository includes example neural network scripts.

## Git setup helper

Run `setup_git.sh` in this directory to configure Git with your username, email, and remote. You'll still need to provide your GitHub username and personal access token the first time you push.

## Environment variables

Source `env_paths.sh` to populate environment variables for each example script.
```bash
source env_paths.sh
```
After sourcing, variables like `$CAT_DOG_CLASSIFIER2_PY` or `$VAE_PY` will hold the absolute path to the corresponding Python script. This lets you quickly run any script:

```bash
python "$VAE_PY"
```

