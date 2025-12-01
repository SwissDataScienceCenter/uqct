ARG BASE_IMAGE="nvidia/cuda:13.0.1-base-ubuntu24.04"
ARG PYTHON_VERSION=3.12


# Stage 1: generate requirements-base.txt from pyproject.toml
# Here we generate a list of base dependencies which we expect to change less frequently.
# In the main image, we will mount and install the list of base dependencies.
# This way, the layer containing the base dependencies can be cached if there are no changes.
FROM python:3.12-slim as generator
ARG PYTHON_VERSION

WORKDIR /src
RUN --mount=from=ghcr.io/astral-sh/uv:latest,source=/uv,target=/bin/uv \
    --mount=type=bind,source=pyproject.toml,target=/src/pyproject.toml \
    uv pip compile --python ${PYTHON_VERSION} --emit-index-url --group base -o /src/requirements-base.txt

# Stage 2: main image
FROM ${BASE_IMAGE}
ARG USER="user"
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG PYTHON_VERSION

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl file gcc git git-lfs htop libgl1 libglib2.0-0 ncdu openssh-client openssh-server psmisc rsync screen sudo tmux unzip vim wget nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

# install powerline-go
ARG POWERLINE_GO_VERSION=1.25
RUN set -eu; \
    uname_arch="$(uname -m)"; \
    case "${uname_arch}" in \
      x86_64) arch=amd64 ;; \
      aarch64|arm64) arch=arm64 ;; \
      *) echo "Unsupported arch: ${uname_arch}" >&2; exit 1 ;; \
    esac; \
    url="https://github.com/justjanne/powerline-go/releases/download/v${POWERLINE_GO_VERSION}/powerline-go-linux-${arch}"; \
    curl -fsSL "${url}" -o /usr/local/bin/powerline-go; \
    chmod a+x /usr/local/bin/powerline-go
    
# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# uv: allow downloading Python builds and set default link mode
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/home/${USER}/.venv

# handle the USER setup
RUN \
    # Rename existing group with matching GID to desired group name (if exists)
    existing_group=$(getent group ${GROUP_ID} | cut -d: -f1) && \
    if [ -n "$existing_group" ] && [ "$existing_group" != "${USER}" ]; then \
    groupmod -n ${USER} "$existing_group"; \
    elif ! getent group ${GROUP_ID} > /dev/null; then \
    groupadd -g ${GROUP_ID} ${USER}; \
    fi && \
    \
    # Rename existing user with matching UID to desired username (if exists)
    existing_user=$(getent passwd ${USER_ID} | cut -d: -f1) && \
    if [ -n "$existing_user" ] && [ "$existing_user" != "${USER}" ]; then \
    usermod -l ${USER} "$existing_user" && usermod -d /home/${USER} -m ${USER}; \
    elif ! getent passwd ${USER_ID} > /dev/null; then \
    useradd -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash -m ${USER}; \
    fi && \
    \
    # Add user to extra groups
    usermod -a -G users ${USER}

# passwordless sudo
RUN echo "${USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Prepare user-owned SSH layout and host keys
# Generate host keys as root, then copy into a user-owned tree
RUN mkdir -p /home/user/.config/sshd /home/user/.local/state/sshd  /home/user/.cache/sshd/session && \
    printf '%s\n' \
    'Port 2222' \
    'PidFile /home/user/.local/state/sshd/sshd.pid' \
    'PasswordAuthentication no' \
    'KbdInteractiveAuthentication no' \
    'ChallengeResponseAuthentication no' \
    'UsePAM no' \
    'PermitRootLogin no' \
    'PubkeyAuthentication yes' \
    'AllowTcpForwarding yes' \
    'AllowAgentForwarding yes' \
    'HostKey /myhome/.local/share/sshd/host_rsa_key' \
    'HostKey /myhome/.local/share/sshd/host_ed25519_key' \
    > /home/user/.config/sshd/sshd_config && \
    chown -R user:user /home/user/.config /home/user/.local /home/user/.cache && \
    chmod 700  /home/user/.config/sshd /home/user/.local/state/sshd /home/user/.cache/sshd/session && \
    chmod 600 /home/user/.config/sshd/sshd_config


# add /entrypoint.sh script
COPY entrypoint.sh /entrypoint.sh
RUN chown ${USER}:${USER} /entrypoint.sh && chmod +x /entrypoint.sh && \
    mkdir /scratch && chown ${USER}:${USER} /scratch

# create workspace
WORKDIR /home/${USER}

# switch to user and create the project virtual environment
USER ${USER}
WORKDIR /home/${USER}
ENV PATH="/usr/local/bin:${PATH}"

# copy compiled requirements from uv-base and install base deps
RUN --mount=type=bind,from=generator,source=/src/requirements-base.txt,target=/tmp/requirements-base.txt \
    uv venv --python ${PYTHON_VERSION} && uv pip install -r /tmp/requirements-base.txt

# keep syncing project deps from pyproject.toml via BuildKit mount
RUN --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --python ${PYTHON_VERSION} --no-install-project --dev

# ensure .bash_profile sources .bashrc
RUN echo 'if [ -f ~/.bashrc ]; then\n    . ~/.bashrc\nfi' >> /home/${USER}/.bash_profile

# write .bashrc
RUN printf '%s\n' \
    'if [[ "$POWERLINE_SHELL_DISABLE" != "1" ]]; then' \
    '    function _update_ps1() {' \
    '        # shellcheck disable=SC2046' \
    '        PS1="$(/usr/local/bin/powerline-go -error $? -jobs $(jobs -p | wc -l) -mode compatible -modules ssh,venv,cwd,git,root)"' \
    '    }' \
    '' \
    '    if [ "$TERM" != "linux" ] && [ -f "/usr/local/bin/powerline-go" ]; then' \
    '        PROMPT_COMMAND="_update_ps1; $PROMPT_COMMAND"' \
    '    fi' \
    'fi' \
    "source /home/${USER}/.venv/bin/activate" \
    'if [ -f /myhome/.bashrc ]; then' \
    '    . /myhome/.bashrc' \
    'fi' \
    >> /home/${USER}/.bashrc

EXPOSE 2222
ENTRYPOINT ["/entrypoint.sh"]
