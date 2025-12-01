#!/bin/bash
set -e
if [ $# -eq 0 ]; then
    # Symlink /myhome/.ssh to /home/user/.ssh
    if [ -e "/myhome/.ssh" ] && [ ! -e "/home/user/.ssh" ]; then
        ln -s /myhome/.ssh /home/user/.ssh
        echo "Symlinked /myhome/.ssh to /home/user/.ssh"
    fi
    # Download and extract VS Code server if not present
    VSCODE_SERVER_VERSION="stable"
    VSCODE_SERVER_DIR="/home/user/.vscode-server/bin/${VSCODE_SERVER_VERSION}"
    if [ ! -d "$VSCODE_SERVER_DIR" ]; then
        VSCODE_ARCH="$(uname -m)"
        case "${VSCODE_ARCH}" in
          x86_64) VSCODE_ARCH="x64";;
          aarch64|arm64) VSCODE_ARCH="arm64";;
        esac
        VSCODE_TAR="vscode-server-linux-${VSCODE_ARCH}.tar.gz"
        DOWNLOAD_URL="https://update.code.visualstudio.com/latest/server-linux-${VSCODE_ARCH}/stable"
        curl -fsSL "$DOWNLOAD_URL" -o "/tmp/${VSCODE_TAR}"
        mkdir -p "$VSCODE_SERVER_DIR"
        tar --strip-components=1 -xzf "/tmp/${VSCODE_TAR}" -C "$VSCODE_SERVER_DIR"
        rm "/tmp/${VSCODE_TAR}"
        chmod -R 755 /home/user/.vscode-server
    fi
    if [ -e "/myhome/vscode-extensions.txt" ]; then
        SERVER_EXTENSIONS_DIR="/home/user/.vscode-server/extensions"
        VSCODE_BIN="${VSCODE_SERVER_DIR}/bin/code-server"
        mkdir -p "$SERVER_EXTENSIONS_DIR"
        while IFS= read -r ext || [ -n "$ext" ]; do
            echo "Installing VS Code extension: $ext"
            "$VSCODE_BIN" \
                --extensions-dir "$SERVER_EXTENSIONS_DIR" \
                --install-extension "$ext" || true
        done < /myhome/vscode-extensions.txt
    fi
    exec /usr/sbin/sshd -D -e -f /home/user/.config/sshd/sshd_config || exec /bin/bash
else
    source /home/user/.venv/bin/activate
    exec "$@"
fi
