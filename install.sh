#!/bin/sh
set -eu

# Lemma CLI Standalone Installer
# Repository: https://github.com/arvarik/lemma

REPO="arvarik/lemma"
BINARY_NAME="lemma"

# Color support check
if [ -t 1 ]; then
  BOLD="$(printf '\033[1m')"
  GREEN="$(printf '\033[32m')"
  YELLOW="$(printf '\033[33m')"
  RED="$(printf '\033[31m')"
  RESET="$(printf '\033[0m')"
else
  BOLD=""
  GREEN=""
  YELLOW=""
  RED=""
  RESET=""
fi

log_info() {
  printf "%s==>%s %s%s\n" "$GREEN" "$RESET" "$BOLD" "$1" "$RESET"
}

log_warn() {
  printf "%sWarning:%s %s\n" "$YELLOW" "$RESET" "$1" >&2
}

log_error() {
  printf "%sError:%s %s\n" "$RED" "$RESET" "$1" >&2
}

# 1. Detect Operating System
OS="$(uname -s)"
case "$OS" in
  Linux*)  OS_TARGET="unknown-linux-gnu" ;;
  Darwin*) OS_TARGET="apple-darwin" ;;
  *)
    log_error "Unsupported operating system: $OS"
    exit 1
    ;;
esac

# 2. Detect Architecture
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64)   ARCH_TARGET="x86_64" ;;
  aarch64|arm64)  ARCH_TARGET="aarch64" ;;
  *)
    log_error "Unsupported CPU architecture: $ARCH"
    exit 1
    ;;
esac

TARGET="${ARCH_TARGET}-${OS_TARGET}"
log_info "Detected platform: ${TARGET}"

# 3. Detect Download Tool (curl or wget)
download_file() {
  _url="$1"
  _output="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$_url" -o "$_output"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$_output" "$_url"
  else
    log_error "Neither curl nor wget found. Please install one to continue."
    exit 1
  fi
}

# 4. Resolve Target Version
if [ -z "${VERSION:-}" ]; then
  log_info "Resolving latest release tag from GitHub..."
  if command -v curl >/dev/null 2>&1; then
    REDIRECT_URL="$(curl -fsSLI -o /dev/null -w "%{url_effective}" "https://github.com/${REPO}/releases/latest")"
  else
    REDIRECT_URL="$(wget -S --spider --max-redirect=0 "https://github.com/${REPO}/releases/latest" 2>&1 | grep -i "Location:" | awk '{print $2}' | tr -d '\r')"
  fi
  VERSION="${REDIRECT_URL##*/}"
  if [ -z "$VERSION" ] || [ "$VERSION" = "latest" ]; then
    log_error "Failed to resolve latest release tag for ${REPO}."
    exit 1
  fi
fi

# Ensure version format starts with 'v'
case "$VERSION" in
  v*) TAG="$VERSION" ;;
  *)  TAG="v${VERSION}" ;;
esac

TARBALL="${BINARY_NAME}-${TARGET}.tar.gz"
CHECKSUM_FILE="${TARBALL}.sha256"
DOWNLOAD_BASE="https://github.com/${REPO}/releases/download/${TAG}"

# 5. Create Secure Temporary Directory
TMP_DIR="$(mktemp -d 2>/dev/null || mktemp -d -t 'lemma-install')"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT INT TERM

# 6. Download Tarball & Checksum
log_info "Downloading ${TARBALL} (${TAG})..."
download_file "${DOWNLOAD_BASE}/${TARBALL}" "${TMP_DIR}/${TARBALL}"

log_info "Downloading ${CHECKSUM_FILE}..."
download_file "${DOWNLOAD_BASE}/${CHECKSUM_FILE}" "${TMP_DIR}/${CHECKSUM_FILE}"

# 7. Checksum Validation (POSIX compatible for Linux & macOS)
compute_sha256() {
  _target_file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$_target_file" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$_target_file" | awk '{print $1}'
  else
    log_warn "Neither sha256sum nor shasum found; skipping checksum validation."
    return 0
  fi
}

log_info "Verifying SHA-256 checksum..."
EXPECTED_HASH="$(awk '{print $1}' "${TMP_DIR}/${CHECKSUM_FILE}")"
ACTUAL_HASH="$(compute_sha256 "${TMP_DIR}/${TARBALL}")"

if [ -n "$ACTUAL_HASH" ]; then
  if [ "$EXPECTED_HASH" != "$ACTUAL_HASH" ]; then
    log_error "Checksum verification failed!"
    printf "  Expected: %s\n" "$EXPECTED_HASH" >&2
    printf "  Actual:   %s\n" "$ACTUAL_HASH" >&2
    exit 1
  fi
  log_info "Checksum verified successfully (${ACTUAL_HASH})"
fi

# 8. Unpack Binary
log_info "Extracting archive..."
tar -xzf "${TMP_DIR}/${TARBALL}" -C "${TMP_DIR}"

UNPACKED_BIN="$(find "${TMP_DIR}" -type f -name "${BINARY_NAME}" | head -n 1)"
if [ -z "$UNPACKED_BIN" ] || [ ! -f "$UNPACKED_BIN" ]; then
  log_error "Could not find '${BINARY_NAME}' executable in downloaded archive."
  exit 1
fi
chmod +x "$UNPACKED_BIN"

# 9. Determine Destination Directory
if [ -n "${BIN_DIR:-}" ]; then
  DEST_DIR="$BIN_DIR"
elif [ -w "/usr/local/bin" ] || [ "$(id -u)" -eq 0 ]; then
  DEST_DIR="/usr/local/bin"
else
  DEST_DIR="${HOME}/.local/bin"
fi

mkdir -p "$DEST_DIR"
DEST_PATH="${DEST_DIR}/${BINARY_NAME}"

log_info "Installing ${BINARY_NAME} to ${DEST_PATH}..."
mv "$UNPACKED_BIN" "$DEST_PATH"

# 10. Verification and PATH check
log_info "${BINARY_NAME} installed successfully!"
if "$DEST_PATH" --version >/dev/null 2>&1; then
  INSTALLED_VERSION="$("$DEST_PATH" --version)"
  printf "%s\n" "$INSTALLED_VERSION"
fi

case ":${PATH}:" in
  *:"${DEST_DIR}":*) ;;
  *)
    log_warn "'${DEST_DIR}' is not in your PATH."
    printf "Add it to your shell configuration file:\n"
    printf "  export PATH=\"%s:\$PATH\"\n\n" "$DEST_DIR"
    ;;
esac

log_info "Run '${BINARY_NAME} --help' to get started."
