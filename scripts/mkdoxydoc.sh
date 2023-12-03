#!/bin/bash

# Default theme version
DEFAULT_VERSION="0.0.2"

# Take theme version as an input argument, or use the default
THEME_VERSION="${1:-$DEFAULT_VERSION}"

# Define variables
ROOT_DIR=$(pwd)
DOC_DIR="${ROOT_DIR}/doc/apidocs"
THEME_URL="https://github.com/HaoZeke/doxyYoda/releases/download/${THEME_VERSION}/doxyYoda_${THEME_VERSION}.tar.gz"
THEME_TAR="${DOC_DIR}/doxyYoda_${THEME_VERSION}.tar.gz"
THEME_DIR="${DOC_DIR}/doxyYoda"
CFG_FILE="${DOC_DIR}/Doxygen-proj.cfg"
TAGS_DIR="${DOC_DIR}/tags"
CPP_REF_TAG_URL="https://upload.cppreference.com/mwiki/images/f/f8/cppreference-doxygen-web.tag.xml"
CPP_REF_TAG_FILE="${TAGS_DIR}/cppreference-doxygen-web.tag.xml"

# Ensure the documentation directory exists
if [ ! -d "$DOC_DIR" ]; then
    echo "Creating documentation directory..."
    mkdir -p "$DOC_DIR"
fi

cd "$DOC_DIR" || {
    echo "Error: Failed to navigate to documentation directory"
    exit 1
}

# Check if the theme is already extracted
if [ -d "$THEME_DIR" ]; then
    echo "Theme already downloaded and extracted."
else
    # Check if the tarball already exists
    if [ -f "$THEME_TAR" ]; then
        echo "Theme tarball already exists. Skipping download."
    else
        # Download the theme tarball
        echo "Downloading theme version ${THEME_VERSION}..."
        wget -O "$THEME_TAR" "$THEME_URL" || {
            echo "Error: Download failed"
            exit 1
        }
    fi

    # Extract the theme
    echo "Extracting theme..."
    mkdir -p "$THEME_DIR"
    tar -xf "$THEME_TAR" -C "$THEME_DIR" --strip-components=1 || {
        echo "Error: Extraction failed"
        exit 1
    }
fi

# Download CPP Reference Tags
if [ ! -d "$TAGS_DIR" ]; then
    echo "Creating tags directory..."
    mkdir -p "$TAGS_DIR"
fi

cd "$TAGS_DIR" || {
    echo "Error: Failed to navigate to tags directory"
    exit 1
}

if [ -f "$CPP_REF_TAG_FILE" ]; then
    echo "CPP Reference tags already downloaded."
else
    echo "Downloading CPP Reference tags..."
    curl -o "$CPP_REF_TAG_FILE" "$CPP_REF_TAG_URL" || {
        echo "Error: Tags download failed"
        exit 1
    }
fi

# Run Doxygen with the specified configuration
echo "Generating documentation with Doxygen..."
cd "$ROOT_DIR" || {
    echo "Error: Failed to navigate back to root"
    exit 1
}
doxygen "$CFG_FILE" || {
    echo "Error: Doxygen generation failed"
    exit 1
}

echo "Documentation generation complete."
