#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "Running Rust project checks..."

# Format check
echo -e "\n${GREEN}Checking code formatting...${NC}"
cargo fmt --all -- --check

# Clippy
echo -e "\n${GREEN}Running clippy...${NC}"
cargo clippy -- -D warnings

# Build
echo -e "\n${GREEN}Building project...${NC}"
cargo build

# Tests
echo -e "\n${GREEN}Running tests...${NC}"
cargo test

echo -e "\n${GREEN}All checks passed!${NC}"
