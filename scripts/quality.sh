#!/bin/bash
# Code quality check script

set -e

echo "==================================="
echo "LLM Test Platform - Code Quality Check"
echo "==================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track status
FAILED=0

echo -e "${YELLOW}[1/4] Running Ruff (linting)...${NC}"
if ruff check .; then
    echo -e "${GREEN}[OK] Ruff checks passed${NC}"
else
    echo -e "${RED}[ERROR] Ruff checks failed${NC}"
    FAILED=1
fi
echo ""

echo -e "${YELLOW}[2/4] Running Ruff (formatting check)...${NC}"
if ruff format --check .; then
    echo -e "${GREEN}[OK] Ruff format checks passed${NC}"
else
    echo -e "${YELLOW}[WARNING] Ruff format issues found. Run 'ruff format .' to fix${NC}"
    FAILED=1
fi
echo ""

echo -e "${YELLOW}[3/4] Running Mypy (type checking)...${NC}"
if mypy ui/ config/ core/*.py 2>/dev/null; then
    echo -e "${GREEN}[OK] Mypy checks passed${NC}"
else
    echo -e "${YELLOW}[WARNING] Mypy found type issues (non-blocking)${NC}"
fi
echo ""

echo -e "${YELLOW}[4/4] Running Pytest (unit tests)...${NC}"
if pytest tests/ -q --tb=no; then
    echo -e "${GREEN}[OK] All tests passed${NC}"
else
    echo -e "${RED}[ERROR] Tests failed${NC}"
    FAILED=1
fi
echo ""

echo "==================================="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All quality checks passed!${NC}"
    exit 0
else
    echo -e "${RED}Some quality checks failed${NC}"
    exit 1
fi
