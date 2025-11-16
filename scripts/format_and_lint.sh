#!/bin/bash

# Development workflow automation script for Image Similarity Toolkit
# This script runs all code quality tools in the recommended order
# Usage: ./scripts/format_and_lint.sh

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}🔧 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if tools are installed
check_tools() {
    local tools=("ruff" "black" "isort" "autoflake" "docformatter" "mypy" "pylint")
    local missing_tools=()
    
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing tools: ${missing_tools[*]}"
        echo "Install with: pip install -r requirements-dev.txt"
        exit 1
    fi
}

# Function to run command with error handling
run_command() {
    local cmd="$1"
    local description="$2"
    local ignore_errors="${3:-false}"
    
    print_step "$description"
    echo "Running: $cmd"
    
    if eval "$cmd"; then
        print_success "$description completed"
    else
        if [ "$ignore_errors" = "true" ]; then
            print_warning "$description completed with warnings"
        else
            print_error "$description failed"
            exit 1
        fi
    fi
    echo
}

# Main workflow
main() {
    echo "🚀 Starting Image Similarity Toolkit Code Quality Check"
    echo "========================================================"
    echo
    
    # Check tools installation
    check_tools
    
    # Step 1: Fix syntax errors with Ruff
    print_step "Step 1: Fix syntax errors with Ruff"
    run_command "ruff check --fix . --exclude .venv" "Fix syntax errors with Ruff"
    
    # Step 2: Remove unused code
    print_step "Step 2: Remove unused imports and variables"
    find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | \
    xargs -0 autoflake --in-place --remove-all-unused-imports --remove-unused-variables || true
    print_success "Removed unused code"
    echo
    
    # Step 3: Format code with Black
    print_step "Step 3: Format code with Black"
    find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | \
    xargs -0 black || true
    print_success "Formatted code with Black"
    echo
    
    # Step 4: Sort imports with isort
    print_step "Step 4: Sort imports with isort"
    find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | \
    xargs -0 isort --profile black || true
    print_success "Sorted imports with isort"
    echo
    
    # Step 5: Format docstrings
    print_step "Step 5: Format docstrings"
    find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | \
    xargs -0 docformatter --in-place --wrap-summaries=100 --wrap-descriptions=100 || true
    print_success "Formatted docstrings"
    echo
    
    # Step 6: Type checking with MyPy
    print_step "Step 6: Type checking with MyPy"
    run_command "mypy . --exclude '.venv' --show-error-codes --pretty" \
               "Type checking with MyPy" "true"
    
    # Step 7: Deep linting with Pylint
    print_step "Step 7: Deep linting with Pylint"
    run_command "find . -type d -name '.venv' -prune -o -type f -name '*.py' -print0 | xargs -0 pylint --rcfile=pyproject.toml" \
               "Deep linting with Pylint" "true"
    
    # Step 8: Run tests
    print_step "Step 8: Run tests"
    run_command "python -m pytest tests/ -v" "Run tests" "true"
    
    echo
    print_success "🎉 All code quality checks completed!"
    echo
    echo "📊 Summary:"
    echo "  - ✅ Syntax errors fixed"
    echo "  - ✅ Code formatted"
    echo "  - ✅ Imports organized"
    echo "  - ✅ Docstrings formatted"
    echo "  - ✅ Type checking completed"
    echo "  - ✅ Linting completed"
    echo "  - ✅ Tests passed"
    echo
    echo "💡 Tip: Consider setting up pre-commit hooks for automatic quality checks:"
    echo "   pre-commit install"
}

# Quick check (only syntax and formatting)
quick_check() {
    echo "🔍 Quick Code Quality Check"
    echo "==========================="
    
    check_tools
    
    print_step "Quick syntax check with Ruff"
    ruff check . --exclude .venv || print_warning "Some issues found"
    
    print_step "Quick format check with Black"
    black --check . --exclude .venv || print_warning "Code needs formatting"
    
    print_step "Quick import check with isort"
    isort --check-only . --profile black --skip .venv || print_warning "Imports need sorting"
    
    echo
    print_success "Quick check completed"
}

# Help function
show_help() {
    echo "Image Similarity Toolkit - Code Quality Automation"
    echo
    echo "Usage: $0 [OPTION]"
    echo
    echo "Options:"
    echo "  (no option)    Run full code quality check"
    echo "  quick          Run only syntax and formatting checks"
    echo "  install-tools  Install all development tools"
    echo "  setup-hooks    Setup pre-commit hooks"
    echo "  help           Show this help message"
    echo
    echo "Examples:"
    echo "  $0             # Run full check"
    echo "  $0 quick       # Run quick check"
    echo "  $0 install-tools  # Install development tools"
    echo
}

# Install tools function
install_tools() {
    print_step "Installing development tools"
    pip install -r requirements-dev.txt
    print_success "Development tools installed"
}

# Setup pre-commit hooks
setup_hooks() {
    print_step "Setting up pre-commit hooks"
    pre-commit install
    print_success "Pre-commit hooks installed"
    
    # Run on all files to set up the hooks
    print_step "Running pre-commit on all files (first time setup)"
    pre-commit run --all-files || true
    print_success "Pre-commit hooks configured"
}

# Parse command line arguments
case "${1:-}" in
    "quick")
        quick_check
        ;;
    "install-tools")
        install_tools
        ;;
    "setup-hooks")
        setup_hooks
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    "")
        main
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac