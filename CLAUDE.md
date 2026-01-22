# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a homework repository for **MATH4665 - Special Topics in Applied Mathematics I** (Professor Xia Jiahan). The repository contains mathematical solutions with accompanying computational implementations and professional LaTeX documentation.

## Repository Structure

```
/
├── HW1/                      # Homework 1 directory
│   ├── Sol1.tex             # LaTeX source for solutions
│   ├── Sol1.pdf             # Compiled solution PDF (deliverable)
│   ├── HW1.pdf              # Assignment instructions
│   ├── *.py                 # Python implementation scripts
│   └── conversation_history.md  # Development log
├── MATH4665.pdf             # Course syllabus
└── README.md                # Basic metadata
```

Each homework assignment gets its own subdirectory (e.g., `HW1/`, `HW2/`, etc.) containing:
- LaTeX source files (`*.tex`)
- Compiled PDF deliverables (`*.pdf`)
- Python implementation scripts (`*.py`)
- LaTeX build artifacts (`*.aux`, `*.log`, `*.fls`, `*.fdb_latexmk`, `*.synctex.gz`, `*.out`)

## Build and Compilation Commands

### LaTeX Compilation

The project uses **pdflatex** (TeX Live 2023) for compiling LaTeX documents.

```bash
# Compile LaTeX source (run twice for cross-references)
pdflatex Sol1.tex
pdflatex Sol1.tex

# Or use latexmk if available (handles multiple passes automatically)
latexmk -pdf Sol1.tex

# Clean build artifacts
latexmk -c  # or manually remove *.aux, *.log, *.fls, etc.
```

### Python Scripts

Python scripts use **NumPy** for numerical computations and are designed for reproducibility:

```bash
# Run Python simulation/script
python mc_control_variates.py
```

Key characteristics:
- Fixed random seeds for reproducibility
- Outputs formatted results to console
- Results are manually integrated into LaTeX reports

## Development Workflow

1. **Mathematical Development**: Write solution content in LaTeX (`Sol1.tex`)
   - Include mathematical derivations, proofs, and explanations
   - Use standard packages: `amsmath`, `amssymb`, `amsthm`, `geometry`, `hyperref`, `listings`, `xcolor`

2. **Computational Implementation**: Create Python scripts to implement numerical methods
   - Test algorithms and generate results
   - Use `np.random.seed()` for reproducible outputs

3. **Integration**: Embed code and results into LaTeX
   - Use `lstinputlisting` or inline `lstlisting` environment to include Python source
   - Add formatted output as verbatim or quoted text

4. **Documentation**: Maintain `conversation_history.md` in each HW directory to track development decisions and iterations

## LaTeX Document Structure

The solution LaTeX files follow this structure:

```latex
\documentclass[12pt]{article}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{listings}  % For code inclusion
\usepackage{xcolor}    % For syntax highlighting

% Code listing configuration
\lstset{
    language=Python,
    basicstyle=\ttfamily\small,
    keywordstyle=\color{blue},
    ...
}

\begin{document}
\section*{Question 1}
% Mathematical content

% Included Python code
\lstinputlisting{script.py}

% Output results
\begin{verbatim}
...
\end{verbatim}
\end{document}
```

## Key Patterns and Conventions

### Code Organization
- Each homework has standalone Python scripts (not a package)
- Functions are documented with docstrings explaining the mathematics
- Main execution guarded by `if __name__ == "__main__":`
- Results printed with formatted output for manual inclusion in LaTeX

### Mathematical Notation in LaTeX
- Use `\emph{}` for emphasized terms (e.g., *eigenvectors*, *power iteration*)
- Display equations in `align*` or `\[ \]` environments
- Inline equations with `$ $`
- Theorems and proofs use `amsthm` environments

### Git Workflow
- The repository uses git (main branch)
- Commit messages follow conventional format:
  - `feat(HW1):` for new features
  - `docs(HW1):` for documentation changes
  - `chore:` for file organization

## Common Tasks

### Adding a New Homework Solution

1. Create new directory: `mkdir HW2/`
2. Copy assignment PDF: `HW2.pdf` → `HW2/HW2.pdf`
3. Create LaTeX source: `HW2/Sol2.tex` (copy structure from `HW1/Sol1.tex`)
4. Implement Python scripts in `HW2/` as needed
5. Compile and test: `cd HW2 && pdflatex Sol2.tex && pdflatex Sol2.tex`
6. Update root `README.md` with new homework date

### Modifying Existing Solutions

- Edit LaTeX source directly, then recompile
- If Python code changes, re-run script and update output in LaTeX
- Maintain consistency in mathematical notation and formatting
