#!/usr/bin/env python3
"""Compile the working ACL report using the existing local TeX Live image."""
from pathlib import Path
import os
import re
import subprocess

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "acl-style"


def main():
    build = HERE / "build"
    build.mkdir(exist_ok=True)
    source = (HERE / "main.tex").read_text() + "\n" + "\n".join(
        p.read_text() for p in sorted((HERE / "sections").glob("*.tex"))
    )
    if "\u2014" in source or "\u2013" in source or "--" in source:
        raise ValueError("en/em dash found in LaTeX draft; use single hyphens or -{}- for literal double hyphens")
    cited = {key.strip() for group in re.findall(r"\\cite\w*\{([^}]+)\}", source) for key in group.split(",")}
    keys = set(re.findall(r"@\w+\{([^,]+),", (HERE / "references.bib").read_text()))
    if cited - keys:
        raise ValueError(f"missing references: {sorted(cited - keys)}")
    command = ["podman", "run", "--rm", "--network", "none",
               "-v", f"{HERE}:/work:rw", "-v", f"{TEMPLATE}:/acl-template:ro", "-w", "/work",
               "-e", "TEXINPUTS=.:/acl-template:", "-e", "BSTINPUTS=/acl-template:",
               "-e", "BIBINPUTS=/work:", "docker.io/texlive/texlive:latest"]
    commands = [["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "./main.tex"],
                ["bibtex", "build/main"],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "./main.tex"],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "./main.tex"]]
    with (build / "compile.log").open("w") as log:
        for args in commands:
            result = subprocess.run(command + args, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"compile failed: {' '.join(args)}; inspect {build / 'compile.log'}")
    final_log = (build / "main.log").read_text()
    issues = [line for line in final_log.splitlines() if "undefined" in line.lower()
              or "Overfull" in line or "multiply defined" in line]
    if issues:
        raise RuntimeError("LaTeX issues: " + "\n".join(issues))
    print(f"Compiled: {build / 'main.pdf'}")
    print(f"Citations checked: {len(cited)}; no missing references, overfull boxes, or en/em dashes.")


if __name__ == "__main__":
    main()
