rule all:
    input:
        "results/plankton_classification.png",


rule run_notebook:
    input:
        script="01_plankton_classification.py",
    output:
        "results/plankton_classification.png",
    shell:
        """
        jupytext --to notebook {input.script}
        jupyter execute 01_plankton_classification.ipynb
        """
