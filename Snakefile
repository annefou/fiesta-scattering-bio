rule all:
    input:
        "results/stacking_val_trained_results.json",


rule scattering_features:
    input:
        script="01_scattering_features.py",
    output:
        "results/features_train_balanced.npz",
        "results/features_val.npz",
        "results/features_test.npz",
    shell:
        """
        jupytext --to notebook {input.script}
        jupyter execute --inplace 01_scattering_features.ipynb
        """


rule cnn_predictions:
    input:
        script="02_cnn_predictions.py",
    output:
        "results/cnn_predictions_val.npz",
        "results/cnn_predictions_test.npz",
    shell:
        """
        jupytext --to notebook {input.script}
        jupyter execute --inplace 02_cnn_predictions.ipynb
        """


rule stacking:
    input:
        script="03_stacking.py",
        features_train="results/features_train_balanced.npz",
        features_val="results/features_val.npz",
        features_test="results/features_test.npz",
        cnn_val="results/cnn_predictions_val.npz",
        cnn_test="results/cnn_predictions_test.npz",
    output:
        "results/stacking_val_trained_results.json",
    shell:
        """
        jupytext --to notebook {input.script}
        jupyter execute --inplace 03_stacking.ipynb
        """
