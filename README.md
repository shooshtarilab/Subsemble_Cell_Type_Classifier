[![License: CC BY 4.0](https://licensebuttons.net/l/by/4.0/80x15.png)](https://creativecommons.org/licenses/by/4.0/)

# Subsemble Cell Type Classifier

## Description
Code and Extended Data for the manuscript "Cell-type classification of single-cell transcriptomics using the Subsemble supervised ensemble machine learning classifier".

## File Structure
<pre>
├── Datasets.zip                                             // Two scRNA-seq datasets and cell type labels used for cross-validation of cell type classification performance.
│   └── li_crc_counts.tsv
│   └── li_crc_labels.txt
│   └── vangalen_aml_counts.csv
│   └── vangalen_aml_labels.csv
├── Figure_Data.zip                                          // Classification performance metrics used to generate Figures 2, 3, and 4.
│   └── Figure 2
│   └── Figure 3
│   └── Figure 4AB
│   └── Figure 4C
├── Subsemble.ipynb                                          // Jupyter notebook with example Subsemble classifier for user input.
├── skf_metrics_cc.py                                        // Python script for classification performance benchmarking using N-fold CV or LOOCV.
│   └── PCAWG_sigProfiler_DBS_signatures_in_samples.csv
</pre>

## Citation
[Chen, D., Shooshtari, P. (2022). Cell-type classification of single-cell transcriptomics using the Subsemble supervised ensemble machine learning classifier. In preparation.

## License
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
