# Evaluation and fair comparison

The challenge assesses predictive performance, experimental quality and model design. A higher score than the classical reference is not required for a submission to be considered.

## Two comparisons

| Comparison | Purpose | Keep comparable |
| --- | --- | --- |
| Supplied classical reference | Evaluate the complete prediction pipeline. | Released training examples, validation and test splits, permitted data and target classes. The reference may use all supplied features. |
| Matched classical control | Investigate the contribution of the quantum component. | Input representation, shared preprocessing, training examples and tuning effort. Replace the quantum component with a suitable classical model. |

For a QNN and classical neural control, aim for trainable counts within about 20% of the quantum branch plus its trainable readout. This is a practical target, not a requirement to distort either model. Explain a mismatch. Keep shared encoders unchanged for the paired comparison and report their counts separately, as well as the full pipeline total.

Include all trainable classical and quantum components; count shared values once and separate frozen values. SVMs and trees do not have parameter counts directly comparable to QNN angles. For those comparisons, report suitable costs such as support-vector counts, tree sizes, kernel storage and runtime. Parameter matching alone does not establish equal capacity or cost.

## Data and computational budget

Use the released training set for the main comparison and the same subset for both methods in a learning-curve experiment. The small starter subset is for quick checks. Keep the validation and test sets fixed. Fit additional imputation, scaling, selection and compression on training data only.

The main result uses exact local simulation, up to eight qubits in each circuit, and the runtime guidance in the repository README. Runtime includes fitting, feature generation, cache construction and prediction. Report hardware, circuit evaluations, qubits and depth. Finite-shot or hardware experiments may be reported as additional analysis.

State the approximate number of tried configurations and the tuning time for the paired models. Use comparable effort, rather than requiring identical optimizers or epoch counts.

## Measurements

Run the selected quantum or hybrid model and matched stochastic control with seeds 0, 1 and 2. Report mean and standard deviation; a deterministic model may reuse its identical result with that fact stated.

Use macro-F1 over the full published class list as the primary metric. Also report accuracy, per-class precision and recall, a confusion matrix, training and prediction time, and model resources. Additional metrics, such as balanced accuracy, are welcome when they answer a clear question.

Report convergence against elapsed time or objective evaluations if studied; epochs alone need not represent comparable work. Three training seeds describe training variability, not statistical confidence over new fields. Any uncertainty claim about the difference between models should use paired field or spatial-block resampling and state the method.

## Judging

| Criterion | Points | Basis |
| --- | --- | --- |
| Predictive performance | 30 | 30 multiplied by mean final-test macro-F1 on a 0–1 scale. |
| Experimental quality and analysis | 40 | Appropriate controls, one focused comparison, resource measurements and supported interpretation. |
| Model design and reproducibility | 30 | Justified architecture and encoding, executable implementation and clear presentation. |

Two judges rate each of the latter categories from 0 to 5; average their ratings and multiply by the category weight divided by 5. Use 0 for missing evidence, 3 for adequate work and 5 for clear, well-supported work. Resolve ties by experimental quality, then predictive performance. Clear negative findings can receive full analysis credit.

Optional verified IBM real-hardware inference may earn up to **+5 bonus points beyond 100**. Provide a backend name, job ID and a brief ideal-simulator comparison. Hardware training is not required.

## Submission essentials

Provide one runnable notebook or code package containing the selected model, matched control, configurations and results, plus three to five slides. Include a short execution instruction and acknowledge reused code or methods. The README gives the submission route and deadline.

Train using only the released training labels; use the validation set for selection and the test set for final predictions. The quantum outputs should participate in the submitted prediction workflow. Organizers will rerun the workflow on the supplied data. There is no separate written-report requirement.
