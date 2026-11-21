# Creative approaches

These are optional starting points for the Moroccan crop-classification challenge. Choose one approach and a question you can investigate within the available time. The references describe general methods or studies on other tasks; they do not establish performance on the released Moroccan dataset.

## Approaches and references

| Approach | A starting implementation | A useful comparison | Method reference |
| --- | --- | --- | --- |
| QNN with data re-uploading | Alternate feature-encoding gates with shallow trainable rotations. | One versus several encoding repetitions, keeping the data and training budget fixed. | [Pérez-Salinas et al. (2020)](https://quantum-journal.org/papers/q-2020-02-06-226/) |
| Quantum kernel classifier | Compute quantum feature-state similarities and fit a classical SVM. | Linear and RBF kernels on the same inputs and training examples. | [Havlíček et al. (2019)](https://arxiv.org/abs/1804.11326) |
| Classical encoder with a quantum head | Use a small classical encoder or the supplied compressed features as input to a QNN. | Replace the quantum head with a classical head of similar trainable size; keep the shared encoder unchanged. | [Mari et al. (2020)](https://quantum-journal.org/papers/q-2020-10-09-340/) |
| Quantum features with a classical classifier | Use fixed quantum circuits to generate features, then train a classical classifier. | The same learner without the quantum features; optionally add a comparable classical random-feature map. | [Wilson et al. (2018; revised 2019)](https://arxiv.org/abs/1806.08321) |
| Classical and quantum ensemble | Combine the supplied classical model with quantum-model predictions. | The classical model alone and, where feasible, a comparable classical-only ensemble. | [Grossi et al. (2022)](https://arxiv.org/abs/2208.07963) |
| Quantum convolution and pooling | Apply repeated local circuit blocks to meaningfully ordered measurements. | A shallow QNN or classical model on the same ordered inputs. | [Cong, Choi and Lukin (2019)](https://arxiv.org/abs/1810.03787) |

The encoder and ensemble studies support those design patterns, rather than a particular crop classifier. The QCNN reference studies quantum-state tasks; applying its structure to satellite features requires an explanation of the adaptation.

## Choose one question

A useful experiment changes one main factor while keeping the comparison clear. Examples include:

- **Encoding or depth:** Does a second encoding repetition alter macro-F1, training stability or runtime?
- **Training-set size:** How do both models behave on the same fixed subsets, such as 25%, 50% and 100% of the released training examples?
- **Quantum contribution:** Does replacing the quantum component change the result after the classical preprocessing is held fixed?
- **Missing measurements:** How does a documented missing-feature perturbation affect both models on validation data?
- **Error patterns:** Which crop classes are confused, and do the models make the same mistakes?
- **Sampling:** How do predictions change when an exact model is evaluated with a finite number of shots? Keep this separate from the main exact-simulation result.

A result below the reference can still support a useful conclusion. For example: “The deeper circuit used more evaluations, while the observed F1 difference was smaller than the variation across seeds.” Limit that conclusion to the tested data and settings.

## Practical suggestions

Start with the four-feature view and the supplied four-qubit model. Confirm a complete run before trying a larger circuit. Cache fixed quantum features or kernel matrices when appropriate, and include the time to create those caches in the reported computation.

Count independent trainable values, not gates or qubits. Shared parameters count once. Include learned encoding scales, classical heads and fusion weights; report frozen encoder parameters separately. A fixed quantum circuit can have no trained angles and still require computation.

For a QNN versus a small neural control, aim for trainable counts within about 20%. For example, a circuit with 36 trainable angles followed by a linear readout from six measured features to three classes has 57 trainable values: 36 + 18 weights + 3 biases. A 6-input, 5-hidden-unit, 3-output MLP has 53. This is an approximate parameter match, not proof of equal model capacity or computational cost.

For stacking, use out-of-fold training predictions when fitting the model that combines the branches. Fit all additional preprocessing on training data only. For a QCNN, adjacent dates or bands may provide a meaningful order; arbitrary PCA component order does not establish locality. The Qiskit QCNN example is binary and needs an appropriate multiclass readout if the released task has three classes.

## Qiskit starting points

- [Neural network classifier tutorial](https://stfc.github.io/qiskit-machine-learning/tutorials/02_neural_network_classifier_and_regressor.html)
- [Quantum kernel tutorial](https://stfc.github.io/qiskit-machine-learning/tutorials/03_quantum_kernel.html)
- [QCNN tutorial](https://stfc.github.io/qiskit-machine-learning/tutorials/11_quantum_convolutional_neural_networks.html)
- [QMLSampler documentation](https://stfc.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.primitives.QMLSampler.html): the exact mode uses `shots=None`; a finite integer selects sampled evaluation.

The tested package versions and working examples in this repository take precedence over older code snippets in the papers.

## References

1. Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., and Latorre, J. I. (2020). *Data re-uploading for a universal quantum classifier*. Quantum 4, 226. [Paper](https://quantum-journal.org/papers/q-2020-02-06-226/).
2. Havlíček, V., et al. (2019). *Supervised learning with quantum-enhanced feature spaces*. Nature 567, 209–212. [Author preprint](https://arxiv.org/abs/1804.11326).
3. Mari, A., Bromley, T. R., Izaac, J., Schuld, M., and Killoran, N. (2020). *Transfer learning in hybrid classical-quantum neural networks*. Quantum 4, 340. [Paper](https://quantum-journal.org/papers/q-2020-10-09-340/).
4. Wilson, C. M., et al. (2018; revised 2019). *Quantum Kitchen Sinks: An algorithm for machine learning on near-term quantum computers*. arXiv:1806.08321. [Preprint](https://arxiv.org/abs/1806.08321).
5. Grossi, M., et al. (2022). *Mixed Quantum–Classical Method for Fraud Detection With Quantum Feature Selection*. IEEE Transactions on Quantum Engineering 3. [Author preprint](https://arxiv.org/abs/2208.07963).
6. Cong, I., Choi, S., and Lukin, M. D. (2019). *Quantum Convolutional Neural Networks*. Nature Physics. [Author preprint](https://arxiv.org/abs/1810.03787).
7. Bowles, J., Ahmed, S., and Schuld, M. (2024). *Better than classical? The subtle art of benchmarking quantum machine learning models*. [Preprint](https://arxiv.org/abs/2403.07059). Background on controls and interpretation.
