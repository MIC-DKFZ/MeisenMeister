# CLI Guide

This page is the practical reference for the installed console commands.

## Available Commands

- `mm_extract_dataset_fingerprint`
- `mm_create_breast_segmentations`
- `mm_homogenize`
- `mm_plan_experment`
- `mm_preprocess`
- `mm_plan_and_preprocess`
- `mm_create_5fold`
- `mm_train`
- `mm_predict`
- `mm_predict_from_modelfolder`

## Common Usage Pattern

For a dataset with id `1`, a typical end-to-end flow looks like this:

```bash
mm_create_breast_segmentations -d 1
mm_extract_dataset_fingerprint -d 1
mm_plan_and_preprocess -d 1 --num-workers 4
mm_create_5fold -d 1
mm_train -d 1 -f 0
mm_predict -d 1 -i /path/to/images -o /path/to/preds -f 0
```

## Command Reference

### `mm_create_breast_segmentations`

Creates breast segmentations in `masksTr` next to the raw training images.

Example:

```bash
mm_create_breast_segmentations -d 1
```

Use this before fingerprinting or preprocessing if the raw dataset does not already contain the expected masks.

### `mm_extract_dataset_fingerprint`

Reads the raw dataset and computes ROI geometry statistics used for planning.

Example:

```bash
mm_extract_dataset_fingerprint -d 1 --num-workers 8
```

Writes:

- `dataset_fingerprint.json` under the dataset’s preprocessed directory

### `mm_homogenize`

Resamples all non-reference image channels into the `_0000` image space and overwrites the raw NIfTI files.

Example:

```bash
mm_homogenize -d 1
```

This command is intentionally guarded by a confirmation prompt because it modifies raw data in place.

### `mm_plan_experment`

Builds `mmPlans.json` from `dataset_fingerprint.json`.

Example:

```bash
mm_plan_experment -d 1
```

The command name is spelled exactly as shown because that is the current installed entrypoint.

### `mm_preprocess`

Creates the preprocessed ROI dataset from the raw data and `mmPlans.json`.

Example:

```bash
mm_preprocess -d 1 --num-workers 8
```

### `mm_plan_and_preprocess`

Runs planning and preprocessing in one step.

Example:

```bash
mm_plan_and_preprocess -d 1 --num-workers 8
```

This is the most convenient option when you do not need to inspect or edit the plan between steps.

### `mm_create_5fold`

Creates `splits.json` in the preprocessed dataset folder.

Example:

```bash
mm_create_5fold -d 1
```

The split logic is case-aware and checks for train/validation leakage at the case level.

### `mm_train`

Runs a registered trainer on one fold.

Basic example:

```bash
mm_train -d 1 -f 0
```

Select a trainer and architecture:

```bash
mm_train -d 1 -f 0 --trainer mmTrainer -a ResNet3D18
```

Resume an existing experiment:

```bash
mm_train -d 1 -f 0 -c
```

Start a fresh fine-tuning run from external weights:

```bash
mm_train -d 1 -f 0 -w /path/to/weights.pt
```

Start a fine-tuning run with a custom experiment-name suffix:

```bash
mm_train -d 1 -f 0 -w /path/to/weights.pt --postfix finetuningNNSSL
```

Resume that suffixed experiment later:

```bash
mm_train -d 1 -f 0 -c --postfix finetuningNNSSL
```

Important rule:

- `-c` and `-w` cannot be used together

That is a deliberate validation rule. Resume means “load experiment state from checkpoint.” Weights means “start a new run from an external initialization.” The code refuses to guess which one you meant.

### `mm_predict`

Runs ROI-level inference using a locally configured dataset id plus the matching results folder under `MM_RESULTS`.

Example:

```bash
mm_predict -d 1 -i /path/to/images -o /path/to/preds -f 0 1 2 3 4
```

### `mm_predict_from_modelfolder`

Runs the same inference flow from a portable experiment folder, without requiring `-d` or local `MM_RAW` / `MM_PREPROCESSED` setup.

The folder must be an experiment root of the form:

`<results>/<dataset_name>/<Trainer>_<Architecture>[_<postfix>]`

and must contain:

- `dataset.json`
- `mmPlans.json`
- `fold_<n>/model_best.pt` and/or `fold_<n>/model_last.pt`

Example:

```bash
mm_predict_from_modelfolder /shared/Dataset_001_Test/mmTrainer_ResNet3D18 -i /path/to/images -o /path/to/preds -f 0 1 2 3 4
```

Training copies `dataset.json` and `mmPlans.json` into the experiment folder automatically so the folder can be shared for inference on another machine.

## A Few Practical Notes

- Dataset ids are integers on the CLI, but are internally normalized to three digits.
- The commands assume the path roots described in [Setup and Paths](setup.md) are already configured.
- If a command seems to “not find” a dataset, the first thing to check is whether the folder under `MM_RAW` matches `Dataset_<id>_*`.

## Related Reading

- [High-Level Overview](overview.md)
- [Training and Experiments](training.md)
- [Data and Pipeline Notes](pipeline.md)
