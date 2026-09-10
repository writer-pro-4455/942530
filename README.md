# Maize Edge AI Reproducibility Repository

This repository accompanies the manuscript **“A Lightweight Edge AI Deployment Framework for Real-Time Maize Disease Detection: MobileNetV3Small Compression, ScoreCAM Explainability, and MQTT-Based IoT Integration.”**

It provides a clean, auditable implementation scaffold for:

- deterministic data splitting and preprocessing,
- comparative training of a Custom CNN, MobileNetV2, MobileNetV3Small, and EfficientNetB0,
- held-out evaluation,
- TensorFlow Lite FP16 / INT8 post-training quantization,
- CPU-only inference benchmarking,
- ScoreCAM explainability,
- MQTT telemetry publishing,
- ThingsBoard integration testing.

> **Important scope statement**
>
> The environmental fields used by the IoT demo (temperature, humidity, soil moisture, illuminance, and battery voltage) are **simulated communication-test telemetry only**. They are **not classifier inputs**, were not acquired from synchronized physical sensors, and should not be interpreted as sensor fusion or disease–environment correlation evidence.

## Repository structure

```text
.
├── configs/
│   ├── experiment.json
│   └── inference_config.json
├── data/
│   └── README.md
├── models/
│   └── README.md
├── results/
│   ├── README.md
│   └── manuscript_reference_metrics.csv
├── scripts/
│   ├── check_repository.py
│   └── reproduce_pipeline.py
├── src/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── data.py
│   ├── evaluate.py
│   ├── iot_demo.py
│   ├── models.py
│   ├── mqtt_publish.py
│   ├── quantize.py
│   ├── scorecam.py
│   └── train.py
├── thingsboard/
│   ├── README.md
│   ├── dashboard_config.example.json
│   └── mqtt_payload_example.json
├── .gitignore
├── CITATION.cff
├── DATA_AND_CODE_AVAILABILITY_TEMPLATE.md
├── LICENSE
├── README.md
├── REPRODUCIBILITY_CHECKLIST.md
└── requirements.txt
```

## Dataset

### Primary dataset

**Corn or Maize Leaf Disease Dataset** (Kaggle)

https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset

The manuscript uses 4,188 RGB maize-leaf images across four classes:

- Common Rust: 1,306
- Northern Leaf Blight: 1,146
- Healthy: 1,162
- Gray Leaf Spot: 574

The configured split is 70% training, 15% validation, and 15% testing, using a fixed random seed of 42.

### Secondary dataset

PlantDoc is used for cross-dataset / field-condition evaluation. See:

Singh et al., “PlantDoc: A Dataset for Visual Plant Disease Detection,” 2020.

https://doi.org/10.1145/3371158.3371196

Because an exact manuscript-time PlantDoc subset manifest is not available in this package, users should record and publish the exact image list used for any reported reproduction.

## Environment

The manuscript reports TensorFlow 2.13.0 CPU, TensorFlow Lite Runtime 2.13.0, scikit-learn 1.3.0, paho-mqtt 1.6.1, and ThingsBoard Community Edition 4.2.1.1.

The manuscript currently lists the Python version as `3.1`, which is almost certainly incomplete or typographical. **Verify and replace it with the exact Python version used in the original experiment before claiming exact reproduction.**

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Expected dataset layout

Place the primary dataset outside version control and point the configuration to it. A class-directory layout is expected, for example:

```text
data/raw/maize/
├── Common_Rust/
├── Gray_Leaf_Spot/
├── Healthy/
└── Northern_Leaf_Blight/
```

See `data/README.md` for details.

## Experiment configuration

All main settings are centralized in:

```text
configs/experiment.json
```

Key values include:

- image size: 224 × 224
- batch size: 32
- random seed: 42
- train/validation/test: 70/15/15
- Phase 1 learning rate: 1e-3
- Phase 1 maximum epochs: 10
- Phase 2 learning rate: 1e-5
- Phase 2 maximum epochs: 25
- fine-tuning fraction: final 30% of the pretrained backbone
- representative INT8 calibration samples: 100
- ScoreCAM sampled channels: 50
- ScoreCAM overlay alpha: 0.45
- MQTT QoS: 1
- disease HIGH threshold: 0.80

## Train models

Example:

```bash
python -m src.train --config configs/experiment.json --model custom_cnn
python -m src.train --config configs/experiment.json --model mobilenet_v2
python -m src.train --config configs/experiment.json --model mobilenet_v3_small
python -m src.train --config configs/experiment.json --model efficientnet_b0
```

The Custom CNN is a from-scratch baseline. The transfer-learning architectures use ImageNet initialization.

The revised manuscript interpretation is deliberately conservative: **the Custom CNN is the strongest classifier in the reported comparison; MobileNetV3Small is retained as the implemented deployment case study, not claimed as the globally optimal classifier.**

## Evaluate

```bash
python -m src.evaluate \
  --config configs/experiment.json \
  --model-path models/mobilenet_v3_small_best.keras \
  --model-name mobilenet_v3_small
```

Outputs include overall metrics, per-class metrics, prediction CSVs, and confusion-matrix data.

## Quantize MobileNetV3Small

```bash
python -m src.quantize \
  --config configs/experiment.json \
  --model-path models/mobilenet_v3_small_best.keras \
  --output-dir models
```

This generates FP16 and INT8 TensorFlow Lite models. The INT8 representative calibration set uses 100 training images, as configured in `experiment.json`.

## Benchmark CPU inference

```bash
python -m src.benchmark \
  --config configs/experiment.json \
  --model-path models/mobilenet_v3_small_int8.tflite \
  --split test \
  --runs 100 \
  --warmup 5
```

The benchmark measures:

- serialized model size,
- wall-clock latency,
- latency standard deviation,
- throughput (FPS),
- Python-side peak traced memory.

**Energy is intentionally not estimated from processor TDP.** A TDP × latency calculation is not a valid direct measurement of workload energy and should not be extrapolated from an x86 laptop CPU to ARM hardware. If energy is needed, use direct hardware power instrumentation and report the measurement protocol.

## ScoreCAM

```bash
python -m src.scorecam \
  --model-path models/mobilenet_v3_small_best.keras \
  --image path/to/leaf.jpg \
  --output results/scorecam_example.png \
  --max-channels 50 \
  --alpha 0.45
```

The script implements a forward-pass ScoreCAM workflow and limits the number of activation channels for CPU feasibility.

## MQTT / ThingsBoard

Create a ThingsBoard device and obtain its device access token. Do not commit the token.

Set it as an environment variable:

```bash
export THINGSBOARD_DEVICE_TOKEN="YOUR_DEVICE_TOKEN"
```

On Windows PowerShell:

```powershell
$env:THINGSBOARD_DEVICE_TOKEN="YOUR_DEVICE_TOKEN"
```

Publish one inference payload:

```bash
python -m src.mqtt_publish \
  --host localhost \
  --port 1883 \
  --prediction "Common Rust" \
  --confidence 0.932 \
  --latency-ms 273.03
```

Run the 20-cycle integration test:

```bash
python -m src.iot_demo \
  --config configs/inference_config.json \
  --model-path models/mobilenet_v3_small_int8.tflite \
  --image-dir data/raw/maize \
  --cycles 20 \
  --simulate-telemetry
```

Alert logic:

- `HIGH`: disease prediction and confidence ≥ 0.80
- `MEDIUM`: disease prediction and confidence < 0.80
- `NONE`: Healthy prediction

The 20-cycle integration run should be interpreted as a communication and alert-rule test, not as a replacement for the held-out classification benchmark.

## ThingsBoard files

`thingsboard/dashboard_config.example.json` is a semantic example of the intended dashboard fields. It is **not represented as the original exported ThingsBoard dashboard** because the exact original export was not available when this package was assembled.

Before manuscript resubmission, export and commit the actual dashboard / rule-chain configuration used in the experiment if available. See `thingsboard/README.md`.

## Reference metrics from the manuscript

`results/manuscript_reference_metrics.csv` records the values currently reported in the revised manuscript for comparison during reproduction. These values are reference targets, not programmatically generated evidence in this repository.

Key reported values include:

| Model | Held-out accuracy | Macro F1 | Training time (s) |
|---|---:|---:|---:|
| Custom CNN | 97.8% | 0.976 | 312.4 |
| MobileNetV2 | 91.2% | 0.908 | 1847.3 |
| MobileNetV3Small | 94.6% | 0.943 | 2134.6 |
| EfficientNetB0 | 88.7% | 0.881 | 2891.5 |

For the implemented MobileNetV3Small deployment case, the revised manuscript reports approximately:

- Float32 serialized size: 3.87 MB
- INT8 serialized size: 0.60 MB
- INT8 held-out accuracy: 93.1%
- INT8 mean inference latency on the evaluated x86 CPU host: 273.03 ± 6.66 ms
- throughput: approximately 3.7 FPS
- Python-side peak traced memory: approximately 48 MB

These results must be independently regenerated and compared before making a strict “exact reproduction” claim.

## Reproducibility checklist

Run:

```bash
python scripts/check_repository.py
```

Then follow `REPRODUCIBILITY_CHECKLIST.md` before creating an archival release.

## What must still be supplied by the original authors

For full auditability of the paper’s exact numerical results, the authors should add:

1. exact Python version,
2. exact processor model,
3. exact primary train/validation/test image manifests,
4. exact PlantDoc subset manifest,
5. original trained model checkpoints,
6. hashes for released model artifacts,
7. raw TensorBoard logs,
8. raw benchmark output,
9. raw 20-cycle MQTT integration log,
10. actual ThingsBoard dashboard and rule-chain exports.

This repository deliberately does not fabricate unavailable artifacts.

## Citation

Citation metadata is provided in `CITATION.cff`.

## License

See `LICENSE`.
