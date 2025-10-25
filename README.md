# Sempress Core

**Open-Source Semantic Compression Library**

[![Paper](https://img.shields.io/badge/Paper-sempress.net-blue)](https://sempress.net/paper.pdf)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **Note:** This is the open-source core compression algorithm. For the managed API and enterprise features, visit [sempress.net](https://sempress.net).

Sempress achieves **50-125% better compression than gzip** on numeric-heavy datasets (IoT telemetry, ML features, financial data) through learned vector quantization.

---

## 🚀 Quick Start

```bash
# Install
pip install -e .

# Compress a CSV file
sempress encode --in data.csv --out data.smp --lock-cols id,timestamp --k 64

# Decompress
sempress decode --in data.smp --out data_reconstructed.csv

# Evaluate quality
sempress eval --original data.csv --recon data_reconstructed.csv
```

---

## 📊 Performance

| Dataset | Sempress | Gzip | **Improvement** |
|---------|----------|------|-----------------|
| **IoT Telemetry** | 8.08× | 3.58× | **+125%** 🔥 |
| **Sensor Physics** | 5.88× | 2.76× | **+113%** 🔥 |
| **ML Features** | 5.46× | 3.09× | **+77%** ✅ |
| **Financial Data** | 3.80× | 2.51× | **+51%** ✅ |

**Average:** 5.81× (Sempress) vs 2.99× (Gzip) = **+94% improvement**

---

## 💡 Key Features

- **Semantic Compression**: Learns column-wise patterns using K-Means vector quantization
- **Lossless Locked Columns**: Automatically preserves strings, categoricals, and IDs with 100% fidelity
- **Optional Residuals**: Achieve near-zero error on precision-critical columns (financial, scientific)
- **Uncertainty Tracking**: Flags cells with high quantization error for quality monitoring
- **Fast Decode**: Competitive with gzip+CSV parse (0.9-1.5× overhead)

---

## 📖 How It Works

Sempress applies **per-column K-Means vector quantization** to numeric data:

1. **Column Analysis**: Auto-detects numeric vs categorical columns
2. **Learn Codebooks**: K-Means learns k=64 centroids per numeric column
3. **Encode to Indices**: Replace values with nearest centroid index (uint16)
4. **Add Residuals** (optional): Store exact errors for high-precision columns
5. **Package**: Msgpack + Zstd container with schema and metadata

**Result:** Exploit semantic patterns in numeric data instead of treating tables as byte streams.

---

## 🛠️ Installation

### Requirements
- Python 3.10+
- pandas, numpy, scikit-learn, msgpack, zstandard

### Install from Source

```bash
git clone https://github.com/jalyper/sempress-core.git
cd sempress-core
pip install -e .
```

### Dependencies

```bash
pip install pandas numpy scikit-learn msgpack zstandard
```

---

## 📚 Usage Guide

### Basic Compression

```bash
# Encode CSV to .smp format
sempress encode \
  --in data.csv \
  --out data.smp \
  --lock-cols user_id,timestamp \
  --k 64
```

**Options:**
- `--lock-cols`: Columns to preserve losslessly (comma-separated)
- `--residual-cols`: High-precision columns (store exact errors)
- `--k`: Codebook size (default: 64, range: 16-256)
- `--uncert-thresh`: Flag cells with >X relative error (default: 0.2)

### Decompression

```bash
# Decode .smp back to CSV
sempress decode \
  --in data.smp \
  --out data_reconstructed.csv
```

### Quality Evaluation

```bash
# Compare original vs reconstructed
sempress eval \
  --original data.csv \
  --recon data_reconstructed.csv \
  --lock-cols user_id,timestamp
```

**Metrics:**
- **Locked columns**: Exact match rate (should be 100%)
- **Numeric columns**: RMSE, MAPE, KS-distance
- **Uncertainty**: % of cells flagged

---

## 🐍 Python API

```python
from sempress import encode_csv, decode_to_csv
from sempress.table_encoder import EncodeConfig

# Configure encoder
config = EncodeConfig(
    lock_cols=['user_id', 'timestamp'],
    residual_cols=['amount'],
    k=64,
    uncertainty_thresh=0.2
)

# Encode
compressed_blob = encode_csv('data.csv', config)

# Save to file
with open('data.smp', 'wb') as f:
    f.write(compressed_blob)

# Decode
decode_to_csv(compressed_blob, 'reconstructed.csv')
```

---

## 📊 Benchmarking

Run comprehensive benchmarks on your data:

```bash
# Generate synthetic datasets
python scripts/generate_datasets.py --rows 100000

# Run benchmarks
python scripts/comprehensive_benchmark.py --out results.json

# Generate figures
python scripts/generate_figures.py
```

**Included datasets:**
- IoT Telemetry (sensor readings)
- ML Features (user behavior)
- Financial (stock market OHLC)
- Sensor Physics (accelerometer, magnetometer)

---

## 🎯 When to Use Sempress

### ✅ Sempress Excels On:

- **High numeric density** (>60% numeric columns)
- **IoT/sensor data** (temperature, pressure, acceleration)
- **ML feature stores** (continuous features for training)
- **Financial data** (tick data, OHLC prices)
- **Large datasets** (>10K rows)

### ⚠️ Use Gzip Instead For:

- **Text-heavy tables** (<50% numeric)
- **Small tables** (<5K rows)
- **Real-time streaming** (Sempress has higher encode overhead)
- **High categorical cardinality**

---

## 📄 Research Paper

**Full paper:** [https://sempress.net/paper.pdf](https://sempress.net/paper.pdf)

**Citation:**
```bibtex
@article{sempress2025,
  title={Sempress: Semantic Compression for Numeric Tabular Data via Learned Vector Quantization},
  author={Anderson, Keaton},
  year={2025},
  note={Independent research with implementation assistance from AI coding agents},
  url={https://sempress.net}
}
```

---

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- **Streaming ingestion** (chunked encoding for >100GB files)
- **Learned entropy coding** (autoregressive priors on index sequences)
- **Time-series VQ** (segment-wise codebooks for temporal data)
- **Database integrations** (PostgreSQL extension, ClickHouse codec)
- **Text compression** (LLM-based semantic tokens for mixed data)

**How to contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📁 Repository Structure

```
sempress/
├── src/sempress/           # Core library
│   ├── table_encoder.py    # K-Means VQ encoder
│   ├── table_decoder.py    # Decoder with uncertainty
│   ├── container.py        # Msgpack + Zstd packaging
│   └── cli.py              # Command-line interface
├── scripts/                # Benchmarking & datasets
│   ├── generate_datasets.py
│   ├── comprehensive_benchmark.py
│   └── generate_figures.py
├── data/                   # Sample datasets
├── tests/                  # Unit tests
├── docs/                   # Documentation & paper
└── README.md               # This file
```

---

## 🧪 Running Tests

```bash
# Install test dependencies
pip install pytest

# Run tests
pytest tests/

# With coverage
pytest --cov=sempress tests/
```

---

## 📊 Reproducing Paper Results

```bash
# Generate datasets
python scripts/generate_datasets.py

# Run all benchmarks (takes ~10 minutes)
python scripts/comprehensive_benchmark.py

# Generate paper figures
python scripts/generate_figures.py

# Results saved to logs/ and docs/assets/
```

---

## 📈 Performance Benchmarks

**Encode time (100K rows):**
- Telemetry: 5.83s
- ML Features: 11.20s
- Financial: 9.08s

**Decode time (100K rows):**
- Telemetry: 0.28s (1.47× gzip+parse)
- ML Features: 0.55s (1.28× gzip+parse)
- Financial: 0.28s (1.17× gzip+parse)

**Memory usage:**
- Peak during encode: 2-3× original file size
- Peak during decode: 1.5-2× original file size

---

## 🐛 Known Issues

- **In-memory processing**: Files must fit in RAM (working on streaming)
- **Fixed k per column**: No adaptive sizing yet
- **CSV-only**: Parquet/Arrow support coming soon

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Star History

If you find Sempress useful, please star the repository! ⭐

---

## 📞 Contact

- **Website:** [https://sempress.net](https://sempress.net)
- **Paper:** [https://sempress.net/paper.pdf](https://sempress.net/paper.pdf)
- **Issues:** [GitHub Issues](https://github.com/jalyper/sempress-core/issues)
- **Email:** research@sempress.net

---

## 🏢 Enterprise & Managed API

Looking for a managed compression service? Visit [sempress.net](https://sempress.net) for:
- REST API for compression/decompression
- Cloud storage integration
- Automatic optimization
- Enterprise support

---

## 🙏 Acknowledgments

Independent research (no external funding).

Built with: Python, pandas, numpy, scikit-learn, msgpack, zstandard

---

**Made with ❤️ for the data compression community**
