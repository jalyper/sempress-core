# Contributing to Sempress

Thank you for your interest in contributing to Sempress! We welcome contributions from the community.

## 🌟 Ways to Contribute

- **Bug Reports**: Found a bug? Open an issue with details to reproduce
- **Feature Requests**: Have an idea? Describe it in an issue
- **Code Contributions**: Submit pull requests with improvements
- **Documentation**: Help improve docs, examples, and tutorials
- **Benchmarks**: Share results on your datasets
- **Integrations**: Build plugins for databases, frameworks

## 🚀 Quick Start for Contributors

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/sempress.git
cd sempress

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linter
ruff check src/
```

## 📋 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

**Branch naming conventions:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `perf/` - Performance improvements

### 2. Make Your Changes

- Write clear, documented code
- Add tests for new functionality
- Update documentation as needed
- Follow the existing code style

### 3. Test Your Changes

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_encoder.py

# Check coverage
pytest --cov=sempress tests/
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "Add feature: description of what you did"
```

**Commit message guidelines:**
- Use present tense ("Add feature" not "Added feature")
- Be descriptive but concise
- Reference issues if applicable (#123)

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub.

## 🎯 Priority Areas for Contribution

### High Priority

**Streaming Ingestion**
- Process datasets in chunks
- Support files >100GB
- Maintain low memory footprint

**Learned Entropy Coding**
- Train autoregressive model on index sequences
- Improve compression by 10-20%
- Keep decode speed competitive

**Time-Series Optimization**
- Segment-wise codebooks
- Temporal pattern exploitation
- 20-40% improvement on time-series data

### Medium Priority

**Database Integrations**
- PostgreSQL extension (pg_sempress)
- ClickHouse codec
- Snowflake UDF
- Parquet codec

**Format Support**
- Read from Parquet/Arrow
- Write to Parquet/Arrow
- JSON/MessagePack input

**Adaptive Codebook Sizing**
- Per-column k selection based on variance
- Automatic parameter tuning
- Quality vs compression trade-offs

### Ongoing

**Documentation**
- Tutorials for common use cases
- API reference improvements
- Performance tuning guide

**Testing**
- Increase test coverage (target: 90%+)
- Property-based testing
- Stress testing on large datasets

**Benchmarking**
- More real-world datasets
- Comparison with other semantic compression methods
- Hardware-specific optimizations

## 📝 Code Style

We follow PEP 8 with some modifications:

```python
# Use ruff for linting
ruff check src/

# Use black for formatting (optional)
black src/

# Type hints encouraged
def encode_column(values: np.ndarray, k: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

**Key guidelines:**
- Maximum line length: 100 characters
- Use descriptive variable names
- Add docstrings to public functions
- Type hints for function signatures

## 🧪 Testing Guidelines

### Writing Tests

```python
import pytest
from sempress import encode_csv, decode_to_csv

def test_basic_encode_decode():
    """Test basic encode/decode preserves data."""
    # Arrange
    input_csv = "test_data.csv"
    
    # Act
    blob = encode_csv(input_csv, config)
    decode_to_csv(blob, "output.csv")
    
    # Assert
    # Verify reconstruction quality
```

**Test coverage expectations:**
- Core functionality: 100%
- Edge cases: Important scenarios
- Error handling: All exceptions

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_encoder.py

# With coverage
pytest --cov=sempress --cov-report=html tests/

# Verbose output
pytest -v tests/
```

## 📚 Documentation

### Docstring Format

```python
def encode_csv(input_path: str, config: EncodeConfig) -> bytes:
    """Encode a CSV file to Sempress format.
    
    Args:
        input_path: Path to input CSV file
        config: Encoder configuration with locked columns, k, etc.
        
    Returns:
        Compressed binary blob (.smp format)
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If configuration is invalid
        
    Example:
        >>> config = EncodeConfig(lock_cols=['id'], k=64)
        >>> blob = encode_csv('data.csv', config)
    """
```

### README Updates

If you add a feature, update:
- Quick Start section (if applicable)
- API documentation
- Examples
- Performance benchmarks (if relevant)

## 🐛 Bug Reports

**Good bug reports include:**

1. **Description**: Clear summary of the issue
2. **Reproduction steps**: Minimal code to reproduce
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**: Python version, OS, dependencies
6. **Error messages**: Full stack trace

**Example:**

```
## Bug: Encoder fails on columns with NaN values

**Steps to reproduce:**
1. Create CSV with NaN in numeric column
2. Run: `sempress encode --in data.csv --out data.smp`
3. Error: KeyError in table_encoder.py line 123

**Expected:** Handle NaN gracefully or provide clear error message
**Actual:** Crashes with confusing error

**Environment:**
- Python 3.11.4
- pandas 2.2.0
- Ubuntu 22.04

**Stack trace:**
[paste full error]
```

## 💡 Feature Requests

**Good feature requests include:**

1. **Use case**: Why is this feature needed?
2. **Proposed solution**: How should it work?
3. **Alternatives**: Other approaches considered?
4. **Examples**: Code showing desired API

**Example:**

```
## Feature Request: Support for Parquet input

**Use case:**
Many data pipelines use Parquet. Currently have to convert to CSV first.

**Proposed solution:**
Add --format parquet option:
`sempress encode --in data.parquet --out data.smp --format parquet`

**Implementation idea:**
Use pyarrow to read Parquet, convert to pandas DataFrame internally

**Alternative:**
Could also support Arrow IPC format
```

## 🤝 Code Review Process

### What We Look For

- **Correctness**: Does it work as intended?
- **Tests**: Are changes tested?
- **Documentation**: Is it documented?
- **Style**: Follows code conventions?
- **Performance**: No obvious slowdowns?

### Review Timeline

- **Initial response**: Within 2-3 days
- **Full review**: Within 1 week
- **Merge decision**: Depends on scope and quality

We'll provide constructive feedback and work with you to get your contribution merged!

## 📞 Getting Help

**Stuck? Need guidance?**

- Open a [Discussion](https://github.com/jalyper/sempress/discussions) for questions
- Check [existing issues](https://github.com/jalyper/sempress/issues)
- Email: research@sempress.net

## 🎉 Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- GitHub contributors page
- Research paper acknowledgments (for significant contributions)

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to Sempress!** 🚀

Your contributions help advance semantic compression research and benefit the entire data science community.
