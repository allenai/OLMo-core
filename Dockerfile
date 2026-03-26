FROM python:3.10-slim

WORKDIR /app

# Copy project metadata and source files
COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/
COPY README.md .

# Install dependencies, including all optional groups such as eval
RUN pip install --upgrade pip && \
    pip install .[all]

CMD ["bash"]