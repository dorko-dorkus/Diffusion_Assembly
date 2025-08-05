FROM python:3.12-slim

# Reproducible working directory
WORKDIR /app

# Install pinned dependencies
COPY env.lock ./
RUN pip install --no-cache-dir -r env.lock

# Copy source code and install package
COPY . ./
RUN pip install --no-cache-dir -e .

# Default command runs tests and smoke reproduction
CMD ["bash", "-lc", "pytest && python reproduce.py --smoke"]
