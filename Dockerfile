FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency files first — Docker layer cache busts only on dep changes
COPY pyproject.toml uv.lock ./

# Install production dependencies only
RUN uv sync --no-dev --frozen

# Copy source
COPY src/ src/
COPY scripts/ scripts/

ENTRYPOINT ["uv", "run", "python"]
