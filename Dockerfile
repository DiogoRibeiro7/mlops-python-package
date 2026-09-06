# Install the same locked runtime dependencies used by the wheel gate.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm
WORKDIR /app
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt /tmp/requirements.txt
RUN uv venv /opt/venv \
    && uv pip sync --python /opt/venv/bin/python /tmp/requirements.txt
COPY dist/*.whl /tmp/wheels/
RUN uv pip install --python /opt/venv/bin/python --no-deps /tmp/wheels/*.whl \
    && uv pip check --python /opt/venv/bin/python
CMD ["bikes", "--help"]
