FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade "pip>=26.2,<27" "setuptools>=84.1.0,<85" && pip install --no-cache-dir '.[dashboard]' && useradd --create-home --uid 10001 roadsense && mkdir outputs && chown roadsense outputs
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY configs ./configs
COPY scripts ./scripts
USER roadsense
EXPOSE 8000 8501
CMD ["uvicorn", "roadsense.cloud.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
