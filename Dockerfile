# cache ML model
FROM python:3.10 AS model_cache

RUN pip install transformers torch
RUN python -c 'from transformers import pipeline;\
    model_path = "facebook/bart-large-cnn";\
    sentiment_task = pipeline("summarization", model=model_path)'


# cache build wheel for package
FROM python:3.10 AS builder

# install hatch
RUN pip install --no-cache-dir --upgrade hatch
COPY . /code

# build python package
WORKDIR /code
RUN hatch build -t wheel


FROM python:3.10 as main

# copy wheel and cache from previous stages
COPY --from=model_cache /root/.cache/ /root/.cache/
COPY --from=builder /code/dist /code/dist
RUN pip install --no-cache-dir --upgrade /code/dist/*

# run bot
CMD ["python", "-m", "lemmy_summarizer.main"]
