# The other recognizers (mlops/baselines): the platform's image plus the Hugging Face libraries. Build the main
# image first (docker compose build mlops-worker), then this one.
FROM madoc-mlops:latest

USER root
RUN pip install transformers==4.57.1 peft==0.17.1 accelerate==1.10.1 \
 && mkdir -p /hf-cache && chown mlops /hf-cache
USER mlops

CMD ["python", "-m", "mlops.worker"]