FROM python:3.9.13

WORKDIR /app
COPY . .

RUN python -m pip install --upgrade pip
RUN pip install -r requirements-dev.txt
RUN wget https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_windows_amd64.exe
RUN mv frpc_windows_amd64.exe /usr/local/lib/python3.9/site-packages/gradio/frpc_linux_aarch64_v0.2

# ENTRYPOINT ["python", "src/app.py"]
CMD ["python", "src/app.py"]