docker build . -t genai-lib
docker run -p 7860:7860 -e GRADIO_SERVER_NAME=0.0.0.0 genai-lib:latest