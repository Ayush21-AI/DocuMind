# Use the official Ollama base image
FROM ollama/ollama:0.15.6 as builder

# Set working directory
WORKDIR /app

# Start the temporary server, wait until it’s ready, download the model, verify it, and then kill the server.
RUN echo "Starting temporary Ollama server..." && \
    # Start the server in the background and capture its PID
    ollama serve & \
    SERVER_PID=$! && \
    echo "Waiting for server to be ready..." && \
    sleep 10 && \
    echo "Downloading model llama3.2:latest..." && \
    ollama pull llama3.2:latest && \
    echo "Verifying model download..." && \
    if ! ollama list | grep "llama3.2:latest"; then \
      echo "Model verification failed: llama3.2:latest not found!" && kill $SERVER_PID && exit 1; \
    else \
      echo "Model verification succeeded: llama3.2:latest is present."; \
    fi && \
    echo "Shutting down temporary server (PID: $SERVER_PID)..." && \
    kill $SERVER_PID

# Use the same image as runtime
FROM builder

# Expose the API port
EXPOSE 11434

# Set the default command to start the Ollama service
CMD ["serve"]
