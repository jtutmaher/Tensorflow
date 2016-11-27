# Base Image
FROM tensorflow/tensorflow

# Maintainer
MAINTAINER Jacob Tutmaher

# Working Directory
RUN mkdir -p /usr/local/app
WORKDIR /usr/local/app

# Copy files
COPY convolution.py .
COPY MNIST_data/ ./MNIST_data

# Open Tensorboard Port
EXPOSE 8888

# Run Command
ENTRYPOINT ["python","convolution.py"]
