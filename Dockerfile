FROM python:3.11-slim
# Set the working directory
WORKDIR /app  
# Copy the requirements file      
COPY . /app

RUN apt update -y && apt install awscli -y
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "app.py"]