# Use an official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the port Flask runs on
EXPOSE 8080

# Command to run the app
CMD ["python", "app.py"]
