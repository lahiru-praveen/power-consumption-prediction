# Use a lightweight official Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency specifications and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model asset
COPY main.py .
COPY hourly_phase_model.pkl .

# Expose the standard port for Cloud Run
EXPOSE 8080

# Execute Uvicorn server bound to Cloud Run's required host layout
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]