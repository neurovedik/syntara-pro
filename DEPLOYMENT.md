# SYNTARA-PRO: Docker Deployment Configuration

## Quick Start with Docker

### 1. Build the Image
```bash
# Build SYNTARA-PRO image
docker build -t syntara-pro:latest .

# Build with custom tag
docker build -t syntara-pro:v1.0 .
```

### 2. Run the Container
```bash
# Basic run
docker run -p 8000:8000 syntara-pro:latest

# Run with environment variables
docker run -p 8000:8000 \
  -e SYNTARA_AGILEVEL=8 \
  -e SYNTARA_MAX_MEMORY=32 \
  -e SYNTARA_API_KEYS="your-api-key-here" \
  syntara-pro:latest

# Run with volume mount
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  syntara-pro:latest

# Run in background
docker run -d -p 8000:8000 --name syntara-pro syntara-pro:latest
```

### 3. Docker Compose
```bash
# Start with docker-compose
docker-compose up -d

# Scale to multiple instances
docker-compose up -d --scale syntara-pro=3

# Stop services
docker-compose down
```

---

## Production Deployment

### Environment Configuration
```bash
# .env file
SYNTARA_HOST=0.0.0.0
SYNTARA_PORT=8000
SYNTARA_AGILEVEL=8
SYNTARA_MAX_MEMORY=32
SYNTARA_MAX_CONCURRENT=1000
SYNTARA_API_KEYS=key1,key2,key3
SYNTARA_ENABLE_VISION=true
SYNTARA_ENABLE_RAG=true
SYNTARA_ENABLE_TRANSLATION=true
SYNTARA_LOG_LEVEL=INFO
```

### Load Balancer Configuration (Nginx)
```nginx
# nginx.conf
upstream syntara_pro {
    server syntara-pro-1:8000;
    server syntara-pro-2:8000;
    server syntara-pro-3:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://syntara_pro;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Enable WebSocket support for streaming
    location /ws {
        proxy_pass http://syntara_pro;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: syntara-pro
spec:
  replicas: 3
  selector:
    matchLabels:
      app: syntara-pro
  template:
    metadata:
      labels:
        app: syntara-pro
    spec:
      containers:
      - name: syntara-pro
        image: syntara-pro:latest
        ports:
        - containerPort: 8000
        env:
        - name: SYNTARA_AGILEVEL
          value: "8"
        - name: SYNTARA_MAX_MEMORY
          value: "32"
        - name: SYNTARA_API_KEYS
          valueFrom:
            secretKeyRef:
              name: syntara-secrets
              key: api-keys
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: syntara-pro-service
spec:
  selector:
    app: syntara-pro
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: syntara-pro-ingress
spec:
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        backend:
          service:
            name: syntara-pro-service
            port:
              number: 80
```

---

## Monitoring & Logging

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'syntara-pro'
    static_configs:
      - targets: ['syntara-pro:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "SYNTARA-PRO Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(syntara_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, syntara_response_time_seconds)"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "syntara_memory_usage_bytes"
          }
        ]
      }
    ]
  }
}
```

---

## Security Configuration

### SSL/TLS Setup
```bash
# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout syntara-pro.key -out syntara-pro.crt

# Run with SSL
docker run -p 8443:8000 \
  -v $(pwd)/syntara-pro.crt:/app/ssl/cert.pem \
  -v $(pwd)/syntara-pro.key:/app/ssl/key.pem \
  syntara-pro:latest
```

### API Rate Limiting
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  syntara-pro:
    image: syntara-pro:latest
    environment:
      - SYNTARA_RATE_LIMIT_PER_MINUTE=1000
      - SYNTARA_RATE_LIMIT_BURST=2000
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - syntara-pro
```

---

## Backup & Recovery

### Data Backup Script
```bash
#!/bin/bash
# backup_syntara.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/syntara-pro"
CONTAINER_NAME="syntara-pro"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup configuration
docker exec $CONTAINER_NAME tar -czf - /app/config | tar -xzf - -C $BACKUP_DIR/config_$DATE

# Backup data
docker exec $CONTAINER_NAME tar -czf - /app/data | tar -xzf - -C $BACKUP_DIR/data_$DATE

# Backup logs
docker exec $CONTAINER_NAME tar -czf - /app/logs | tar -xzf - -C $BACKUP_DIR/logs_$DATE

# Clean old backups (keep last 7 days)
find $BACKUP_DIR -type d -mtime +7 -exec rm -rf {} \;

echo "Backup completed: $BACKUP_DIR"
```

### Recovery Script
```bash
#!/bin/bash
# restore_syntara.sh

BACKUP_DATE=$1
BACKUP_DIR="/backups/syntara-pro"
CONTAINER_NAME="syntara-pro"

if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: $0 <backup_date>"
    echo "Available backups:"
    ls -la $BACKUP_DIR
    exit 1
fi

# Stop container
docker stop $CONTAINER_NAME

# Restore configuration
tar -czf - $BACKUP_DIR/config_$BACKUP_DATE | docker exec -i $CONTAINER_NAME tar -xzf - -C /app/config

# Restore data
tar -czf - $BACKUP_DIR/data_$BACKUP_DATE | docker exec -i $CONTAINER_NAME tar -xzf - -C /app/data

# Start container
docker start $CONTAINER_NAME

echo "Restore completed from: $BACKUP_DATE"
```

---

## Performance Optimization

### Production Dockerfile
```dockerfile
# Dockerfile.prod
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash syntara
USER syntara

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "syntara_pro_server.py", "--host", "0.0.0.0", "--port", "8000"]
```

### Multi-Stage Build
```dockerfile
# Dockerfile.multi
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.9-slim

WORKDIR /app

# Copy installed packages
COPY --from=builder /root/.local /root/.local

# Copy application
COPY . .

# Set path
ENV PATH=/root/.local/bin:$PATH

# Create non-root user
RUN useradd --create-home --shell /bin/bash syntara
USER syntara

EXPOSE 8000

CMD ["python", "syntara_pro_server.py"]
```

---

## Troubleshooting

### Common Docker Issues
```bash
# Check container logs
docker logs syntara-pro

# Check container status
docker ps -a

# Enter container for debugging
docker exec -it syntara-pro /bin/bash

# Monitor resource usage
docker stats syntara-pro

# Clean up unused resources
docker system prune -f
```

### Performance Tuning
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  syntara-pro:
    image: syntara-pro:latest
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
    environment:
      - SYNTARA_WORKERS=4
      - SYNTARA_MAX_CONNECTIONS=1000
      - SYNTARA_TIMEOUT=60
    ulimits:
      nofile:
        soft: 65536
        hard: 65536
```

---

*For complete deployment guide, see SYNTARA_PRO_MANUAL.md*
