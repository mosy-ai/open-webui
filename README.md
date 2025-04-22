# Setup for local development
- Use docker compose to start the services
```bash
docker compose -f docker-compose.dev.yaml up -d
```
- Start the data processor 
```bash
docker exec -it data-processor bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

- Start the frontend and backend openwebui
```bash
docker exec -it openwebui bash 

## Inside the openwebui container, run the following command to start the services
bash dev.sh
```
