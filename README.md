# Setup git submodules
```bash
git submodule update --init --recursive
```

# Setup for local development
- Use docker compose to start the services
```bash
docker compose -f docker-compose.dev.yaml up -d
```

- Start the frontend and backend openwebui
```bash
docker exec -it openwebui bash 

## Inside the openwebui container, run the following command to start the services
bash dev.sh
```

# Setup for production 
- Use docker compose to start the services
```bash
docker compose -f docker-compose.yaml up -d
```