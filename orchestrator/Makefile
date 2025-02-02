# Makefile for MongoDB setup

DOCKER_COMPOSE_FILE=docker-compose.yaml

# Start MongoDB
up:
	docker-compose -f $(DOCKER_COMPOSE_FILE) up -d

# Stop MongoDB
down:
	docker-compose -f $(DOCKER_COMPOSE_FILE) down

# Restart MongoDB
restart:
	docker-compose -f $(DOCKER_COMPOSE_FILE) down && docker-compose -f $(DOCKER_COMPOSE_FILE) up -d

# Clean up MongoDB containers and volumes
clean:
	docker-compose -f $(DOCKER_COMPOSE_FILE) down --volumes --remove-orphans

# Check MongoDB logs
logs:
	docker-compose -f $(DOCKER_COMPOSE_FILE) logs -f

# Execute a Mongo shell
mongo:
	docker exec -it mongodb-container mongo
