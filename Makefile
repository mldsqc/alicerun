rerun:
	docker build -t digitalize:v1 .
	docker-compose --env-file .env up

