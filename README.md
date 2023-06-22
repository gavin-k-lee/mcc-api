# MCC App FastAPI back-end

This repository contains the code used for a [FastAPI](https://fastapi.tiangolo.com/) server. It is deployed on [Railway](https://railway.app/).

You can access the API using HTTP requests at [https://fastapi-production-c09f.up.railway.app/](https://fastapi-production-c09f.up.railway.app/). For example:

```
curl --request POST \
  --url https://fastapi-production-c09f.up.railway.app/predict \
  --header 'Content-Type: application/json' \
  --data '{
	"query": "guitar"
}'
```

Swagger docs are available at [https://fastapi-production-c09f.up.railway.app/docs](https://fastapi-production-c09f.up.railway.app/docs)!
