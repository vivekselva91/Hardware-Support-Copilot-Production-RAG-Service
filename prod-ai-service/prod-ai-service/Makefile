.PHONY: install test run ui eval docker

install:
	pip install -r requirements.txt

test:
	pytest tests/ -q

run:
	uvicorn app.main:app --reload --port 8000

ui:
	streamlit run ui.py

eval:
	python -m app.core.evaluate

docker:
	docker compose up --build
