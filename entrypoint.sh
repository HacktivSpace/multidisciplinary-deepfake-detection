#!/bin/bash

set -e

set -x

echo "Waiting for database to be ready..."
while ! nc -z db 5432; do
  sleep 1
done

echo "Running database migrations..."
python manage.py migrate

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Starting application..."
exec "$@"
