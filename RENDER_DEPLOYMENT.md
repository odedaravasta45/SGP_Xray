# Render deployment

## Build command
```bash
pip install -r requirements.txt && python manage.py collectstatic --noinput
```

## Start command
```bash
gunicorn xray_site.wsgi:application
```

## Environment variables
- `DJANGO_SECRET_KEY`: generate a strong random secret in Render
- `DEBUG`: `False`
- `ALLOWED_HOSTS`: your Render hostname, e.g. `sgp-xray.onrender.com`
- `CSRF_TRUSTED_ORIGINS`: your Render URL, e.g. `https://sgp-xray.onrender.com`

## Important
The repository currently contains placeholder/small `.pt` files. Before inference can work, copy the real trained `detector_best.pt` into `models/detector_best.pt` and make it available to the deployed service. Do not commit large model weights to ordinary Git if they exceed your Git hosting limits; use Git LFS or model/object storage as appropriate.
