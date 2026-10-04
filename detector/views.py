import json
import os
import uuid
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse, Http404
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from PIL import Image, UnidentifiedImageError

from ml.inference import ModelNotReadyError, analyze_xray

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}


def upload_view(request):
    return render(request, 'upload.html')


def _safe_image(uploaded):
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError('Please upload a PNG, JPG, JPEG, or WEBP image.')
    if uploaded.size > settings.MAX_UPLOAD_SIZE:
        raise ValueError('Image is too large. Maximum size is 15 MB.')
    try:
        with Image.open(uploaded) as im:
            im.verify()
    except (UnidentifiedImageError, OSError):
        raise ValueError('The uploaded file is not a valid image.')
    return suffix


@require_http_methods(['POST'])
def analyze_view(request):
    uploaded = request.FILES.get('xray')
    if not uploaded:
        return render(request, 'upload.html', {'error': 'Please choose an X-ray image.'}, status=400)

    try:
        suffix = _safe_image(uploaded)
    except ValueError as exc:
        return render(request, 'upload.html', {'error': str(exc)}, status=400)

    result_id = uuid.uuid4().hex
    input_name = f'{result_id}{suffix}'
    input_path = Path(settings.MEDIA_ROOT) / 'uploads' / input_name
    input_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open('wb+') as destination:
        for chunk in uploaded.chunks():
            destination.write(chunk)

    try:
        result = analyze_xray(str(input_path), str(settings.MEDIA_ROOT / 'results'), result_id)
    except ModelNotReadyError as exc:
        return render(request, 'upload.html', {'error': str(exc), 'model_missing': True}, status=503)
    except Exception as exc:
        return render(request, 'upload.html', {'error': f'Analysis failed: {exc}'}, status=500)

    result_path = Path(result['annotated_path'])
    payload = {
        'result_id': result_id,
        'original_name': uploaded.name,
        'annotated_url': f'{settings.MEDIA_URL}results/{result_path.name}',
        'input_url': f'{settings.MEDIA_URL}uploads/{input_name}',
        'detections': result['detections'],
        'message': result['message'],
    }
    json_path = Path(settings.MEDIA_ROOT) / 'results' / f'{result_id}.json'
    json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return redirect('result', result_id=result_id)


def result_view(request, result_id):
    result_file = Path(settings.MEDIA_ROOT) / 'results' / f'{result_id}.json'
    if not result_file.exists():
        raise Http404('Result not found.')
    data = json.loads(result_file.read_text(encoding='utf-8'))
    return render(request, 'result.html', data)
