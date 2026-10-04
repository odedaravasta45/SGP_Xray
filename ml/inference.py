from pathlib import Path
import cv2
import numpy as np
import torch

from .model import build_detector, CLASS_NAMES, IMG_SIZE_DET, NUM_FINDINGS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'detector_best.pt'
SCORE_THRESHOLD = 0.05
NMS_THRESHOLD = 0.40
MAX_DETECTIONS = 100

_MODEL = None

class ModelNotReadyError(RuntimeError):
    pass


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not MODEL_PATH.exists():
        raise ModelNotReadyError(
            'The trained model weights are not included yet. Copy detector_best.pt from your trained notebook output into models/detector_best.pt, then restart Django.'
        )
    model = build_detector().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()
    _MODEL = model
    return model


def _letterbox(img, size):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, scale, left, top


def _tensor(img):
    x = img.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).float()


def _iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:,0]); y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2]); y2 = np.minimum(box[3], boxes[:,3])
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    area1 = max(0, box[2]-box[0]) * max(0, box[3]-box[1])
    area2 = np.maximum(0, boxes[:,2]-boxes[:,0]) * np.maximum(0, boxes[:,3]-boxes[:,1])
    return inter / np.maximum(area1 + area2 - inter, 1e-9)


def _nms(boxes, scores, threshold):
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1: break
        rest = order[1:]
        order = rest[_iou(boxes[i], boxes[rest]) < threshold]
    return keep


def analyze_xray(input_path, results_dir, result_id):
    model = _load_model()
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Could not read the uploaded image.')
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = rgb.shape[:2]
    canvas, scale, left, top = _letterbox(rgb, IMG_SIZE_DET)
    x = _tensor(canvas).to(DEVICE)
    with torch.no_grad():
        pred = model([x])[0]

    boxes = pred['boxes'].detach().cpu().numpy()
    scores = pred['scores'].detach().cpu().numpy()
    labels = pred['labels'].detach().cpu().numpy() - 1

   # DEBUG: show what the model is actually producing
    print("=" * 60)
    print("RAW MODEL OUTPUT")
    print("Number of predictions:", len(scores))

    if len(scores) > 0:
        top_idx = np.argsort(scores)[::-1][:10]

        for i in top_idx:
            print(
                f"label={labels[i]}, "
                f"class={CLASS_NAMES[labels[i]] if 0 <= labels[i] < len(CLASS_NAMES) else 'UNKNOWN'}, "
                f"score={scores[i]:.6f}"
            )

        print("Maximum confidence:", float(scores.max()))
    else:
        print("MODEL RETURNED ZERO PREDICTIONS")

    print("=" * 60)

    mask = (
        (scores >= SCORE_THRESHOLD)
        & (labels >= 0)
    & (labels < NUM_FINDINGS)
)
    boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

    final = []
    for cls in sorted(set(labels.tolist())):
        idx = np.where(labels == cls)[0]
        for k in _nms(boxes[idx], scores[idx], NMS_THRESHOLD):
            j = idx[k]
            b = boxes[j].copy()
            b[[0,2]] = (b[[0,2]] - left) / scale
            b[[1,3]] = (b[[1,3]] - top) / scale
            b[[0,2]] = np.clip(b[[0,2]], 0, w0-1)
            b[[1,3]] = np.clip(b[[1,3]], 0, h0-1)
            final.append((float(scores[j]), int(labels[j]), b))

    final.sort(key=lambda z: z[0], reverse=True)
    final = final[:MAX_DETECTIONS]

    annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    detections = []
    for score, label, b in final:
        x1,y1,x2,y2 = [int(round(v)) for v in b]
        x1=max(0,min(x1,w0-1)); y1=max(0,min(y1,h0-1)); x2=max(x1+1,min(x2,w0-1)); y2=max(y1+1,min(y2,h0-1))
        text = f'{CLASS_NAMES[label]}  {score*100:.1f}%'
        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,210,120), 3)
        cv2.rectangle(annotated, (x1,y1-28), (min(w0-1,x1+max(180,len(text)*10)),y1), (0,210,120), -1)
        cv2.putText(annotated, text, (x1+5,max(18,y1-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
        detections.append({'label': CLASS_NAMES[label], 'class_id': label, 'confidence': round(score,4), 'confidence_percent': round(score*100,2), 'box': [x1,y1,x2,y2]})

    out_path = Path(results_dir) / f'{result_id}_annotated.jpg'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    return {'annotated_path': str(out_path), 'detections': detections, 'message': 'Analysis completed.' if detections else 'No finding detected above the current confidence threshold.'}
