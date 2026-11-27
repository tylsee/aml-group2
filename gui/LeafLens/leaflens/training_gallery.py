"""Training-only class-reference gallery (no retrieval model or test labels)."""
from __future__ import annotations
import base64
import html
import io
from dataclasses import dataclass
from PIL import Image, ImageOps
from .data import TrainRecord


@dataclass(frozen=True)
class Reference:
    record: TrainRecord
    image_uri: str


def _thumbnail(record):
    try:
        with Image.open(record.image_path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            image.thumbnail((400, 400))
            out = io.BytesIO()
            image.save(out, 'JPEG', quality=85)
        return 'data:image/jpeg;base64,' + base64.b64encode(out.getvalue()).decode('ascii')
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def _read_examples(records, limit):
    examples = []
    for record in sorted(records, key=lambda r: r.filename):
        uri = _thumbnail(record)
        if uri is not None:
            examples.append(Reference(record, uri))
        if len(examples) >= limit:
            break
    return examples


def select_references(records, crop, disease):
    """Prefer exact label matches; never label a related image as an exact match.

    Counts refer to manifest rows, not the number of files found locally.
    References are deterministic class examples, not visual nearest neighbours.
    """
    crop, disease = str(crop).strip().casefold(), str(disease).strip().casefold()
    exact, same_disease, same_crop = [], [], []
    for r in records:
        c = r.crop.strip().casefold() == crop
        d = r.disease.strip().casefold() == disease
        if c and d:
            exact.append(r)
        elif d:
            same_disease.append(r)
        elif c:
            same_crop.append(r)
    groups = []
    exact_images = _read_examples(exact, 6)
    if exact_images:
        groups.append(('Exact crop–disease class', len(exact), exact_images))
    else:
        # Include both types if possible, to show how the two concepts were observed separately.
        groups.extend((('Same disease · different crop', len(same_disease), _read_examples(same_disease, 3)),
                       ('Same crop · different disease', len(same_crop), _read_examples(same_crop, 3))))
    return len(exact), groups


def _esc(value):
    return html.escape(str(value), quote=True)


def _label(value):
    return str(value).replace('_', ' ').strip().capitalize()


def gallery_html(records, crop, disease, root):
    exact_count, groups = select_references(records, crop, disease)
    has_exact_images = bool(exact_count and groups and groups[0][0] == 'Exact crop–disease class')
    if has_exact_images:
        description = f'{exact_count} training image(s) have the predicted crop–disease label. Showing up to six available examples.'
    elif exact_count:
        description = f'{exact_count} training row(s) match this label, but none of their image files could be opened. Related examples are shown if available.'
    elif len(records) == 8000:
        description = 'No exact training composition exists. These related examples illustrate the predicted crop and disease separately.'
    else:
        description = 'No exact composition was found in the supplied metadata. Related training examples are shown if available.'
    content = ['<section class="result-shell training-references" style="min-height:0;margin-top:10px">',
               '<div class="gallery-header" style="margin-top:0;border-top:0;padding-top:0"><div><h3>Relevant training images</h3>',
               f'<p>{_esc(description)}</p></div><span class="train-badge">TRAINING SET ONLY</span></div>']
    if not any(refs for _, _, refs in groups):
        content.append(f'<div class="gallery-message">No reference photographs could be opened. Place the original DPD-Small images in <b>{_esc(root)}</b> or set <code>training_image_root</code> in config.json. The prediction is still available.</div>')
    else:
        for title, count, refs in groups:
            content.append(f'<div style="margin:14px 0 7px;color:#405a46;font-weight:600;font-size:13px">{_esc(title)} · {len(refs)} shown / {count} training rows</div>')
            if not refs:
                content.append('<div class="gallery-message">No readable examples for this group.</div>')
                continue
            content.append('<div class="train-grid">')
            for ref in refs:
                r = ref.record
                label = f'{_label(r.crop)} / {_label(r.disease)}'
                content.append(f'<div class="train-card"><img src="{ref.image_uri}" alt="{_esc(label)}" style="height:clamp(110px,16vw,190px);object-fit:contain"><div class="train-meta"><b title="{_esc(label)}">{_esc(label)}</b><small style="overflow-wrap:anywhere" title="{_esc(r.filename)}">{_esc(r.filename)}</small></div></div>')
            content.append('</div>')
    content.append('<p class="reference-footnote">Examples are selected by training labels, not ranked visual similarity. They illustrate the model prediction; they do not verify that the uploaded image is correctly classified. No validation or official-test images are selected.</p></section>')
    return ''.join(content)
