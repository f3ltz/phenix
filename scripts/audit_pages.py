import fitz

doc = fitz.open('d:/Projects/Phenix/phenix_midsem_eval_report.pdf')
print(f'Total pages in generated PDF: {len(doc)}')

for i, page in enumerate(doc):
    print(f'=== Page {i+1} ===')
    images = page.get_images()
    print(f'Images: {len(images)}')
    rects = page.get_drawings()
    print(f'Drawings: {len(rects)}')
    text_len = len(page.get_text())
    print(f'Text length: {text_len} characters')
    # Check bounding box of text
    blocks = page.get_text('blocks')
    if blocks:
        min_y = min(b[1] for b in blocks)
        max_y = max(b[3] for b in blocks)
        print(f'Vertical span: y={min_y:.1f} to {max_y:.1f} (Page height: {page.rect.height:.1f})')
        if max_y > page.rect.height - 30:
            print('  WARNING: Potential overflow near bottom!')
