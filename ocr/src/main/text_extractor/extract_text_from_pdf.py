import io
import logging
import cv2
import pymupdf
import numpy as np
from typing import List, Tuple
from fastapi import HTTPException
import asyncio
from ocr.src.main.image_processing_utils.image_processing import resize_image_if_needed
from ocr.src.main.text_extractor.extract_text_from_image import extract_text_from_images


async def process_pdf_page(page_index: int, page) -> Tuple[int, str]:
    """
    Async task to process a single page of a PDF.
    Tries text extraction first, falls back to OCR if needed.
    Returns a tuple: (page_index, extracted_text)
    """
    try:
        # Attempt block-based text extraction
        blocks = page.get_text("blocks")
        if blocks and any(b[4].strip() for b in blocks):
            blocks.sort(key=lambda b: (b[1], b[0]))
            page_text = "\n".join([b[4] for b in blocks if b[4].strip()])
            logging.debug(f"[Text] Page {page_index}: {len(page_text)} characters extracted.")
            return page_index, page_text

        # Fallback to OCR
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = resize_image_if_needed(img)

        _, buffer = cv2.imencode(".png", img)
        stream = io.BytesIO(buffer.tobytes())

        text = await extract_text_from_images(stream)
        logging.info(f"[OCR] Page {page_index}: OCR extracted {len(text)} characters.")
        return page_index, text

    except Exception as e:
        logging.error(f"[ERROR] Page {page_index} failed: {e}")
        return page_index, ""


async def extract_text_from_pdf(pdf_stream: io.BytesIO, max_invoice_pdf_pages: int) -> str:
    """
    Extract text from a PDF using block-based extraction or OCR.
    All pages are processed concurrently and final text is returned in order.
    """
    try:
        pdf_doc = pymupdf.open(stream=pdf_stream, filetype="pdf")

        if len(pdf_doc) > max_invoice_pdf_pages:
            logging.warning(f"PDF rejected: {len(pdf_doc)} pages exceeds {max_invoice_pdf_pages}-page limit.")
            raise HTTPException(
                status_code=400,
                detail=f"PDF exceeds maximum of {max_invoice_pdf_pages} pages. This service is designed for invoices, not large documents."
            )

        if len(pdf_doc) == 0:
            raise HTTPException(status_code=400, detail="PDF file has no pages.")

        tasks = [
            asyncio.wait_for(process_pdf_page(idx, pdf_doc.load_page(idx)), timeout=15)
            for idx in range(len(pdf_doc))
        ]

        results: List[Tuple[int, str]] = await asyncio.gather(*tasks, return_exceptions=True)

        # Sort successful pages by index to maintain reading order.
        sorted_results = sorted(
            (res for res in results if not isinstance(res, Exception)),
            key=lambda x: x[0]
        )

        # Surface dropped pages rather than silently discarding them — a page
        # that timed out or errored means the extracted text is incomplete.
        dropped = [i for i, res in enumerate(results) if isinstance(res, Exception)]
        if dropped:
            logging.warning(
                f"[PDF] {len(dropped)}/{len(results)} page(s) failed or timed out "
                f"and were skipped (indices: {dropped}). Extracted text may be incomplete."
            )

        final_text = "\n".join(text for _, text in sorted_results)

        logging.info(
            f"[PDF] Completed {len(sorted_results)}/{len(results)} pages. "
            f"Final text length: {len(final_text)} characters.")
        return final_text

    except HTTPException:
        raise
    except Exception as e:
        logging.error("Error during PDF processing (parallel text+OCR)", exc_info=True)
        raise HTTPException(status_code=400, detail=f"PDF processing failed: {str(e)}")
