from __future__ import annotations

import os
import mimetypes
from typing import List, Optional, Dict, Any
import math
import re
from difflib import SequenceMatcher
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, BackgroundTasks
from fastapi.responses import RedirectResponse
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, text, func
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Document
from ..schemas import DocumentOut
from ..ocr import extract_text_from_pdf, extract_text_from_image, extract_text_from_word
import asyncio
import json

router = APIRouter()

# Khởi tạo templates trong main và mount, ở đây chỉ lấy đường dẫn
from ..main import templates, upload_dir  # noqa: E402

ALLOWED_IMAGE = {"image/jpeg", "image/png"}
ALLOWED_PDF = {"application/pdf"}
ALLOWED_WORD = {
    "application/Word",  # .docx
    "application/msword"  # .doc (legacy)
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


# Advanced Search Functions
def fuzzy_match(query: str, text: str, threshold: float = 0.6) -> bool:
    """Fuzzy matching using sequence matcher"""
    if not query or not text:
        return False
    return SequenceMatcher(None, query.lower(), text.lower()).ratio() >= threshold

def parse_boolean_query(query: str) -> Dict[str, Any]:
    """Parse boolean search query with AND, OR, NOT operators"""
    query = query.strip()
    
    # Handle NOT operator
    not_terms = []
    if ' NOT ' in query.upper():
        parts = re.split(r'\s+NOT\s+', query, flags=re.IGNORECASE)
        if len(parts) == 2:
            query = parts[0].strip()
            not_terms = [term.strip() for term in parts[1].split() if term.strip()]
    
    # Handle OR operator
    or_terms = []
    if ' OR ' in query.upper():
        or_terms = [term.strip() for term in re.split(r'\s+OR\s+', query, flags=re.IGNORECASE) if term.strip()]
        query = ""
    else:
        # Handle AND operator (default)
        and_terms = [term.strip() for term in re.split(r'\s+AND\s+', query, flags=re.IGNORECASE) if term.strip()]
        if len(and_terms) > 1:
            query = ""
        else:
            and_terms = [query] if query else []
    
    return {
        'query': query,
        'and_terms': and_terms if 'and_terms' in locals() else [],
        'or_terms': or_terms,
        'not_terms': not_terms
    }

def parse_wildcard_query(query: str) -> str:
    """Convert wildcard query to SQL LIKE pattern"""
    if not query:
        return query
    
    # Escape SQL special characters except wildcards
    query = query.replace('\\', '\\\\')
    query = query.replace('%', '\\%')
    query = query.replace('_', '\\_')
    
    # Convert wildcards
    query = query.replace('*', '%')
    query = query.replace('?', '_')
    
    return f"%{query}%"

def calculate_relevance_score(query: str, title: str, content: str, tags: str = "") -> float:
    """Calculate relevance score for search results"""
    if not query:
        return 0.0
    
    query_lower = query.lower()
    title_lower = title.lower() if title else ""
    content_lower = content.lower() if content else ""
    tags_lower = tags.lower() if tags else ""
    
    score = 0.0
    
    # Exact match in title (highest weight)
    if query_lower in title_lower:
        score += 10.0
    
    # Exact match in content
    if query_lower in content_lower:
        score += 5.0
    
    # Exact match in tags
    if query_lower in tags_lower:
        score += 8.0
    
    # Fuzzy match in title
    if fuzzy_match(query, title, 0.7):
        score += 6.0
    
    # Fuzzy match in content
    if fuzzy_match(query, content, 0.7):
        score += 3.0
    
    # Word count bonus
    query_words = query_lower.split()
    title_words = title_lower.split()
    content_words = content_lower.split()
    
    for word in query_words:
        if word in title_words:
            score += 2.0
        if word in content_words:
            score += 1.0
    
    return score

def save_upload(file: UploadFile, dest_dir: str) -> str:
    """Save uploaded file with safe name handling"""
    os.makedirs(dest_dir, exist_ok=True)
    filename = file.filename or "uploaded"
    # tránh trùng tên
    base, ext = os.path.splitext(filename)
    counter = 1
    safe_name = filename
    while os.path.exists(os.path.join(dest_dir, safe_name)):
        safe_name = f"{base}_{counter}{ext}"
        counter += 1
    filepath = os.path.join(dest_dir, safe_name)
    with open(filepath, "wb") as f:
        f.write(file.file.read())
    return safe_name


@router.get("/")
async def list_documents(
    request: Request,
    db: Session = Depends(get_db),
    q: Optional[str] = Query(None, description="Search in content"),
    title: Optional[str] = Query(None, description="Filter by title"),
    tags: Optional[str] = Query(None, description="Filter by tags"),
    type: Optional[str] = Query(None, description="Filter by document type (pdf/image)"),
    fuzzy: bool = Query(False, description="Enable fuzzy search"),
    wildcard: bool = Query(False, description="Enable wildcard search"),
    boolean: bool = Query(False, description="Enable boolean operators"),
    sort_by: str = Query("relevance", description="Sort by: relevance, date, title"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(5, ge=1, le=5, description="Items per page - max 5"),
):
    """List documents with search and pagination"""
    try:
        # Base query
        base_query = db.query(Document)
        conditions: List[str] = []
        params = {}

        # Apply search filters with advanced features
        if q:
            if boolean:
                # Parse boolean query
                parsed = parse_boolean_query(q)
                if parsed['and_terms']:
                    and_conditions = []
                    for i, term in enumerate(parsed['and_terms']):
                        if wildcard:
                            term = parse_wildcard_query(term)
                        and_conditions.append(f"(title LIKE :and_term_{i} OR content LIKE :and_term_{i} OR tags LIKE :and_term_{i})")
                        params[f"and_term_{i}"] = f"%{term}%"
                    if and_conditions:
                        conditions.append(f"({' AND '.join(and_conditions)})")
                
                if parsed['or_terms']:
                    or_conditions = []
                    for i, term in enumerate(parsed['or_terms']):
                        if wildcard:
                            term = parse_wildcard_query(term)
                        or_conditions.append(f"(title LIKE :or_term_{i} OR content LIKE :or_term_{i} OR tags LIKE :or_term_{i})")
                        params[f"or_term_{i}"] = f"%{term}%"
                    if or_conditions:
                        conditions.append(f"({' OR '.join(or_conditions)})")
                
                if parsed['not_terms']:
                    for i, term in enumerate(parsed['not_terms']):
                        if wildcard:
                            term = parse_wildcard_query(term)
                        conditions.append(f"(title NOT LIKE :not_term_{i} AND content NOT LIKE :not_term_{i} AND tags NOT LIKE :not_term_{i})")
                        params[f"not_term_{i}"] = f"%{term}%"
            else:
                # Regular search
                if wildcard:
                    q = parse_wildcard_query(q)
                    conditions.append("(title LIKE :q OR content LIKE :q OR tags LIKE :q)")
                    params["q"] = q
                else:
                    # Use FULLTEXT if available
                    conditions.append("MATCH(title, content, tags) AGAINST (:q IN NATURAL LANGUAGE MODE)")
                    params["q"] = q
        
        if title:
            if wildcard:
                title = parse_wildcard_query(title)
            conditions.append("title LIKE :title")
            params["title"] = f"%{title}%"
        if tags:
            if wildcard:
                tags = parse_wildcard_query(tags)
            conditions.append("tags LIKE :tags")
            params["tags"] = f"%{tags}%"
        
        if type:
            if type == 'pdf':
                conditions.append("mime_type LIKE :mime_type")
                params["mime_type"] = "application/pdf"
            elif type == 'image':
                conditions.append("mime_type IN ('image/jpeg', 'image/png')")

        # Build query
        if conditions:
            where_sql = " AND ".join(conditions)
            # Count total first
            count_sql = f"SELECT COUNT(*) FROM documents WHERE {where_sql}"
            total = db.execute(text(count_sql), params).scalar()
            
            # Determine sort order
            order_clause = "created_at DESC"
            if sort_by == "relevance" and q:
                # For relevance sorting, we'll do it in Python after fetching
                order_clause = "created_at DESC"
            elif sort_by == "title":
                order_clause = "title ASC"
            elif sort_by == "date":
                order_clause = "created_at DESC"
            
            # Then get paginated results
            stmt = text(
                f"SELECT * FROM documents WHERE {where_sql} "
                f"ORDER BY {order_clause} LIMIT :limit OFFSET :offset"
            )
            params["limit"] = limit
            params["offset"] = (page - 1) * limit
            rows = db.execute(stmt, params).mappings().all()
            docs = [Document(**dict(r)) for r in rows]
            
            # Apply relevance scoring and sorting if needed
            if sort_by == "relevance" and q and docs:
                for doc in docs:
                    doc.relevance_score = calculate_relevance_score(
                        q, doc.title or "", doc.content or "", doc.tags or ""
                    )
                docs.sort(key=lambda x: getattr(x, 'relevance_score', 0), reverse=True)
        else:
            # No search conditions - use ORM
            total = base_query.count()
            docs = (
                base_query.order_by(Document.created_at.desc())
                .offset((page - 1) * limit)
                .limit(limit)
                .all()
            )

        total_pages = math.ceil(total / limit)
        
        # Auto-redirect logic
        if total > 0:
            # If current page is beyond total pages, redirect to last page
            if page > total_pages:
                query_params = request.query_params
                new_params = query_params.copy()
                new_params["page"] = total_pages
                redirect_url = f"/documents/?{new_params}"
                return RedirectResponse(url=redirect_url)
            
            # If current page is empty but there are documents, redirect to page 1
            if not docs and page > 1:
                query_params = request.query_params
                new_params = query_params.copy()
                new_params["page"] = 1
                redirect_url = f"/documents/?{new_params}"
                return RedirectResponse(url=redirect_url)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "documents": docs,
                "q": q or "",
                "title": title or "",
                "tags": tags or "",
                "type": type or "",
                "page": page,
                "limit": limit,
                "total": total,
                "total_pages": total_pages,
            },
        )
    except Exception as e:
        print(f"Error in list_documents: {e}")
        # Return empty page with error message
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "documents": [],
                "q": q or "",
                "title": title or "",
                "tags": tags or "",
                "type": type or "",
                "page": 1,
                "limit": limit,
                "total": 0,
                "total_pages": 0,
                "error": f"Database error: {str(e)}"
            },
        )


@router.get("/{doc_id}")
async def document_detail(
    doc_id: int,
    request: Request,
    db: Session = Depends(get_db),
    q: Optional[str] = Query(None),
):
    """Get document details"""
    doc = db.get(Document, doc_id)
    if not doc:
        request.state.flash("Không tìm thấy tài liệu", "error")
        return RedirectResponse(url="/documents")
    return templates.TemplateResponse(
        "detail.html",
        {"request": request, "doc": doc, "q": q or ""}
    )


@router.get("/{doc_id}/download")
async def download_document(doc_id: int, request: Request, db: Session = Depends(get_db)):
    """Download document file"""
    doc = db.get(Document, doc_id)
    if not doc:
        request.state.flash("Không tìm thấy tài liệu", "error")
        return RedirectResponse(url="/documents")
    
    filepath = os.path.join(upload_dir, doc.filename)
    if not os.path.exists(filepath):
        request.state.flash("File không tồn tại", "error")
        return RedirectResponse(url="/documents")
    
    mime = doc.mime_type or mimetypes.guess_type(filepath)[0] or "application/octet-stream"
    return FileResponse(filepath, media_type=mime, filename=doc.filename)


@router.post("/upload")
async def upload_document(
    request: Request,
    title: str = Form(...),
    tags: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload new document"""
    # Validate file
    if not file.content_type:
        raise HTTPException(status_code=400, detail="Thiếu content-type")
    if file.content_type not in ALLOWED_IMAGE and file.content_type not in ALLOWED_PDF and file.content_type not in ALLOWED_WORD:
        raise HTTPException(status_code=400, detail="Định dạng không hỗ trợ. Chỉ hỗ trợ PDF, ảnh (JPEG/PNG) và Word (.docx/.doc)")

    # Check file size
    file_contents = file.file.read()
    if len(file_contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File vượt quá 10MB")
    file.file.seek(0)  # Reset file pointer

    try:
        saved_name = save_upload(file, upload_dir)
        filepath = os.path.join(upload_dir, saved_name)

        # Extract text content with fast mode for better speed
        content = ""
        try:
            if file.content_type in ALLOWED_PDF:
                content = extract_text_from_pdf(filepath)
            elif file.content_type in ALLOWED_IMAGE:
                content = extract_text_from_image(filepath, fast_mode=True)
            elif file.content_type in ALLOWED_WORD:
                content = extract_text_from_word(filepath)
        except Exception as e:
            # Log error but continue
            print(f"Error extracting content: {e}")
            request.state.flash("Không thể trích xuất nội dung từ file, nhưng file đã được lưu", "warning")

        doc = Document(
            title=title,
            filename=saved_name,
            content=content,
            tags=tags,
            mime_type=file.content_type,
            filesize=os.path.getsize(filepath),
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        
        request.state.flash("Tải lên tài liệu thành công", "success")
        return {"id": doc.id, "message": "Tải lên thành công"}

    except Exception as e:
        # Clean up file if saved
        if "saved_name" in locals():
            try:
                os.remove(os.path.join(upload_dir, saved_name))
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Lỗi khi tải lên: {str(e)}")


@router.post("/{doc_id}/reprocess")
async def reprocess_document(
    doc_id: int, 
    request: Request, 
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """Reprocess document OCR with progress tracking"""
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu")
    
    filepath = os.path.join(upload_dir, doc.filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File không tồn tại")
    
    # Start background OCR processing
    background_tasks.add_task(process_ocr_with_progress, doc_id, filepath, doc.mime_type, db)
    
    return {"message": "Đang xử lý OCR...", "status": "processing"}

async def process_ocr_with_progress(doc_id: int, filepath: str, mime_type: str, db: Session):
    """Background OCR processing with progress updates"""
    try:
        # Simulate progress updates
        progress = 0
        content = ""
        
        if mime_type in ALLOWED_PDF:
            # PDF processing with progress simulation
            progress = 10
            content = extract_text_from_pdf(filepath)
            progress = 90
        elif mime_type in ALLOWED_IMAGE:
            # Image processing
            progress = 20
            content = extract_text_from_image(filepath, fast_mode=True)
            progress = 80
        elif mime_type in ALLOWED_WORD:
            # Word processing
            progress = 30
            content = extract_text_from_word(filepath)
            progress = 70
        
        # Update document in database
        doc = db.get(Document, doc_id)
        if doc:
            doc.content = content
            db.commit()
            progress = 100
        
    except Exception as e:
        print(f"Error in background OCR processing: {e}")

@router.get("/{doc_id}/progress")
async def get_ocr_progress(doc_id: int, db: Session = Depends(get_db)):
    """Get OCR processing progress"""
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu")
    
    # Simple progress simulation - in real implementation, you'd use Redis or similar
    return {"progress": 100, "status": "completed", "content_length": len(doc.content or "")}

@router.delete("/{doc_id}")
async def delete_document(doc_id: int, request: Request, db: Session = Depends(get_db)):
    """Delete document and its file"""
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu")
    
    # Delete file first
    filepath = os.path.join(upload_dir, doc.filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Error deleting file: {e}")
            request.state.flash("Không thể xóa file nhưng đã xóa thông tin tài liệu", "warning")
    
    # Then delete DB record
    db.delete(doc)
    db.commit()
    
    request.state.flash("Đã xóa tài liệu thành công", "success")
    return {"message": "Đã xóa tài liệu thành công"}

@router.get("/api/search/suggestions")
async def get_search_suggestions(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Number of suggestions"),
    db: Session = Depends(get_db)
):
    """Get search suggestions based on query"""
    try:
        if len(q) < 2:
            return {"suggestions": []}
        
        # Get suggestions from titles and tags
        suggestions = []
        
        # Title suggestions
        title_suggestions = db.execute(
            text("SELECT DISTINCT title FROM documents WHERE title LIKE :q LIMIT :limit"),
            {"q": f"%{q}%", "limit": limit // 2}
        ).scalars().all()
        
        for title in title_suggestions:
            if title and q.lower() in title.lower():
                suggestions.append({
                    "text": title,
                    "type": "title",
                    "highlight": title.lower().replace(q.lower(), f"<mark>{q}</mark>")
                })
        
        # Tag suggestions
        tag_suggestions = db.execute(
            text("SELECT DISTINCT tags FROM documents WHERE tags LIKE :q LIMIT :limit"),
            {"q": f"%{q}%", "limit": limit // 2}
        ).scalars().all()
        
        for tags in tag_suggestions:
            if tags:
                for tag in tags.split(','):
                    tag = tag.strip()
                    if q.lower() in tag.lower():
                        suggestions.append({
                            "text": tag,
                            "type": "tag",
                            "highlight": tag.lower().replace(q.lower(), f"<mark>{q}</mark>")
                        })
        
        # Remove duplicates and limit results
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion["text"] not in seen:
                seen.add(suggestion["text"])
                unique_suggestions.append(suggestion)
                if len(unique_suggestions) >= limit:
                    break
        
        return {"suggestions": unique_suggestions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting suggestions: {str(e)}")

@router.get("/api/search/history")
async def get_search_history(
    limit: int = Query(10, ge=1, le=50, description="Number of recent searches"),
    db: Session = Depends(get_db)
):
    """Get recent search history"""
    try:
        # This would typically be stored in a separate table
        # For now, return empty list
        return {"history": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting search history: {str(e)}")

@router.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get document statistics"""
    try:
        total_docs = db.query(Document).count()
        limit = 5  # Max 5 documents per page
        total_pages = math.ceil(total_docs / limit) if total_docs > 0 else 1
        
        return {
            "total": total_docs,
            "displayed": min(limit, total_docs),  # Max 5 displayed
            "pages": total_pages,
            "currentPage": 1,
            "limit": limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")
