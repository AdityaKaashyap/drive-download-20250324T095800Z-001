import os
from typing import List, Optional
from fastapi import FastAPI, Request, Form, HTTPException, Depends  # Removed UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base, Session

app = FastAPI(title="Retail Inventory Management")

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database setup
DATABASE_URL = "sqlite:///./inventory.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Inventory Model
class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    # image_url = Column(String, nullable=True)  # Commented out image field

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models for request validation
class ItemCreate(BaseModel):
    name: str
    quantity: int
    price: float

class ItemUpdate(BaseModel):
    name: Optional[str] = None
    quantity: Optional[int] = None
    price: Optional[float] = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/items")
def get_items(db: Session = Depends(get_db)):
    return db.query(Item).all()

@app.post("/items")
async def add_item(
    name: str = Form(...), 
    quantity: int = Form(...), 
    price: float = Form(...), 
    db: Session = Depends(get_db)
):
    new_item = Item(name=name, quantity=quantity, price=price)  # Removed image_url field
    db.add(new_item)
    db.commit()
    db.refresh(new_item)
    return new_item

@app.put("/items/{item_id}")
async def update_item(
    item_id: int,
    name: Optional[str] = Form(None),
    quantity: Optional[int] = Form(None),
    price: Optional[float] = Form(None),
    db: Session = Depends(get_db)
):
    existing_item = db.query(Item).filter(Item.id == item_id).first()
    if not existing_item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    if name:
        existing_item.name = name
    if quantity is not None:
        existing_item.quantity = quantity
    if price is not None:
        existing_item.price = price

    db.commit()
    db.refresh(existing_item)
    return existing_item

@app.delete("/items/{item_id}")
def delete_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(Item).filter(Item.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    db.delete(item)
    db.commit()
    return {"message": "Item sold successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", reload=True)
