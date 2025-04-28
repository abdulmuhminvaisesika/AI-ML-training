import uvicorn
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
async def index():
   return {"message": "Hello World"}
if __name__ == "__main__":
   uvicorn.run("practice:app", host="127.0.0.2", port=8001, reload=True)