from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    """Say Hello to Everyone

    Returns:
        _type_: _description_
    """
    return {"message": "Hello, World"}
