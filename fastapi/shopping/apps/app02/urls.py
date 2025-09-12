from fastapi import APIRouter

app02 = APIRouter()

@app02.get("/jobs")
async def get_jobs(kd, xl, gj):

    return {
        "kd": kd,
        "xl": xl,
        "gj": gj
    }