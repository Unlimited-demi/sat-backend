# main.py
import httpx
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from typing import Dict, List
import uvicorn
import logging
from datetime import date, datetime, timedelta
import base64
import json

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Satellite Image Processing Pipeline API",
    description="A backend that uses the Copernicus Data Space Ecosystem to process satellite data.",
    version="13.5.0" # Version bump for robust date checking in evalscript
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Configuration ---
COPERNICUS_PROCESS_API_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"


# --- EVALSCRIPTS ---
EVALSCRIPTS: Dict[str, str] = {
    "true_color": """
        //VERSION=3
        function setup() {
            return { input: ["B04", "B03", "B02", "dataMask"], output: { bands: 3 } };
        }
        function evaluatePixel(sample) {
            if (sample.dataMask == 0) return [0, 0, 0];
            return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
        }
    """,
    "cloudless_true_color": """
        //VERSION=3
        function setup() {
            return { input: ["B04", "B03", "B02", "SCL", "dataMask"], output: { bands: 4, sampleType: "UINT8" } };
        }
        function evaluatePixel(sample) {
            // SCL values for clouds: 3(shadow), 8(medium), 9(high), 10(thin cirrus), 11(snow)
            if (sample.dataMask === 0 || sample.SCL === 3 || sample.SCL === 8 || sample.SCL === 9 || sample.SCL === 10 || sample.SCL === 11) {
                return [0, 0, 0, 0]; // Return a fully transparent pixel for clouds/snow/shadow
            }
            return [2.5 * 255 * sample.B04, 2.5 * 255 * sample.B03, 2.5 * 255 * sample.B02, 255];
        }
    """,
    "ndvi": """
        //VERSION=3
        function setup() {
            return { input: ["B04", "B08", "dataMask"], output: { bands: 4 } };
        }
        const ndvi_visualizer = [
            { "from": -1.0, "to": -0.2, "color": [0.2, 0.2, 0.8, 1.0] }, { "from": -0.2, "to": 0.0, "color": [0.8, 0.8, 0.6, 1.0] },
            { "from": 0.0, "to": 0.2, "color": [1.0, 1.0, 0.8, 1.0] }, { "from": 0.2, "to": 0.8, "color": [0.2, 0.8, 0.2, 1.0] },
            { "from": 0.8, "to": 1.0, "color": [0.0, 0.5, 0.0, 1.0] }
        ];
        function evaluatePixel(sample) {
            if (sample.dataMask == 0) return [0, 0, 0, 0];
            let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 1e-6);
            for (let i = 0; i < ndvi_visualizer.length; i++) {
                if (ndvi >= ndvi_visualizer[i].from && ndvi < ndvi_visualizer[i].to) return ndvi_visualizer[i].color;
            }
            return ndvi_visualizer[ndvi_visualizer.length - 1].color;
        }
    """,
}
# Temporal list will now use the cloudless script by default
EVALSCRIPTS["temporal_list"] = EVALSCRIPTS["cloudless_true_color"]


# --- Pydantic Models ---
class ProcessingRequest(BaseModel):
   
    lat: float
    lon: float
    startDate: str
    endDate: str
    scriptType: str

# --- Helper Functions ---
def create_bbox(lat: float, lon: float, size_deg: float = 0.05):
    return [lon - size_deg / 2, lat - size_deg / 2, lon + size_deg / 2, lat + size_deg / 2]

async def fetch_single_image(client: httpx.AsyncClient, payload: dict, headers: dict, date: str):
    try:
        response = await client.post(COPERNICUS_PROCESS_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        base64_image = base64.b64encode(response.content).decode("utf-8")
        image_data_url = f"data:image/png;base64,{base64_image}"
        return {"date": date, "image": image_data_url}
    except httpx.HTTPStatusError as e:
        logger.error(f"Failed to fetch image for date {date}: {e.response.text}")
        return {"date": date, "image": None, "error": e.response.text}

# --- Main API Endpoint ---
@app.post("/api/sentinel-hub/process", tags=["Processing Pipeline"])
async def process_image(request: ProcessingRequest):
    from oauthlib.oauth2 import BackendApplicationClient
    from requests_oauthlib import OAuth2Session

    # Your client credentials
    client_id = 'sh-fd8891d3-b21e-4495-8de2-844b29cb0dd8'
    client_secret = 'EiW3swr7TTYLqgIskYQqLfEgllEcqaGD'

    # Create a session
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)

    # Get token for the session
    token = oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                            client_secret=client_secret, include_client_id=True)

    token = token.get("access_token")
    # --- Input Validation ---
    try:
        start_date = datetime.strptime(request.startDate, "%Y-%m-%d").date()
        end_date = datetime.strptime(request.endDate, "%Y-%m-%d").date()
        if start_date > date.today() or end_date > date.today():
             raise HTTPException(status_code=400, detail="Invalid date range: Cannot request imagery from the future.")
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="Invalid date range: Start date cannot be after end date.")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")

    if request.scriptType not in EVALSCRIPTS:
        raise HTTPException(status_code=400, detail=f"Invalid scriptType specified: {request.scriptType}")

    bbox = create_bbox(request.lat, request.lon)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # --- Temporal List Workflow using a two-step Process API call ---
    if request.scriptType == "temporal_list":
        # STEP 1: Discover available dates using a lightweight metadata request
        date_discovery_evalscript = """
            //VERSION=3
            function setup() {
                return {
                    input: ["dataMask"], 
                    output: { 
                        id: "default", 
                        bands: 1, 
                        sampleType: "UINT8" 
                    }
                };
            }
            
            function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
                let uniqueDates = new Set();
                scenes.forEach(scene => {
                    // --- FIX: Added a check to ensure scene.date is a valid object ---
                    // before trying to call toISOString() on it. This makes the script
                    // resilient to scenes that might be missing metadata.
                    if (scene.date && typeof scene.date.toISOString === 'function') {
                        uniqueDates.add(scene.date.toISOString().split('T')[0]);
                    }
                });
                outputMetadata.userData = { "dates": Array.from(uniqueDates) };
            }

            function evaluatePixel(sample) {
                return [1]; // We don't care about the image, only the metadata
            }
        """

        date_discovery_payload = {
            "input": {
                "bounds": {"bbox": bbox},
                "data": [{"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": f"{request.startDate}T00:00:00Z", "to": f"{request.endDate}T23:59:59Z"}}}]
            },
            "output": {"responses": [{"identifier": "userdata", "format": {"type": "application/json"}}]},
            "evalscript": date_discovery_evalscript
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                logger.info("Discovering available dates for temporal list...")
                response = await client.post(COPERNICUS_PROCESS_API_URL, json=date_discovery_payload, headers=headers)
                response.raise_for_status()
                userdata = response.json()
                available_dates = userdata.get("dates", [])
            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=e.response.status_code, detail=f"Error during date discovery: {e.response.text}")

            if not available_dates:
                logger.info("Date discovery successful, but no dates found in the specified range.")
                return JSONResponse(content={"results": []})

            logger.info(f"Found {len(available_dates)} potential dates. Fetching up to 5 images.")
            
            # STEP 2: Fetch the actual images for the discovered dates
            tasks = []
            for scene_date in sorted(available_dates)[:5]: # Limit to 5 images
                scene_dt = datetime.strptime(scene_date, "%Y-%m-%d")
                time_from = scene_dt.strftime("%Y-%m-%dT00:00:00Z")
                time_to = scene_dt.strftime("%Y-%m-%dT23:59:59Z") # Use end of day for the time window
                
                process_payload = {
                    "input": {"bounds": {"bbox": bbox}, "data": [{"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": time_from, "to": time_to}}}]},
                    "output": {"width": 256, "height": 256, "responses": [{"identifier": "default", "format": {"type": "image/png"}}]},
                    "evalscript": EVALSCRIPTS["temporal_list"]
                }
                tasks.append(fetch_single_image(client, process_payload, headers, scene_date))
            
            results = await asyncio.gather(*tasks)
            return JSONResponse(content={"results": [res for res in results if res.get("image")]})

    # --- Single Image Workflow ---
    else:
        request_payload = {
            "input": {"bounds": {"bbox": bbox}, "data": [{"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": f"{request.startDate}T00:00:00Z", "to": f"{request.endDate}T23:59:59Z"}, "mosaickingOrder": "leastCC"}}]},
            "output": {"width": 512, "height": 512, "responses": [{"identifier": "default", "format": {"type": "image/png"}}]},
            "evalscript": EVALSCRIPTS[request.scriptType]
        }
        headers["Accept"] = "image/png"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(COPERNICUS_PROCESS_API_URL, json=request_payload, headers=headers)
            response.raise_for_status()
            return Response(content=response.content, media_type="image/png")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Error from Copernicus API: {e.response.text}")

PING_URLS = [
    "https://sat-backend-55lg.onrender.com",
    "https://sat-frontend.onrender.com/",
]

async def ping_services():
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            for url in PING_URLS:
                try:
                    resp = await client.get(url)
                    logger.info(f"Pinged {url} -> {resp.status_code}")
                except Exception as e:
                    logger.warning(f"Failed to ping {url}: {e}")
            await asyncio.sleep(20)  # wait 20 seconds

@app.on_event("startup")
async def start_pinger():
    asyncio.create_task(ping_services())
# --- Main execution block ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

