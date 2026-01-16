"""
WebSocket handler for real-time streaming of agent updates.
"""

import logging
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.agents.orchestrator import orchestrator, OrchestratorEvent

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_event(self, websocket: WebSocket, event: OrchestratorEvent):
        """Send an orchestrator event to a specific client."""
        if websocket.client_state != WebSocketState.CONNECTED:
            return
        
        try:
            data = {
                "event_type": event.event_type,
                "agent": event.agent_type.value if event.agent_type else None,
                "data": event.data,
                "timestamp": event.timestamp.isoformat(),
            }
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Failed to send event: {e}")


manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming agent updates.
    
    Client sends:
    {
        "action": "query",
        "query": "What if oil prices spike 40%?",
        "mode": "paper",
        "use_web_search": true
    }
    
    Server streams:
    {
        "event_type": "agent_update" | "start" | "complete" | "error",
        "agent": "yutori" | "fabricate" | "freepik" | null,
        "data": {...},
        "timestamp": "2024-01-17T12:00:00Z"
    }
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Wait for client message
            raw_data = await websocket.receive_text()
            
            try:
                import json
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "event_type": "error",
                    "data": {"message": "Invalid JSON"},
                })
                continue
            
            action = data.get("action")
            logger.info(f"Received action: {action}")
            
            if action == "query":
                query = data.get("query", "")
                mode = data.get("mode", "paper")
                use_web_search = data.get("use_web_search", True)
                
                logger.info(f"Processing query: {query[:50]}...")
                
                try:
                    # Process query and stream events
                    async for event in orchestrator.process_query(
                        query=query,
                        mode=mode,
                        use_web_search=use_web_search,
                    ):
                        await manager.send_event(websocket, event)
                        logger.debug(f"Sent event: {event.event_type}")
                except Exception as e:
                    logger.exception(f"Error processing query: {e}")
                    await websocket.send_json({
                        "event_type": "error",
                        "agent": None,
                        "data": {"message": f"Processing error: {str(e)}"},
                    })
            
            elif action == "stop":
                orchestrator.reset()
                await websocket.send_json({
                    "event_type": "stopped",
                    "agent": None,
                    "data": {"message": "Processing stopped by user"},
                })
            
            elif action == "ping":
                await websocket.send_json({
                    "event_type": "pong",
                    "agent": None,
                    "data": {},
                })
            
            else:
                await websocket.send_json({
                    "event_type": "error",
                    "agent": None,
                    "data": {"message": f"Unknown action: {action}"},
                })
    
    except WebSocketDisconnect:
        logger.info("Client disconnected normally")
        manager.disconnect(websocket)
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "event_type": "error",
                "agent": None,
                "data": {"message": str(e)},
            })
        except:
            pass
        manager.disconnect(websocket)
