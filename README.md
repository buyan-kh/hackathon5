# Tomorrow's Paper

> AI-Powered Market Simulation & News Generation Platform

Query anything → Agents scout, simulate, generate → Tomorrow's news, today.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.11+
- npm or bun

### Frontend (Next.js)

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to see the chat interface.

### Backend (FastAPI)

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
# Edit .env with your API keys

# Run development server
uvicorn app.main:app --reload --port 8000
```

API docs available at [http://localhost:8000/docs](http://localhost:8000/docs)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Tomorrow's Paper                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────────────────────┐   │
│  │   Next.js 14     │    │          FastAPI                 │   │
│  │   Frontend       │◄──►│          Backend                 │   │
│  │                  │ WS │                                   │   │
│  │  • Chat UI       │    │  • Agent Orchestrator            │   │
│  │  • visx Charts   │    │  • WebSocket Streaming           │   │
│  │  • Paper Layout  │    │  • REST API                      │   │
│  └──────────────────┘    └──────────────────────────────────┘   │
│                                     │                            │
│                          ┌──────────┴──────────┐                │
│                          ▼                     ▼                 │
│               ┌──────────────────┐  ┌──────────────────┐        │
│               │     Agents       │  │    Simulations   │        │
│               │                  │  │                  │        │
│               │  • Yutori        │  │  • Tonic         │        │
│               │  • Freepik       │  │    Fabricate     │        │
│               └──────────────────┘  └──────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
agentorchestration/
├── frontend/                 # Next.js 14 React application
│   ├── src/
│   │   ├── app/             # App Router pages
│   │   ├── components/      # React components
│   │   │   ├── ui/          # Base UI (Button, Card)
│   │   │   ├── chat/        # Chat interface
│   │   │   ├── simulation/  # visx charts
│   │   │   └── paper/       # Paper layout
│   │   ├── stores/          # Zustand state management
│   │   └── lib/             # Utilities
│   └── package.json
│
├── backend/                  # FastAPI Python application
│   ├── app/
│   │   ├── main.py          # FastAPI entry point
│   │   ├── api/             # REST & WebSocket routes
│   │   ├── agents/          # Agent orchestration
│   │   ├── models/          # Pydantic data models
│   │   └── core/            # Config & dependencies
│   └── requirements.txt
│
├── design.md                 # Design system documentation
└── README.md                 # This file
```

## 🎨 Design System

Following the **Minimalist Modern** design philosophy:

- **Colors**: Electric Blue gradient (`#0052FF` → `#4D7CFF`)
- **Typography**: Calistoga (display), Inter (UI), JetBrains Mono (code)
- **Animations**: Framer Motion with smooth easing
- **Charts**: visx (Airbnb) for data visualization

## 🤖 Agent Fleet

| Agent | Purpose | Status |
|-------|---------|--------|
| **Yutori** | Web scouting, news gathering | 🔧 Mock |
| **Tonic Fabricate** | Market simulation, scenario generation | 🔧 Mock |
| **Freepik** | Content and image generation | 🔧 Mock |

## 📝 API Endpoints

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Send a chat message |
| GET | `/api/chat/threads` | List conversation threads |
| POST | `/api/simulation` | Create a simulation |
| POST | `/api/paper/generate` | Generate Tomorrow's Paper |

### WebSocket

Connect to `/ws` for real-time agent updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.send(JSON.stringify({
  action: 'query',
  query: 'What if oil prices spike 40%?',
  mode: 'paper',
  use_web_search: true
}));

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Agent update:', data);
};
```

## 🔑 Environment Variables

Create `.env` in the backend directory:

```env
# Application
DEBUG=true

# API Keys
OPENAI_API_KEY=your_key_here
YUTORI_API_KEY=your_key_here
TONIC_API_KEY=your_key_here
FREEPIK_API_KEY=your_key_here
```

## 🛠️ Development

### Running Tests

```bash
# Frontend
cd frontend && npm run lint

# Backend
cd backend && pytest
```

### Building for Production

```bash
# Frontend
cd frontend && npm run build

# Backend uses uvicorn directly
```

## 📄 License

MIT