<!-- 934c6eba-eded-4dd0-9623-4073e91d5174 7894ef38-a85f-41a4-aa3c-df7c317ff81e -->
# Dashboard & Agent Control Plane Plan

## 1. Project Setup & Configuration

- **Initialize Next.js App**: Create `dashboard` directory using Next.js App Router and TypeScript.
- **Theme Configuration**: Configure Tailwind CSS with a "Dark Nebula" theme (Black background, Purple gradients/accents).
- **Dependencies**: Install `shadcn/ui`, `@copilotkit/react-core`, `@copilotkit/react-ui`, `lucide-react`, `axios`/`tanstack-query`.

## 2. Core UI Architecture

- **Layout Shell**: Create a persistent sidebar layout with navigation.
- **Global Styles**: Implement the gradient background `bg-gradient-to-br from-black via-slate-900 to-purple-950` and text styles.
- **Components**:
    - `ui/card`: Glassmorphism style cards for widgets.
    - `ui/button`: Purple glow effects.
    - `ui/input`: Dark translucent inputs.

## 3. Feature Modules

### 3.1 Crawler Control (`src/features/crawler`)

- **Input Form**: URL entry, Batch URL upload, Depth configuration.
- **Action**: "Start Crawl" button triggering `POST /api/v1/crawl`.
- **Feedback**: Toast notifications for job start.

### 3.2 Pipeline Monitor (`src/features/pipeline`)

- **Status Dashboard**: Polling widget to show active tasks (Crawl -> Parse -> Chunk -> Embed -> Graph).
- **Metrics**: Display total nodes/entities created (mocked or fetched from KG stats).
- **Visualization**: Simple progress bars for active batch jobs.

### 3.3 Knowledge Graph Chat Agent (`src/features/agent`)

- **Copilot Integration**: Wrap app in `<CopilotKit>`.
- **Chat Interface**: Use `CopilotSidebar` or `CopilotPopup` for the persistent agent.
- **Agent Capabilities**:
    - **Command**: "Start a crawl for [url]" (Mapped to `useCopilotAction`).
    - **Query**: "What do we know about [topic]?" (Mapped to `POST /api/v1/query` or `useCopilotReadable` context).

## 4. Integration

- **API Client**: Setup `lib/api.ts` with Axios instance pointing to the FastAPI backend (default `http://localhost:8000`).
- **Copilot Backend**: Configure CopilotKit to use the local LLM (LLM Studio) or OpenAI endpoint as per project config.

## 5. Implementation Steps

### Step 1: Scaffold Dashboard

- Initialize Next.js project in `dashboard/`.
- Setup Tailwind and Shadcn.
- Create basic layout and theme.

### Step 2: Build Control Components

- Implement `CrawlerControl` component.
- Implement `PipelineStatus` component.

### Step 3: Integrate CopilotKit

- Install CopilotKit SDKs.
- Configure `<CopilotKit>` provider.
- Add `<CopilotSidebar>` to the layout.
- Define `useCopilotAction` for "Start Crawl" and "Query KG".

### Step 4: Connect to Backend

- Ensure FastAPI backend allows CORS from dashboard port.
- Wire up frontend forms to backend endpoints.

### To-dos

- [ ] Initialize Next.js app with TypeScript, Tailwind, and Shadcn UI
- [ ] Install and configure CopilotKit provider and sidebar
- [ ] Create API client and proxy route for backend communication
- [ ] Implement Crawler Control page with Copilot actions
- [ ] Implement Graph Explorer with visualization and Copilot context
- [ ] Update backend to expose CopilotKit Remote Endpoint (Mock/Stub for now)