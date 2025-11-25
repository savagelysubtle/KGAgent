# MCP Configuration for KG Agent Dashboard

This directory contains MCP (Model Context Protocol) server configuration that allows AI agents to connect to various tools and services.

## Available MCP Servers

### Internal Services
- **kg-agent-api**: Backend API for the KG Agent pipeline (crawling, upload, processing, querying)

### External MCP Servers
- **filesystem**: Access pipeline data and artifacts in `data/` directory
- **brave-search**: Web search capabilities
- **github**: GitHub repository operations
- **postgres**: Database operations
- **puppeteer**: Browser automation
- **sequential-thinking**: Complex problem-solving workflows
- **memory**: Persistent agent memory

## Environment Variables

Create a `.env` file in the dashboard directory with:

```bash
# Brave Search API
BRAVE_API_KEY=your_brave_api_key

# GitHub Access
GITHUB_TOKEN=your_github_token

# Database (if using Postgres MCP)
DATABASE_URL=postgresql://user:pass@localhost:5432/kgagent
```

## Usage

The agent can access these tools through the CopilotKit integration. The MCP servers provide capabilities like:

- **File Operations**: Read/write pipeline artifacts
- **Web Search**: Research topics before adding to knowledge graph
- **Browser Automation**: Advanced web scraping scenarios
- **Database**: Query structured data
- **Memory**: Maintain conversation context across sessions

## Extending

To add more MCP servers, add entries to `mcp.json` following this structure:

```json
{
  "your-server-name": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-package-name"],
    "env": {
      "API_KEY": "${YOUR_API_KEY}"
    },
    "description": "What this server does"
  }
}
```

For available MCP servers, see: https://github.com/modelcontextprotocol/servers

