import { MCPConfigManager } from "@/components/mcp-config-manager";

export default function MCPSettingsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-purple-100 mb-2">MCP Configuration</h1>
        <p className="text-purple-200/60">
          Manage Model Context Protocol servers for agent tool access.
        </p>
      </div>

      <MCPConfigManager />
    </div>
  );
}

