"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";
import { Plus, Trash2, Save, Download, Upload, Server, Loader2 } from "lucide-react";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";

interface MCPServer {
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  url?: string;
  description: string;
  endpoints?: Record<string, any>;
}

interface MCPConfig {
  mcpServers: Record<string, MCPServer>;
}

export function MCPConfigManager() {
  const [config, setConfig] = useState<MCPConfig>({ mcpServers: {} });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [selectedServer, setSelectedServer] = useState<string | null>(null);
  const [editingServer, setEditingServer] = useState<MCPServer | null>(null);
  const [newServerName, setNewServerName] = useState("");

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    try {
      const response = await fetch("/api/mcp-config");
      const data = await response.json();
      setConfig(data);
    } catch (error) {
      console.error("Failed to load config:", error);
      toast.error("Failed to load MCP configuration");
    } finally {
      setLoading(false);
    }
  };

  const saveConfig = async () => {
    setSaving(true);
    try {
      const response = await fetch("/api/mcp-config", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        throw new Error("Failed to save config");
      }

      toast.success("Configuration saved successfully.");
    } catch (error) {
      console.error("Failed to save config:", error);
      toast.error("Failed to save configuration");
    } finally {
      setSaving(false);
    }
  };

  const addServer = () => {
    if (!newServerName) {
      toast.error("Please enter a server name");
      return;
    }
    if (config.mcpServers[newServerName]) {
      toast.error("Server name already exists");
      return;
    }

    const newServer: MCPServer = {
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-example"],
      description: "New MCP server",
      env: {}
    };

    setConfig({
      ...config,
      mcpServers: {
        ...config.mcpServers,
        [newServerName]: newServer
      }
    });
    setNewServerName("");
    toast.success("Server added");
  };

  const deleteServer = (serverName: string) => {
    const { [serverName]: _, ...rest } = config.mcpServers;
    setConfig({ mcpServers: rest });
    toast.success("Server removed");
  };

  const updateServer = (serverName: string, updates: Partial<MCPServer>) => {
    setConfig({
      ...config,
      mcpServers: {
        ...config.mcpServers,
        [serverName]: {
          ...config.mcpServers[serverName],
          ...updates
        }
      }
    });
  };

  const openEditSheet = (serverName: string) => {
    setSelectedServer(serverName);
    setEditingServer({ ...config.mcpServers[serverName] });
  };

  const saveServerEdit = () => {
    if (selectedServer && editingServer) {
      updateServer(selectedServer, editingServer);
      setSelectedServer(null);
      setEditingServer(null);
      toast.success("Server updated");
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-12">
        <Loader2 className="h-8 w-8 animate-spin text-purple-400" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header Actions */}
      <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-purple-100">Quick Actions</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-3">
          <Button
            onClick={saveConfig}
            disabled={saving}
            className="bg-purple-600 hover:bg-purple-700"
          >
            {saving ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Save className="mr-2 h-4 w-4" />}
            Save Config
          </Button>
        </CardContent>
      </Card>

      {/* Add New Server */}
      <Card className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-purple-100">Add MCP Server</CardTitle>
          <CardDescription className="text-purple-200/60">
            Configure a new Model Context Protocol server
          </CardDescription>
        </CardHeader>
        <CardContent className="flex gap-3">
          <Input
            placeholder="Server name (e.g., my-custom-server)"
            value={newServerName}
            onChange={(e) => setNewServerName(e.target.value)}
            className="bg-black/20 border-purple-500/20 text-purple-100"
          />
          <Button onClick={addServer} className="bg-purple-600 hover:bg-purple-700 whitespace-nowrap">
            <Plus className="mr-2 h-4 w-4" />
            Add Server
          </Button>
        </CardContent>
      </Card>

      {/* Server List */}
      <div className="grid gap-4 md:grid-cols-2">
        {Object.entries(config.mcpServers).map(([name, server]) => (
          <Card key={name} className="bg-black/40 border-purple-500/20 backdrop-blur-sm">
            <CardHeader>
              <div className="flex items-start justify-between">
                <div className="space-y-1">
                  <CardTitle className="text-purple-100 flex items-center gap-2">
                    <Server className="h-5 w-5" />
                    {name}
                  </CardTitle>
                  <CardDescription className="text-purple-200/60 text-sm">
                    {server.description}
                  </CardDescription>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => deleteServer(name)}
                  className="text-red-400 hover:text-red-300 hover:bg-red-500/10"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-2">
              {server.command && (
                <div className="text-xs">
                  <span className="text-purple-200/40">Command:</span>
                  <code className="ml-2 text-purple-300 bg-purple-500/10 px-2 py-1 rounded">
                    {server.command}
                  </code>
                </div>
              )}
              {server.url && (
                <div className="text-xs">
                  <span className="text-purple-200/40">URL:</span>
                  <code className="ml-2 text-purple-300 bg-purple-500/10 px-2 py-1 rounded">
                    {server.url}
                  </code>
                </div>
              )}
            </CardContent>
            <CardFooter>
              <Sheet>
                <SheetTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full border-purple-500/20 text-purple-100 hover:bg-purple-500/10"
                    onClick={() => openEditSheet(name)}
                  >
                    Edit Configuration
                  </Button>
                </SheetTrigger>
                <SheetContent className="bg-gradient-to-br from-black via-slate-900 to-purple-950 border-purple-500/20 overflow-y-auto">
                  <SheetHeader>
                    <SheetTitle className="text-purple-100">Edit {selectedServer}</SheetTitle>
                    <SheetDescription className="text-purple-200/60">
                      Modify the server configuration
                    </SheetDescription>
                  </SheetHeader>
                  {editingServer && selectedServer === name && (
                    <div className="space-y-4 mt-6">
                      <div className="space-y-2">
                        <Label className="text-purple-100">Description</Label>
                        <Input
                          value={editingServer.description}
                          onChange={(e) => setEditingServer({ ...editingServer, description: e.target.value })}
                          className="bg-black/20 border-purple-500/20 text-purple-100"
                        />
                      </div>

                      {editingServer.command !== undefined && (
                        <>
                          <div className="space-y-2">
                            <Label className="text-purple-100">Command</Label>
                            <Input
                              value={editingServer.command}
                              onChange={(e) => setEditingServer({ ...editingServer, command: e.target.value })}
                              className="bg-black/20 border-purple-500/20 text-purple-100"
                            />
                          </div>

                          <div className="space-y-2">
                            <Label className="text-purple-100">Arguments (JSON array)</Label>
                            <Textarea
                              value={JSON.stringify(editingServer.args, null, 2)}
                              onChange={(e) => {
                                try {
                                  setEditingServer({ ...editingServer, args: JSON.parse(e.target.value) });
                                } catch (err) {
                                  // Invalid JSON, don't update
                                }
                              }}
                              className="bg-black/20 border-purple-500/20 text-purple-100 font-mono text-sm"
                              rows={4}
                            />
                          </div>
                        </>
                      )}

                      {editingServer.url !== undefined && (
                        <div className="space-y-2">
                          <Label className="text-purple-100">URL</Label>
                          <Input
                            value={editingServer.url}
                            onChange={(e) => setEditingServer({ ...editingServer, url: e.target.value })}
                            className="bg-black/20 border-purple-500/20 text-purple-100"
                          />
                        </div>
                      )}

                      <div className="space-y-2">
                        <Label className="text-purple-100">Environment Variables (JSON)</Label>
                        <Textarea
                          value={JSON.stringify(editingServer.env || {}, null, 2)}
                          onChange={(e) => {
                            try {
                              setEditingServer({ ...editingServer, env: JSON.parse(e.target.value) });
                            } catch (err) {
                              // Invalid JSON
                            }
                          }}
                          className="bg-black/20 border-purple-500/20 text-purple-100 font-mono text-sm"
                          rows={4}
                        />
                      </div>

                      <Button
                        onClick={saveServerEdit}
                        className="w-full bg-purple-600 hover:bg-purple-700"
                      >
                        <Save className="mr-2 h-4 w-4" />
                        Save Changes
                      </Button>
                    </div>
                  )}
                </SheetContent>
              </Sheet>
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  );
}

