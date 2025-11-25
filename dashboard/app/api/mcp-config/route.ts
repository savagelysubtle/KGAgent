import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

const MCP_CONFIG_PATH = path.join(process.cwd(), 'mcp.json');

export async function GET() {
  try {
    const data = await fs.readFile(MCP_CONFIG_PATH, 'utf-8');
    return NextResponse.json(JSON.parse(data));
  } catch (error) {
    console.error('Failed to read mcp.json:', error);
    return NextResponse.json(
      { error: 'Failed to read configuration' },
      { status: 500 }
    );
  }
}

export async function POST(req: Request) {
  try {
    const body = await req.json();
    await fs.writeFile(MCP_CONFIG_PATH, JSON.stringify(body, null, 2));
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Failed to write mcp.json:', error);
    return NextResponse.json(
      { error: 'Failed to save configuration' },
      { status: 500 }
    );
  }
}

