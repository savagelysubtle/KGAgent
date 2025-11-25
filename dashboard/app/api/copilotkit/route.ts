import {
  CopilotRuntime,
  OpenAIAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import OpenAI from "openai";
import { NextRequest } from "next/server";

// Create OpenAI client configured for LM Studio
const openai = new OpenAI({
  apiKey: process.env.LLM_API_KEY || "lm-studio",
  baseURL: process.env.LLM_BASE_URL || "http://localhost:1234/v1",
});

// Create the service adapter
const serviceAdapter = new OpenAIAdapter({ openai });

// Create the CopilotKit runtime
const runtime = new CopilotRuntime();

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });

  return handleRequest(req);
};
