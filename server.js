import express from "express";
import fs from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { GoogleGenAI } from "@google/genai";
import { z } from "zod";

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { isInitializeRequest } from "@modelcontextprotocol/sdk/types.js";

const app = express();
app.use(express.json({ limit: "50mb" }));

const PORT = process.env.PORT || 3000;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const DEFAULT_MODEL = process.env.GEMINI_IMAGE_MODEL || "gemini-3.1-flash-image";
const PUBLIC_BASE_URL = process.env.PUBLIC_BASE_URL || "";

if (!GEMINI_API_KEY) {
  console.warn("WARNING: GEMINI_API_KEY is missing.");
}

const ai = new GoogleGenAI({
  apiKey: GEMINI_API_KEY
});

const imageDir = path.join(process.cwd(), "public", "images");
fs.mkdirSync(imageDir, { recursive: true });

app.use("/images", express.static(imageDir));

app.get("/", (req, res) => {
  res.send("Gemini Nano Banana MCP server is running. Use /mcp as the MCP URL.");
});

app.get("/health", (req, res) => {
  res.json({ ok: true });
});

function safeFilename(name) {
  const clean = String(name || `image-${Date.now()}.png`)
    .replace(/[^a-zA-Z0-9._-]/g, "_");

  if (path.extname(clean)) return clean;
  return `${clean}.png`;
}

function createMcpServer() {
  const server = new McpServer({
    name: "gemini-image-mcp",
    version: "1.0.0"
  });

  server.tool(
    "generate_image",
    "Generate an image using Gemini Nano Banana image generation.",
    {
      prompt: z.string().describe("The image prompt to generate."),
      filename: z.string().optional().describe("Optional file name like cat.png"),
      model: z.string().optional().describe("Optional Gemini image model.")
    },
    async ({ prompt, filename, model }) => {
      const useModel = model || DEFAULT_MODEL;

      const response = await ai.models.generateContent({
        model: useModel,
        contents: prompt
      });

      const parts = response?.candidates?.[0]?.content?.parts || [];

      for (const part of parts) {
        if (part.inlineData?.data) {
          const base64 = part.inlineData.data;
          const mimeType = part.inlineData.mimeType || "image/png";
          const finalName = safeFilename(filename);
          const filePath = path.join(imageDir, finalName);

          fs.writeFileSync(filePath, Buffer.from(base64, "base64"));

          const imageUrl = PUBLIC_BASE_URL
            ? `${PUBLIC_BASE_URL.replace(/\/$/, "")}/images/${finalName}`
            : `/images/${finalName}`;

          return {
            content: [
              {
                type: "text",
                text: `Image generated successfully.\nModel: ${useModel}\nImage URL: ${imageUrl}`
              },
              {
                type: "image",
                data: base64,
                mimeType
              }
            ]
          };
        }
      }

      const textReply = parts
        .map((p) => p.text)
        .filter(Boolean)
        .join("\n");

      return {
        content: [
          {
            type: "text",
            text: textReply || "Gemini did not return an image."
          }
        ]
      };
    }
  );

  return server;
}

const transports = {};

app.post("/mcp", async (req, res) => {
  try {
    const sessionId = req.headers["mcp-session-id"];
    let transport;

    if (sessionId && transports[sessionId]) {
      transport = transports[sessionId];
    } else if (!sessionId && isInitializeRequest(req.body)) {
      transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: () => randomUUID(),
        onsessioninitialized: (newSessionId) => {
          transports[newSessionId] = transport;
        }
      });

      transport.onclose = () => {
        if (transport.sessionId) {
          delete transports[transport.sessionId];
        }
      };

      const server = createMcpServer();
      await server.connect(transport);
    } else {
      res.status(400).json({
        jsonrpc: "2.0",
        error: {
          code: -32000,
          message: "Bad Request: No valid session ID provided"
        },
        id: null
      });
      return;
    }

    await transport.handleRequest(req, res, req.body);
  } catch (error) {
    console.error(error);
    if (!res.headersSent) {
      res.status(500).json({
        jsonrpc: "2.0",
        error: {
          code: -32603,
          message: "Internal server error"
        },
        id: null
      });
    }
  }
});

app.get("/mcp", async (req, res) => {
  const sessionId = req.headers["mcp-session-id"];
  if (!sessionId || !transports[sessionId]) {
    res.status(400).send("Invalid or missing session ID");
    return;
  }

  const transport = transports[sessionId];
  await transport.handleRequest(req, res);
});

app.delete("/mcp", async (req, res) => {
  const sessionId = req.headers["mcp-session-id"];
  if (!sessionId || !transports[sessionId]) {
    res.status(400).send("Invalid or missing session ID");
    return;
  }

  const transport = transports[sessionId];
  await transport.handleRequest(req, res);
});

app.listen(PORT, () => {
  console.log(`Gemini Image MCP server running on port ${PORT}`);
});
