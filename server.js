import express from "express";
import fs from "node:fs";
import path from "node:path";
import { GoogleGenAI } from "@google/genai";

const app = express();
app.use(express.json({ limit: "50mb" }));

const PORT = process.env.PORT || 3000;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const DEFAULT_MODEL = process.env.GEMINI_IMAGE_MODEL || "gemini-3.1-flash-image";

const ai = new GoogleGenAI({
  apiKey: GEMINI_API_KEY
});

const imageDir = path.join(process.cwd(), "public", "images");
fs.mkdirSync(imageDir, { recursive: true });

app.use("/images", express.static(imageDir));

function getBaseUrl(req) {
  const proto = req.headers["x-forwarded-proto"] || req.protocol || "https";
  const host = req.headers["x-forwarded-host"] || req.headers.host;
  return `${proto}://${host}`;
}

function safeFilename(name) {
  const clean = String(name || `image-${Date.now()}.png`)
    .replace(/[^a-zA-Z0-9._-]/g, "_");

  if (path.extname(clean)) return clean;
  return `${clean}.png`;
}

function success(id, result) {
  return {
    jsonrpc: "2.0",
    id,
    result
  };
}

function fail(id, code, message) {
  return {
    jsonrpc: "2.0",
    id: id ?? null,
    error: {
      code,
      message
    }
  };
}

const tools = [
  {
    name: "ping",
    description: "Test if the MCP server is working.",
    inputSchema: {
      type: "object",
      properties: {},
      required: []
    }
  },
  {
    name: "generate_image",
    description: "Generate an image using Gemini Nano Banana image generation and return an image URL.",
    inputSchema: {
      type: "object",
      properties: {
        prompt: {
          type: "string",
          description: "The image prompt."
        },
        filename: {
          type: "string",
          description: "Optional file name, for example cat.png."
        },
        model: {
          type: "string",
          description: "Optional Gemini image model. Default is gemini-3.1-flash-image."
        },
        aspect_ratio: {
          type: "string",
          description: "Optional aspect ratio like 1:1, 9:16, 16:9, 4:5."
        },
        image_size: {
          type: "string",
          description: "Optional image size like 512, 1K, 2K, 4K."
        }
      },
      required: ["prompt"]
    }
  }
];

async function callTool(name, args, req) {
  if (name === "ping") {
    return {
      content: [
        {
          type: "text",
          text: "pong ✅ MCP server is working."
        }
      ]
    };
  }

  if (name !== "generate_image") {
    throw new Error(`Unknown tool: ${name}`);
  }

  if (!GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY is missing in Render Environment Variables.");
  }

  const prompt = args?.prompt;
  if (!prompt) {
    throw new Error("Prompt is required.");
  }

  const model = args?.model || DEFAULT_MODEL;
  const filename = safeFilename(args?.filename);
  const filePath = path.join(imageDir, filename);

  const config = {
    responseModalities: ["TEXT", "IMAGE"]
  };

  if (args?.aspect_ratio || args?.image_size) {
    config.responseFormat = {
      image: {}
    };

    if (args.aspect_ratio) {
      config.responseFormat.image.aspectRatio = args.aspect_ratio;
    }

    if (args.image_size) {
      config.responseFormat.image.imageSize = args.image_size;
    }
  }

  const response = await ai.models.generateContent({
    model,
    contents: prompt,
    config
  });

  const parts = response?.candidates?.[0]?.content?.parts || [];
  const textParts = [];

  for (const part of parts) {
    if (part.text) {
      textParts.push(part.text);
    }

    if (part.inlineData?.data) {
      const buffer = Buffer.from(part.inlineData.data, "base64");
      fs.writeFileSync(filePath, buffer);

      const imageUrl = `${getBaseUrl(req)}/images/${filename}`;

      return {
        content: [
          {
            type: "text",
            text:
              `Image generated successfully ✅\n\n` +
              `Model: ${model}\n` +
              `Image URL: ${imageUrl}\n\n` +
              `${textParts.join("\n")}`
          }
        ]
      };
    }
  }

  return {
    content: [
      {
        type: "text",
        text: textParts.join("\n") || "Gemini did not return an image."
      }
    ]
  };
}

async function handleRpc(message, req) {
  const id = message?.id;
  const method = message?.method;
  const params = message?.params || {};

  console.log("MCP request:", method);

  if (!method) {
    return fail(id, -32600, "Invalid Request");
  }

  if (method === "initialize") {
    return success(id, {
      protocolVersion: params.protocolVersion || "2025-06-18",
      capabilities: {
        tools: {}
      },
      serverInfo: {
        name: "gemini-image-mcp",
        version: "1.0.0"
      }
    });
  }

  if (method === "notifications/initialized") {
    return null;
  }

  if (method === "tools/list") {
    return success(id, {
      tools
    });
  }

  if (method === "tools/call") {
    try {
      const result = await callTool(params.name, params.arguments || {}, req);
      return success(id, result);
    } catch (error) {
      console.error("Tool error:", error);
      return fail(id, -32000, error.message || "Tool failed");
    }
  }

  return fail(id, -32601, `Method not found: ${method}`);
}

app.get("/", (req, res) => {
  res.send("Gemini Nano Banana MCP server is running. Use /mcp as the Claude MCP URL.");
});

app.get("/health", (req, res) => {
  res.json({ ok: true });
});

app.options("/mcp", (req, res) => {
  res.status(204).end();
});

app.get("/mcp", (req, res) => {
  res.status(405).set("Allow", "POST").send("MCP endpoint is ready. Use POST.");
});

app.post("/mcp", async (req, res) => {
  try {
    if (Array.isArray(req.body)) {
      const results = [];

      for (const msg of req.body) {
        const result = await handleRpc(msg, req);
        if (result) results.push(result);
      }

      if (results.length === 0) {
        res.status(202).end();
        return;
      }

      res.json(results);
      return;
    }

    const result = await handleRpc(req.body, req);

    if (!result) {
      res.status(202).end();
      return;
    }

    res.json(result);
  } catch (error) {
    console.error("Server error:", error);
    res.status(500).json(fail(null, -32603, "Internal server error"));
  }
});

app.listen(PORT, () => {
  console.log(`Gemini Image MCP server running on port ${PORT}`);
});
