import { Ollama } from "ollama"
import { LLM, LLMResponse } from "./base";
import { LLMConfig, Message } from "../types";

export class OllamaLLM implements LLM {
  private ollama: Ollama;
  private model: string;

  constructor(config: LLMConfig) {
    this.ollama = new Ollama({
      host: config.config?.url || "http://localhost:11434",
    });
    this.model = config.model || "llama3.1:8b";
  }

  async generateResponse(
    messages: Message[],
    responseFormat?: { type: string },
    tools?: any[],
  ): Promise<string | LLMResponse> {
    const completion = await this.ollama.chat({
      model: this.model,
      messages: messages.map((msg) => {
        const role = msg.role as "system" | "user" | "assistant";
        return {
          role,
          content:
            typeof msg.content === "string"
              ? msg.content
              : JSON.stringify(msg.content),
        };
      }),
      ...(tools && { tools, tool_choice: "auto" }),
    });

    const response = completion.message;

    if (response.tool_calls) {
      return {
        content: response.content || "",
        role: response.role,
        toolCalls: response.tool_calls.map((call) => ({
          name: call.function.name,
          arguments: JSON.stringify(call.function.arguments),
        })),
      };
    }

    return response.content || "";
  }

  async generateChat(messages: Message[]): Promise<LLMResponse> {
    const completion = await this.ollama.chat({
      messages: messages.map((msg) => {
        const role = msg.role as "system" | "user" | "assistant";
        return {
          role,
          content:
            typeof msg.content === "string"
              ? msg.content
              : JSON.stringify(msg.content),
        };
      }),
      model: this.model,
    });
    const response = completion.message;
    return {
      content: response.content || "",
      role: response.role,
    };
  }
}
