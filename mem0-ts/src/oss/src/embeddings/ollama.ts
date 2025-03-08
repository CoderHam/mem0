import { Ollama } from "ollama";
import { Embedder } from "./base";
import { EmbeddingConfig } from "../types";

export class OllamaEmbedder implements Embedder {
  private ollama: Ollama;
  private model: string;

  constructor(config: EmbeddingConfig) {
    this.ollama = new Ollama({
      host: config.url || "http://localhost:11434",
    });
    this.model = config.model || "nomic-embed-text:v1.5";
  }

  async embed(text: string): Promise<number[]> {
    const response = await this.ollama.embeddings({
      model: this.model,
      prompt: text,
    });
    return response.embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const response = await Promise.all(texts.map(text => this.embed(text)));
    return response;
  }
}
