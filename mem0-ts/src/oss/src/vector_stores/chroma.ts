import { ChromaClient, Collection } from 'chromadb';
import { VectorStore } from "./base";
import { SearchFilters, VectorStoreConfig, VectorStoreResult } from "../types";
import * as fs from "fs";

interface ChromaConfig extends VectorStoreConfig {
  client?: ChromaClient;
  host?: string;
  port?: number;
  path?: string;
  url?: string;
  apiKey?: string;
  onDisk?: boolean;
  collectionName: string;
  embeddingModelDims: number;
}

type DistanceType = "Cosine" | "Euclid" | "Dot";

interface ChromaPoint {
  id: string | number;
  vector: { name: string; vector: number[] };
  payload?: Record<string, unknown> | { [key: string]: unknown } | null;
  shard_key?: string;
  version?: number;
}

interface ChromaScoredPoint extends ChromaPoint {
  score: number;
  version: number;
}

interface ChromaNamedVector {
  name: string;
  vector: number[];
}

interface ChromaSearchRequest {
  vector: { name: string; vector: number[] };
  limit?: number;
  offset?: number;
  filter?: ChromaFilter;
}

interface ChromaFilter {
  must?: ChromaCondition[];
  must_not?: ChromaCondition[];
  should?: ChromaCondition[];
}

interface ChromaCondition {
  key: string;
  match?: { value: any };
  range?: { gte?: number; gt?: number; lte?: number; lt?: number };
}

interface ChromaVectorParams {
  size: number;
  distance: "Cosine" | "Euclid" | "Dot" | "Manhattan";
  on_disk?: boolean;
}

interface ChromaCollectionInfo {
  config?: {
    params?: {
      vectors?: {
        size: number;
        distance: "Cosine" | "Euclid" | "Dot" | "Manhattan";
        on_disk?: boolean;
      };
    };
  };
}

export class Chroma implements VectorStore {
  private client: ChromaClient;
  private readonly collectionName: string;

  constructor(config: ChromaConfig) {
    if (config.client) {
      this.client = config.client;
    } else {
      const params: Record<string, any> = {};
      if (config.apiKey) {
        params.apiKey = config.apiKey;
      }
      if (config.url) {
        params.url = config.url;
      }
      if (config.host && config.port) {
        params.host = config.host;
        params.port = config.port;
      }
      if (!Object.keys(params).length) {
        params.path = config.path;
        if (!config.onDisk && config.path) {
          if (
            fs.existsSync(config.path) &&
            fs.statSync(config.path).isDirectory()
          ) {
            fs.rmSync(config.path, { recursive: true });
          }
        }
      }

      this.client = new ChromaClient(params);
    }

    this.collectionName = config.collectionName;
    this.createCol(config.embeddingModelDims, config.onDisk || false);
  }

  private async createCol(
    vectorSize: number,
    onDisk: boolean,
    distance: DistanceType = "Cosine",
  ): Promise<void> {
    try {
      // Check if collection exists
      const collections = await this.client.listCollections();
      const exists = collections.some(col => col === this.collectionName);

      if (!exists) {
        const vectorParams: ChromaVectorParams = {
          size: vectorSize,
          distance: distance as "Cosine" | "Euclid" | "Dot" | "Manhattan",
          on_disk: onDisk,
        };

        try {
          await this.client.createCollection(this.collectionName, {
            vectors: vectorParams,
          });
        } catch (error: any) {
          // Handle case where collection was created between our check and create
          if (error?.status === 409) {
            // Collection already exists - verify it has the correct configuration
            const collectionInfo = (await this.client.getCollection(
              this.collectionName,
            )) as ChromaCollectionInfo;
            const vectorConfig = collectionInfo.config?.params?.vectors;

            if (!vectorConfig || vectorConfig.size !== vectorSize) {
              throw new Error(
                `Collection ${this.collectionName} exists but has wrong configuration. ` +
                  `Expected vector size: ${vectorSize}, got: ${vectorConfig?.size}`,
              );
            }
            // Collection exists with correct configuration - we can proceed
            return;
          }
          throw error;
        }
      }
    } catch (error) {
      if (error instanceof Error) {
        console.error("Error creating/verifying collection:", error.message);
      } else {
        console.error("Error creating/verifying collection:", error);
      }
      throw error;
    }
  }

  private createFilter(filters?: SearchFilters): ChromaFilter | undefined {
    if (!filters) return undefined;

    const conditions: ChromaCondition[] = [];
    for (const [key, value] of Object.entries(filters)) {
      if (
        typeof value === "object" &&
        value !== null &&
        "gte" in value &&
        "lte" in value
      ) {
        conditions.push({
          key,
          range: {
            gte: value.gte,
            lte: value.lte,
          },
        });
      } else {
        conditions.push({
          key,
          match: {
            value,
          },
        });
      }
    }

    return conditions.length ? { must: conditions } : undefined;
  }

  async insert(
    vectors: number[][],
    ids: string[],
    payloads: Record<string, any>[],
  ): Promise<void> {
    const points = vectors.map((vector, idx) => ({
      id: ids[idx],
      vector: vector,
      payload: payloads[idx] || {},
    }));

    await this.client.upsert(this.collectionName, {
      points,
    });
  }

  async search(
    query: number[],
    limit: number = 5,
    filters?: SearchFilters,
  ): Promise<VectorStoreResult[]> {
    const queryFilter = this.createFilter(filters);
    const results = await this.client.search(this.collectionName, {
      vector: query,
      filter: queryFilter,
      limit,
    });

    return results.map((hit) => ({
      id: String(hit.id),
      payload: (hit.payload as Record<string, any>) || {},
      score: hit.score,
    }));
  }

  async get(vectorId: string): Promise<VectorStoreResult | null> {
    const results = await this.client.retrieve(this.collectionName, {
      ids: [vectorId],
      with_payload: true,
    });

    if (!results.length) return null;

    return {
      id: vectorId,
      payload: results[0].payload || {},
    };
  }

  async update(
    vectorId: string,
    vector: number[],
    payload: Record<string, any>,
  ): Promise<void> {
    const point = {
      id: vectorId,
      vector: vector,
      payload,
    };

    await this.client.upsert(this.collectionName, {
      points: [point],
    });
  }

  async delete(vectorId: string): Promise<void> {
    await this.client.delete(this.collectionName, {
      points: [vectorId],
    });
  }

  async deleteCol(): Promise<void> {
    await this.client.deleteCollection(this.collectionName);
  }

  async list(
    filters?: SearchFilters,
    limit: number = 100,
  ): Promise<[VectorStoreResult[], number]> {
    const scrollRequest = {
      limit,
      filter: this.createFilter(filters),
      with_payload: true,
      with_vectors: false,
    };

    const response = await this.client.scroll(
      this.collectionName,
      scrollRequest,
    );

    const results = response.points.map((point) => ({
      id: String(point.id),
      payload: (point.payload as Record<string, any>) || {},
    }));

    return [results, response.points.length];
  }
}
