import { apiClient } from "../client";

export interface GameAnalysis {
  id: number;
  title: string;
  description: string;
  video_file: string;
  processing_status: "pending" | "processing" | "completed" | "failed";
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  error_message: string;
  metadata: Record<string, any>;
  formations: Formation[];
  plays: Play[];
  situations: Situation[];
}

export interface Formation {
  id: number;
  name: string;
  formation_type: "offense" | "defense";
  timestamp: number;
  confidence_score: number;
  player_positions: Record<string, any>;
  metadata: Record<string, any>;
}

export interface Play {
  id: number;
  name: string;
  play_type: "run" | "pass" | "special";
  start_time: number;
  end_time: number;
  success_rate: number | null;
  yards_gained: number | null;
  player_routes: Record<string, any>;
  metadata: Record<string, any>;
  formation?: Formation;
  situation?: Situation;
}

export interface Situation {
  id: number;
  down: number;
  distance: number;
  field_position: number;
  score_differential: number;
  time_remaining: number;
  quarter: number;
  is_red_zone: boolean;
  metadata: Record<string, any>;
}

export const analysisService = {
  // Get all analyses
  getAnalyses: async (params?: Record<string, any>) => {
    const response = await apiClient.get("/analysis/", { params });
    return response.data;
  },

  // Get a single analysis by ID
  getAnalysis: async (id: number) => {
    const response = await apiClient.get(`/analysis/${id}/`);
    return response.data;
  },

  // Create a new analysis
  createAnalysis: async (data: FormData) => {
    const response = await apiClient.post("/analysis/", data, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return response.data;
  },

  // Update an analysis
  updateAnalysis: async (id: number, data: Partial<GameAnalysis>) => {
    const response = await apiClient.patch(`/analysis/${id}/`, data);
    return response.data;
  },

  // Delete an analysis
  deleteAnalysis: async (id: number) => {
    await apiClient.delete(`/analysis/${id}/`);
  },

  // Reprocess an analysis
  reprocessAnalysis: async (id: number) => {
    const response = await apiClient.post(`/analysis/${id}/reprocess/`);
    return response.data;
  },

  // Get formations for an analysis
  getFormations: async (analysisId: number) => {
    const response = await apiClient.get(`/formations/`, {
      params: { analysis: analysisId },
    });
    return response.data;
  },

  // Get plays for an analysis
  getPlays: async (analysisId: number) => {
    const response = await apiClient.get(`/plays/`, {
      params: { analysis: analysisId },
    });
    return response.data;
  },

  // Get situations for an analysis
  getSituations: async (analysisId: number) => {
    const response = await apiClient.get(`/situations/`, {
      params: { analysis: analysisId },
    });
    return response.data;
  },
};
