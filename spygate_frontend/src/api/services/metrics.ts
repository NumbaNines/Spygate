import { apiClient } from "../client";

export interface UserMetrics {
  id: number;
  user: number;
  total_analyses: number;
  total_videos: number;
  total_storage_used: number;
  last_activity: string;
  created_at: string;
  updated_at: string;
}

export interface FeatureUsage {
  id: number;
  user: number;
  feature_name: string;
  usage_count: number;
  last_used: string;
  created_at: string;
  updated_at: string;
}

export interface PerformanceMetric {
  id: number;
  user: number;
  metric_name: string;
  value: number;
  timestamp: string;
  metadata: Record<string, any>;
}

export const metricsService = {
  // Get user metrics
  getUserMetrics: async () => {
    const response = await apiClient.get("/metrics/");
    return response.data;
  },

  // Get feature usage statistics
  getFeatureUsage: async (featureName?: string) => {
    const response = await apiClient.get("/feature-usage/", {
      params: { feature_name: featureName },
    });
    return response.data;
  },

  // Track feature usage
  trackFeatureUsage: async (featureName: string) => {
    const response = await apiClient.post("/feature-usage/", {
      feature_name: featureName,
    });
    return response.data;
  },

  // Get performance metrics
  getPerformanceMetrics: async (params?: {
    metric_name?: string;
    start_date?: string;
    end_date?: string;
  }) => {
    const response = await apiClient.get("/performance/", { params });
    return response.data;
  },

  // Record performance metric
  recordPerformanceMetric: async (data: {
    metric_name: string;
    value: number;
    metadata?: Record<string, any>;
  }) => {
    const response = await apiClient.post("/performance/", data);
    return response.data;
  },
};
