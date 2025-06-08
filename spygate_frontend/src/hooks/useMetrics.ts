import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { metricsService } from "@/api/services/metrics";

export function useUserMetrics() {
  return useQuery({
    queryKey: ["userMetrics"],
    queryFn: () => metricsService.getUserMetrics(),
  });
}

export function useFeatureUsage(featureName?: string) {
  return useQuery({
    queryKey: ["featureUsage", featureName],
    queryFn: () => metricsService.getFeatureUsage(featureName),
  });
}

export function useTrackFeatureUsage() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (featureName: string) =>
      metricsService.trackFeatureUsage(featureName),
    onSuccess: (_, featureName) => {
      queryClient.invalidateQueries({
        queryKey: ["featureUsage", featureName],
      });
      queryClient.invalidateQueries({ queryKey: ["featureUsage"] });
    },
  });
}

export function usePerformanceMetrics(params?: {
  metric_name?: string;
  start_date?: string;
  end_date?: string;
}) {
  return useQuery({
    queryKey: ["performanceMetrics", params],
    queryFn: () => metricsService.getPerformanceMetrics(params),
  });
}

export function useRecordPerformanceMetric() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: {
      metric_name: string;
      value: number;
      metadata?: Record<string, any>;
    }) => metricsService.recordPerformanceMetric(data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: [
          "performanceMetrics",
          { metric_name: variables.metric_name },
        ],
      });
      queryClient.invalidateQueries({ queryKey: ["performanceMetrics"] });
    },
  });
}

// Utility hook to automatically track feature usage
export function useFeature(featureName: string) {
  const { mutate: trackUsage } = useTrackFeatureUsage();

  return {
    trackUsage: () => trackUsage(featureName),
  };
}
